"""
Core environment engine for PS3: "Detect node failures, identify affected
tasks, reroute to healthy nodes."

Framing: a simulated cluster of N nodes. Tasks arrive continuously and must
be assigned to nodes to run. Each node's TRUE health (healthy / degraded /
down) evolves hidden, as an independent Markov chain per node. The agent
never sees true health directly -- only noisy per-node telemetry (heartbeat
ack, latency, error rate) that it must use to infer which nodes are actually
trustworthy, and reroute tasks stuck on bad nodes before they miss deadline.

This same class is used by both:
  - sandbox_env.py   (participant-facing, failure log visible in debug mode)
  - eval_harness.py  (organizer-only, failure timing/config hidden)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

HEALTHY, DEGRADED, DOWN = "healthy", "degraded", "down"
STATES = [HEALTHY, DEGRADED, DOWN]


@dataclass
class Task:
    id: int
    arrival_step: int
    duration_total: float
    deadline: int
    duration_remaining: float
    node: Optional[int] = None


class ClusterEnv:
    """
    obs (returned by reset()/step()) is a dict:
        {
          "step": int,
          "nodes": [
              {"node_id": int, "heartbeat_ok": bool, "latency_ms": float or None,
               "error_rate": float, "queue_len": int, "capacity": int}
              ... one per node
          ],
          "tasks": [
              {"task_id": int, "node": int or None, "duration_remaining": float,
               "deadline": int, "is_new": bool}
              ... every task currently pending or in-flight (not yet
                  completed/failed)
          ],
        }

    Agent's action (passed to step()) is a dict {task_id: node_id} for any
    subset of the tasks currently listed in obs["tasks"] that it wants to
    (re)assign this step. Tasks it omits stay wherever they are (pending
    stays pending; assigned stays on its current node).

    Rerouting a task that was already assigned to a DIFFERENT node RESTARTS
    its duration_remaining (simulates a cold restart on the new node) --
    this is deliberate: it makes needless rerouting costly, so an agent
    should only reroute when it actually believes the current node is bad.

    info dict always contains "step". In expose_health=True (debug) mode it
    additionally contains "node_true_states": List[str] and
    "failed_this_step": List[int] (node ids that changed state this step).
    Real evaluation runs always use expose_health=False.
    """

    def __init__(
        self,
        n_nodes: int = 6,
        node_capacity: int = 4,
        arrival_rate: float = 2.0,
        duration_range=(3, 8),
        slack_range=(4, 10),
        health_transition: Optional[np.ndarray] = None,
        episode_length: int = 400,
        seed: Optional[int] = None,
        expose_health: bool = False,
    ):
        self.n_nodes = n_nodes
        self.node_capacity = node_capacity
        self.arrival_rate = arrival_rate
        self.duration_range = duration_range
        self.slack_range = slack_range
        self.episode_length = episode_length
        self.expose_health = expose_health
        self.rng = np.random.default_rng(seed)

        # Shared 3-state hidden Markov chain per node: healthy <-> degraded
        # <-> down, sticky, with a path back to healthy (transient failures
        # recover; this isn't "permanently break and never fix").
        if health_transition is None:
            self.health_transition = np.array([
                [0.985, 0.012, 0.003],   # healthy   -> healthy/degraded/down
                [0.10,  0.85,  0.05 ],   # degraded  -> healthy/degraded/down
                [0.02,  0.03,  0.95 ],   # down      -> healthy/degraded/down
            ])
        else:
            self.health_transition = health_transition

        self._t = 0
        self._node_states: List[str] = [HEALTHY] * n_nodes
        self._node_state_history: List[List[str]] = []
        self._failure_events: List[tuple] = []  # (step, node_id, new_state)
        self._tasks: Dict[int, Task] = {}
        self._next_task_id = 0
        self._completed_count = 0
        self._failed_count = 0

    def reset(self):
        self._t = 0
        self._node_states = [HEALTHY] * self.n_nodes
        self._node_state_history = [list(self._node_states)]
        self._failure_events = []
        self._tasks = {}
        self._next_task_id = 0
        self._completed_count = 0
        self._failed_count = 0
        self._spawn_tasks()
        return self._make_obs()

    def step(self, actions: Dict[int, int]):
        assert self._t < self.episode_length, "step() called after episode end; call reset()"

        # 1. hidden health transitions, per node independently
        changed_nodes = []
        new_states = []
        for i, state in enumerate(self._node_states):
            probs = self.health_transition[STATES.index(state)]
            new_state = str(self.rng.choice(STATES, p=probs))
            new_states.append(new_state)
            if new_state != state:
                changed_nodes.append(i)
                self._failure_events.append((self._t, i, new_state))
        self._node_states = new_states
        self._node_state_history.append(list(self._node_states))

        # 2. apply agent's (re)assignment actions, respecting capacity
        node_load = self._current_load()
        for task_id, node_id in actions.items():
            if task_id not in self._tasks:
                continue
            assert 0 <= node_id < self.n_nodes, f"invalid node_id {node_id}"
            task = self._tasks[task_id]
            if node_load[node_id] >= self.node_capacity and task.node != node_id:
                continue  # capacity full -- assignment rejected, task stays as-is
            if task.node != node_id:
                if task.node is not None:
                    node_load[task.node] -= 1
                task.node = node_id
                task.duration_remaining = task.duration_total  # cold restart
                node_load[node_id] += 1

        # 3. progress assigned tasks according to their node's TRUE state
        reward = 0.0
        for task in list(self._tasks.values()):
            if task.node is None:
                reward -= 0.01  # small holding cost for leaving work unassigned
                continue
            state = self._node_states[task.node]
            if state == HEALTHY:
                task.duration_remaining -= 1.0
            elif state == DEGRADED:
                task.duration_remaining -= 0.5 if self.rng.random() < 0.5 else 0.0
            # DOWN: frozen, no progress at all

        # 4. resolve completions and deadline misses
        for task_id, task in list(self._tasks.items()):
            if task.duration_remaining <= 0:
                reward += 1.0
                self._completed_count += 1
                del self._tasks[task_id]
            elif self._t >= task.deadline:
                reward -= 1.0
                self._failed_count += 1
                del self._tasks[task_id]

        self._t += 1
        self._spawn_tasks()

        done = self._t >= self.episode_length
        info = {"step": self._t}
        if self.expose_health:
            info["node_true_states"] = list(self._node_states)
            info["failed_this_step"] = changed_nodes

        return self._make_obs(), reward, done, info

    # --- internal helpers ---

    def _spawn_tasks(self):
        n_new = self.rng.poisson(self.arrival_rate)
        for _ in range(n_new):
            dur = float(self.rng.integers(self.duration_range[0], self.duration_range[1] + 1))
            slack = int(self.rng.integers(self.slack_range[0], self.slack_range[1] + 1))
            task = Task(
                id=self._next_task_id,
                arrival_step=self._t,
                duration_total=dur,
                duration_remaining=dur,
                deadline=self._t + int(dur) + slack,
            )
            self._tasks[task.id] = task
            self._next_task_id += 1

    def _current_load(self) -> List[int]:
        load = [0] * self.n_nodes
        for task in self._tasks.values():
            if task.node is not None:
                load[task.node] += 1
        return load

    def _make_obs(self):
        nodes_obs = []
        for i, state in enumerate(self._node_states):
            if state == HEALTHY:
                hb = self.rng.random() < 0.99
                latency = float(self.rng.normal(50, 5))
                err = float(np.clip(self.rng.normal(0.02, 0.01), 0, 1))
            elif state == DEGRADED:
                hb = self.rng.random() < 0.85
                latency = float(self.rng.normal(300, 60))
                err = float(np.clip(self.rng.normal(0.25, 0.06), 0, 1))
            else:  # DOWN
                hb = self.rng.random() < 0.05
                latency = None
                err = float(np.clip(self.rng.normal(0.95, 0.04), 0, 1))
            nodes_obs.append({
                "node_id": i,
                "heartbeat_ok": bool(hb),
                "latency_ms": latency,
                "error_rate": err,
                "queue_len": self._current_load()[i],
                "capacity": self.node_capacity,
            })

        tasks_obs = []
        for task in self._tasks.values():
            tasks_obs.append({
                "task_id": task.id,
                "node": task.node,
                "duration_remaining": task.duration_remaining,
                "deadline": task.deadline,
                "is_new": task.arrival_step == self._t,
            })

        return {"step": self._t, "nodes": nodes_obs, "tasks": tasks_obs}

    # --- organizer/debug-only helpers, not part of the agent-facing API ---

    def get_episode_log(self) -> dict:
        """Ground truth for scoring. Never expose this to an agent."""
        return {
            "node_state_history": list(self._node_state_history),
            "failure_events": list(self._failure_events),
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
        }
