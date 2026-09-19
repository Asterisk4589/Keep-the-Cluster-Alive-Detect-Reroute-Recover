"""
Implement this interface. Your agent will be instantiated fresh for each
evaluation episode and driven by our harness like:

    agent = YourAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
    obs = env.reset()
    agent.reset()
    for t in range(episode_length):
        actions = agent.act(obs)          # dict {task_id: node_id}
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)
        if done:
            break

Rules:
  - obs["nodes"] gives you noisy telemetry per node (heartbeat_ok, latency_ms,
    error_rate, queue_len, capacity). It never tells you the true health
    state directly.
  - obs["tasks"] lists every task currently pending or in-flight, with its
    current node (or None if unassigned), remaining duration, and deadline.
  - act() must return a dict {task_id: node_id} for any subset of the task
    ids in obs["tasks"]. Omitted tasks stay exactly as they are (pending
    stays pending; assigned stays on its current node).
  - Rerouting a task to a DIFFERENT node than it currently has RESTARTS that
    task's remaining duration (cold restart cost) -- don't reroute unless
    you have good reason to believe the current node is unhealthy.
  - node_id must be in [0, n_nodes). Assigning past a node's capacity is
    silently rejected by the environment (task stays as it was).
  - Keep act()/update() fast -- there's a wall-clock budget per episode
    during evaluation (see eval_harness.py TIME_BUDGET_SECONDS).
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, n_nodes: int, node_capacity: int):
        self.n_nodes = n_nodes
        self.node_capacity = node_capacity

    @abstractmethod
    def reset(self) -> None:
        """Called once at the start of each episode. Clear any per-episode state."""
        raise NotImplementedError

    @abstractmethod
    def act(self, obs: dict) -> dict:
        """Return {task_id: node_id} for any tasks you want to (re)assign this step."""
        raise NotImplementedError

    @abstractmethod
    def update(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        """Called after each step with the resulting obs/reward. Use for any learning/bookkeeping."""
        raise NotImplementedError
