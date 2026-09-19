"""Review 1: Basic health-aware scheduler.

Goal: build a working scheduler that reads telemetry and assigns NEW tasks to
reasonable nodes. It intentionally does not migrate in-flight work yet.
"""
from agent_interface import BaseAgent
import math

class MyAgent(BaseAgent):
    def __init__(self, n_nodes, node_capacity):
        super().__init__(n_nodes, node_capacity)
        self.reset()

    def reset(self):
        self.step_no = 0
        self.health = {}

    def _num(self, x, default):
        try:
            x = float(x)
            return x if math.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    def _update_health(self, obs):
        nodes = obs.get("nodes", [])
        lats = [self._num(n.get("latency_ms"), 0.0) for n in nodes]
        lats = [x for x in lats if x > 0]
        median_lat = sorted(lats)[len(lats)//2] if lats else 1.0
        for n in nodes:
            nid = int(n["node_id"])
            hb = bool(n.get("heartbeat_ok", False))
            lat = self._num(n.get("latency_ms"), median_lat)
            err = max(0.0, min(1.0, self._num(n.get("error_rate"), 0.0)))
            queue = max(0, int(self._num(n.get("queue_len"), 0)))
            # 0 = bad, 1 = healthy. This is deliberately simple for Review 1.
            score = 1.0
            if not hb:
                score -= 0.55
            score -= min(0.30, err * 0.60)
            if median_lat > 0 and lat > 1.5 * median_lat:
                score -= 0.20
            score -= min(0.15, queue * 0.03)
            self.health[nid] = max(0.0, min(1.0, score))

    def act(self, obs):
        try:
            self._update_health(obs)
            loads = {int(n["node_id"]): 0 for n in obs.get("nodes", [])}
            for t in obs.get("tasks", []):
                if t.get("node") is not None:
                    loads[int(t["node"])] = loads.get(int(t["node"]), 0) + 1
            actions = {}
            pending = [t for t in obs.get("tasks", []) if t.get("node") is None]
            pending.sort(key=lambda t: int(t.get("deadline", 10**9)))
            for t in pending:
                candidates = []
                for n in obs.get("nodes", []):
                    nid = int(n["node_id"])
                    cap = max(1, int(n.get("capacity", self.node_capacity)))
                    if loads.get(nid, 0) < cap and self.health.get(nid, 0.0) >= 0.45:
                        candidates.append(nid)
                if not candidates:
                    candidates = [int(n["node_id"]) for n in obs.get("nodes", [])
                                  if loads.get(int(n["node_id"]), 0) < max(1, int(n.get("capacity", self.node_capacity)))]
                if candidates:
                    nid = min(candidates, key=lambda x: (loads.get(x, 0), -self.health.get(x, 0.0)))
                    actions[int(t["task_id"])] = nid
                    loads[nid] = loads.get(nid, 0) + 1
            return actions
        except Exception:
            return {}

    def update(self, obs, reward, done, info):
        self.step_no = int(obs.get("step", self.step_no + 1))
