import sys, random
import numpy as np
import bench
from env.cluster_env import ClusterEnv
from baseline_agent import RoundRobinNoHealthCheck
from agent import MyAgent

class Perturbed(ClusterEnv):
    mode = None
    def _make_obs(self):
        o = super()._make_obs(); m = self.mode; r = self.rng
        for n in o["nodes"]:
            if m == "lat_seconds" and n["latency_ms"] is not None: n["latency_ms"] *= 0.001
            if m == "lat_x10" and n["latency_ms"] is not None: n["latency_ms"] *= 10
            if m == "noise_x3":
                if n["latency_ms"] is not None: n["latency_ms"] = max(1.0, 50 + (n["latency_ms"]-50)*1.0 + float(r.normal(0, 10)))
                n["error_rate"] = float(np.clip(n["error_rate"] + r.normal(0, 0.03), 0, 1))
            if m == "hb_flaky" and r.random() < 0.10: n["heartbeat_ok"] = False
            if m == "err_none" and r.random() < 0.05: n["error_rate"] = None
            if m == "lat_none" and r.random() < 0.10: n["latency_ms"] = None
        if m == "drop_node":
            o["nodes"] = [n for n in o["nodes"] if r.random() > 0.10]
        return o

def run(agent_cls, mode, seed, exc_counter=None):
    env = Perturbed(n_nodes=6, node_capacity=4, arrival_rate=2.0, episode_length=400, seed=seed); env.mode = mode
    ag = agent_cls(env.n_nodes, env.node_capacity)
    obs = env.reset(); ag.reset(); tot = 0.0
    for _ in range(400):
        a = ag.act(obs); obs, r, d, i = env.step(a); tot += r; ag.update(obs, r, d, i)
        if d: break
    return tot

modes = [None, "lat_seconds", "lat_x10", "noise_x3", "hb_flaky", "err_none", "lat_none", "drop_node"]
S = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print("mode".ljust(12), "baseline   agent")
for m in modes:
    b = np.mean([run(RoundRobinNoHealthCheck, m, s) for s in range(S)])
    v = np.mean([run(MyAgent, m, s) for s in range(S)])
    print(str(m).ljust(12), f"{b:8.1f} {v:7.1f}", flush=True)
