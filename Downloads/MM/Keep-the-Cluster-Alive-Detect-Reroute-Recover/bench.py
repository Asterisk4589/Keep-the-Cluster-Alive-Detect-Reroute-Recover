"""Independent benchmark harness (reviewer-written). Uses env internals ONLY for measurement/oracle."""
import time, sys, json, statistics as st
import numpy as np
from env.cluster_env import ClusterEnv, HEALTHY, DEGRADED, DOWN
from baseline_agent import RoundRobinNoHealthCheck
from agent_interface import BaseAgent
from agent import MyAgent

class Oracle(BaseAgent):
    """Upper-ish bound: reads true state. Assigns to healthiest free node, evacuates tasks on DOWN nodes."""
    env = None
    def reset(self): pass
    def update(self, *a, **k): pass
    def act(self, obs):
        S = self.env._node_states
        load = [0]*self.n_nodes
        for t in obs["tasks"]:
            if t["node"] is not None: load[t["node"]] += 1
        rank = {HEALTHY:0, DEGRADED:1, DOWN:2}
        acts = {}
        tasks = sorted(obs["tasks"], key=lambda t: t["deadline"])
        for t in tasks:
            src = t["node"]
            if src is not None and S[src] == HEALTHY: continue
            if src is not None and S[src] == DEGRADED and t["duration_remaining"] < 2: continue
            cands = [n for n in range(self.n_nodes) if load[n] < self.node_capacity and S[n] == HEALTHY]
            if not cands:
                if src is not None: continue
                cands = [n for n in range(self.n_nodes) if load[n] < self.node_capacity and S[n] != DOWN]
            if not cands: continue
            n = min(cands, key=lambda n: load[n])
            acts[t["task_id"]] = n; load[n] += 1
            if src is not None: load[src] -= 1
        return acts

def make_env(cfg, seed):
    kw = dict(n_nodes=6, node_capacity=4, arrival_rate=2.0, duration_range=(3,8), slack_range=(4,10), episode_length=400, seed=seed)
    kw.update(cfg)
    return ClusterEnv(**kw)

def run_episode(agent_cls, cfg, seed):
    env = make_env(cfg, seed)
    agent = agent_cls(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
    if isinstance(agent, Oracle): agent.env = env
    obs = env.reset(); agent.reset()
    total = 0.0; reroutes = 0; churn = 0; to_down = 0; to_bad = 0
    a_time = 0.0
    det_lat = []; pending_onset = {}; fq_steps = 0; healthy_node_steps = 0
    max_act = 0.0
    for t in range(env.episode_length):
        cur = {x["task_id"]: x["node"] for x in obs["tasks"]}
        t0 = time.perf_counter()
        actions = agent.act(obs)
        dt = time.perf_counter() - t0; a_time += dt; max_act = max(max_act, dt)
        pre = list(env._node_states)
        obs, r, done, info = env.step(actions)
        post = env._node_states
        total += r
        for tid, n in actions.items():
            src = cur.get(tid)
            if src is None:
                if post[n] == DOWN: to_down += 1
                if post[n] != HEALTHY: to_bad += 1
            elif src != n:
                reroutes += 1
                if post[src] == HEALTHY: churn += 1
        t1 = time.perf_counter()
        agent.update(obs, r, done, info); a_time += time.perf_counter() - t1
        # detection latency (MyAgent only): steps from DOWN onset until node state is SUSPECT/QUARANTINED
        if hasattr(agent, "nodes") and agent.nodes:
            for i in range(env.n_nodes):
                if post[i] == DOWN and pre[i] != DOWN: pending_onset[i] = t
                if post[i] != DOWN and i in pending_onset: pending_onset.pop(i)
                if i in pending_onset and agent.nodes[i].state in ("SUSPECT","QUARANTINED","PROBING"):
                    det_lat.append(t - pending_onset.pop(i))
                if post[i] == HEALTHY:
                    healthy_node_steps += 1
                    if agent.nodes[i].state != "ACTIVE": fq_steps += 1
        if done: break
    log = env.get_episode_log()
    return dict(reward=total, completed=log["completed_count"], failed=log["failed_count"],
                reroutes=reroutes, churn=churn, to_down=to_down, to_bad=to_bad,
                det_lat=(st.mean(det_lat) if det_lat else None), n_det=len(det_lat),
                missed=len(pending_onset), false_q_frac=(fq_steps/max(1,healthy_node_steps)),
                time=a_time, max_act=max_act)

def summarize(name, rows):
    def m(k):
        v = [r[k] for r in rows if r[k] is not None]
        return (sum(v)/len(v)) if v else float("nan")
    tot = m("completed")+m("failed")
    return (f"{name:10s} reward={m('reward'):7.1f} comp={m('completed'):6.1f} fail={m('failed'):5.1f} "
            f"ontime%={100*m('completed')/max(1,tot):5.1f} reroutes={m('reroutes'):6.1f} churn={m('churn'):6.1f} "
            f"toDOWN={m('to_down'):5.1f} detLat={m('det_lat'):5.2f} missedDet={m('missed'):4.2f} falseQ%={100*m('false_q_frac'):5.1f} "
            f"t/ep={m('time'):5.2f}s maxAct={1000*m('max_act'):5.1f}ms")

CONFIGS = {
 "sandbox": {},
 "big_cluster": dict(n_nodes=16, node_capacity=6, arrival_rate=9.0),
 "small_cluster": dict(n_nodes=3, node_capacity=3, arrival_rate=1.0),
 "high_load": dict(arrival_rate=3.4),
 "low_load": dict(arrival_rate=0.8),
 "more_failures": dict(health_transition=np.array([[0.95,0.03,0.02],[0.10,0.80,0.10],[0.05,0.05,0.90]])),
 "rare_failures": dict(health_transition=np.array([[0.998,0.0015,0.0005],[0.15,0.83,0.02],[0.05,0.05,0.90]])),
 "tight_deadline": dict(slack_range=(1,3)),
 "long_tasks": dict(duration_range=(8,20), slack_range=(6,14), arrival_rate=0.9),
 "cap1": dict(node_capacity=1, arrival_rate=0.7, n_nodes=8),
}
if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONFIGS)
    for cname in which:
        print(f"== {cname} ({seeds} seeds)")
        for name, cls in [("baseline", RoundRobinNoHealthCheck), ("MyAgent", MyAgent), ("oracle", Oracle)]:
            rows = [run_episode(cls, CONFIGS[cname], s) for s in range(seeds)]
            print(summarize(name, rows), flush=True)
