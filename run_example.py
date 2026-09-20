"""
Quick demo: run the health-blind round-robin baseline against the sandbox in
debug mode and print what happens to task completion/failure rates around a
node failure, so you can SEE the failure mode you're meant to fix.
Run: python run_example.py
"""

from sandbox_env import make_sandbox_env
from baseline_agent import RoundRobinNoHealthCheck

env = make_sandbox_env(seed=3, debug=True)
agent = RoundRobinNoHealthCheck(n_nodes=env.n_nodes, node_capacity=env.node_capacity)

obs = env.reset()
agent.reset()

completed_before = 0
failed_before = 0

for t in range(env.episode_length):
    actions = agent.act(obs)
    obs, reward, done, info = env.step(actions)
    agent.update(obs, reward, done, info)

    if info["failed_this_step"]:
        for node_id in info["failed_this_step"]:
            new_state = info["node_true_states"][node_id]
            print(f"[t={t:3d}] node {node_id} -> {new_state}")

    if (t + 1) % 100 == 0:
        log = env.get_episode_log()
        d_completed = log["completed_count"] - completed_before
        d_failed = log["failed_count"] - failed_before
        completed_before, failed_before = log["completed_count"], log["failed_count"]
        print(f"  t={t+1:3d}  completed(last 100)={d_completed:3d}  "
              f"failed/missed-deadline(last 100)={d_failed:3d}")

    if done:
        break

print("\nWatch failed/missed-deadline counts spike right after a node goes down --")
print("that's tasks stuck frozen on a dead node because the baseline never rechecks health.")
