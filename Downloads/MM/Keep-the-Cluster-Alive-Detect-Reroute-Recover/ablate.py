import sys
import numpy as np
from bench import CONFIGS, run_episode, Oracle
from baseline_agent import RoundRobinNoHealthCheck
from agent import MyAgent

seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfgs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONFIGS)

def mean_result(cls, cfg):
    rows = [run_episode(cls, CONFIGS[cfg], s) for s in range(seeds)]
    return (
        np.mean([r["reward"] for r in rows]),
        np.mean([r["churn"] for r in rows]),
        np.mean([r["failed"] for r in rows]),
    )

print("config".ljust(18), "baseline reward", "agent reward", "agent churn", "agent failed")
for cfg in cfgs:
    b = mean_result(RoundRobinNoHealthCheck, cfg)
    a = mean_result(MyAgent, cfg)
    print(
        cfg.ljust(18),
        f"{b[0]:14.1f}",
        f"{a[0]:12.1f}",
        f"{a[1]:11.1f}",
        f"{a[2]:11.1f}",
        flush=True,
    )
