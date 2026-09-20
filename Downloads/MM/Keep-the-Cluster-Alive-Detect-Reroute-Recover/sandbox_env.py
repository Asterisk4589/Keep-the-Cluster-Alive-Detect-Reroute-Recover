"""
SANDBOX environment. This is yours to build and test against as much as you
want -- unlimited episodes, no scoring, and an optional debug mode that
reveals ground-truth node health so you can sanity-check your own detection
logic.

IMPORTANT: this sandbox's node count, capacity, arrival rate and failure
dynamics are illustrative examples, NOT the exact configuration used for
final evaluation. The real eval may use a different number of nodes, a
different capacity, a different arrival rate, and different failure timing.
Build an agent that generalizes from telemetry signal, don't hardcode
thresholds tuned to this exact sandbox.

Usage:
    from sandbox_env import make_sandbox_env
    env = make_sandbox_env(seed=0, debug=True)
    obs = env.reset()
    obs, reward, done, info = env.step({101: 2, 104: 0})
"""

from env.cluster_env import ClusterEnv

N_NODES = 6
NODE_CAPACITY = 4
EPISODE_LENGTH = 400


def make_sandbox_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    """
    debug=True exposes info['node_true_states'] and info['failed_this_step']
    so you can plot/verify your detector against ground truth. Do your final
    tuning with debug=False so you're not accidentally relying on signal you
    won't have during evaluation.
    """
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,
        arrival_rate=2.0,
        duration_range=(3, 8),
        slack_range=(4, 10),
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )
