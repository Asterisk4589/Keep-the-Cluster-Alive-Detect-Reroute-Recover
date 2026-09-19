"""
Smoke tests for review1_basic/agent.py (MyAgent).
Checks:
  1. Agent instantiates and implements the interface.
  2. Runs a full episode without crashing.
  3. Every returned action is a valid task_id -> node_id in [0, n_nodes).
  4. Agent never assigns a task to a node beyond capacity in a way that
     would violate env rules (though env itself silently rejects overflows,
     we check whether the agent is *trying* to overflow, which signals bad logic).
  5. Reasonable performance vs baseline round robin (sanity, not a hard pass/fail).
"""
import sys
import traceback
from sandbox_env import make_sandbox_env
from agent import MyAgent
from baseline_agent import RoundRobinNoHealthCheck


def run_episode(agent_cls, seed, debug=True, **agent_kwargs):
    env = make_sandbox_env(seed=seed, debug=debug)
    agent = agent_cls(n_nodes=env.n_nodes, node_capacity=env.node_capacity, **agent_kwargs)
    obs = env.reset()
    agent.reset()

    total_reward = 0.0
    invalid_task_ids = 0
    invalid_node_ids = 0
    overflow_attempts = 0
    crashed = False
    crash_trace = None

    for t in range(env.episode_length):
        valid_task_ids = {task["task_id"] for task in obs["tasks"]}
        try:
            actions = agent.act(obs)
        except Exception:
            crashed = True
            crash_trace = traceback.format_exc()
            break

        if not isinstance(actions, dict):
            raise AssertionError(f"act() returned {type(actions)}, expected dict")

        # Validate every returned action BEFORE stepping the env
        node_load_now = {}
        for task in obs["tasks"]:
            if task["node"] is not None:
                node_load_now[task["node"]] = node_load_now.get(task["node"], 0) + 1

        for task_id, node_id in actions.items():
            if task_id not in valid_task_ids:
                invalid_task_ids += 1
                continue
            if not (isinstance(node_id, (int,)) and 0 <= node_id < env.n_nodes):
                invalid_node_ids += 1
                continue
            # check for capacity overflow attempts (moving to a NEW node beyond capacity)
            task_obj = next(tt for tt in obs["tasks"] if tt["task_id"] == task_id)
            if task_obj["node"] != node_id:
                if node_load_now.get(node_id, 0) >= env.node_capacity:
                    overflow_attempts += 1

        try:
            obs, reward, done, info = env.step(actions)
        except Exception:
            crashed = True
            crash_trace = traceback.format_exc()
            break

        try:
            agent.update(obs, reward, done, info)
        except Exception:
            crashed = True
            crash_trace = traceback.format_exc()
            break

        total_reward += reward
        if done:
            break

    log = env.get_episode_log()
    return {
        "total_reward": total_reward,
        "completed": log["completed_count"],
        "failed": log["failed_count"],
        "invalid_task_ids": invalid_task_ids,
        "invalid_node_ids": invalid_node_ids,
        "overflow_attempts": overflow_attempts,
        "crashed": crashed,
        "crash_trace": crash_trace,
        "steps_run": t + 1,
    }


def main():
    seeds = [0, 1, 2, 3, 4]
    print("=" * 70)
    print("TEST 1: Interface compliance (reset/act/update exist, abstract methods)")
    print("=" * 70)
    from agent_interface import BaseAgent
    assert issubclass(MyAgent, BaseAgent), "MyAgent must subclass BaseAgent"
    a = MyAgent(n_nodes=6, node_capacity=4)
    for method in ("reset", "act", "update"):
        assert hasattr(a, method), f"missing method {method}"
    print("PASS: MyAgent subclasses BaseAgent and implements reset/act/update\n")

    print("=" * 70)
    print("TEST 2: Runs full episodes without crashing, valid actions only")
    print("=" * 70)
    all_ok = True
    results = []
    for seed in seeds:
        r = run_episode(MyAgent, seed=seed, debug=True)
        results.append((seed, r))
        status = "CRASHED" if r["crashed"] else "OK"
        print(f"seed={seed:2d}  status={status:8s}  steps={r['steps_run']:4d}  "
              f"reward={r['total_reward']:8.2f}  completed={r['completed']:4d}  "
              f"failed={r['failed']:4d}  invalid_task_ids={r['invalid_task_ids']}  "
              f"invalid_node_ids={r['invalid_node_ids']}  overflow_attempts={r['overflow_attempts']}")
        if r["crashed"]:
            all_ok = False
            print("  --- crash trace ---")
            print(r["crash_trace"])
        if r["invalid_task_ids"] or r["invalid_node_ids"]:
            all_ok = False
        if r["overflow_attempts"] > 0:
            print(f"  NOTE: agent attempted {r['overflow_attempts']} over-capacity NEW placements "
                  f"(env silently rejects these, but it signals the agent isn't tracking load "
                  f"correctly within a step)")

    print()
    if all_ok:
        print("PASS: no crashes, no invalid task/node ids across all seeds\n")
    else:
        print("FAIL: see above\n")

    print("=" * 70)
    print("TEST 3: Does it beat health-blind round robin baseline?")
    print("=" * 70)
    for seed in seeds:
        rr = run_episode(RoundRobinNoHealthCheck, seed=seed, debug=True)
        my = next(r for s, r in results if s == seed)
        delta_reward = my["total_reward"] - rr["total_reward"]
        delta_completed = my["completed"] - rr["completed"]
        delta_failed = my["failed"] - rr["failed"]
        verdict = "BETTER" if delta_reward > 0 else ("WORSE" if delta_reward < 0 else "TIE")
        print(f"seed={seed:2d}  baseline_reward={rr['total_reward']:8.2f}  "
              f"agent_reward={my['total_reward']:8.2f}  delta={delta_reward:+8.2f}  "
              f"[{verdict}]  delta_completed={delta_completed:+4d}  delta_failed={delta_failed:+4d}")

    print()
    print("=" * 70)
    print("TEST 4: No manual reroute of in-flight tasks (README claims this is intentional)")
    print("=" * 70)
    # Run one episode, track whether agent ever changes an already-assigned task's node
    env = make_sandbox_env(seed=7, debug=True)
    agent = MyAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
    obs = env.reset()
    agent.reset()
    reroutes_attempted = 0
    for t in range(env.episode_length):
        assigned_before = {task["task_id"]: task["node"] for task in obs["tasks"] if task["node"] is not None}
        actions = agent.act(obs)
        for task_id, node_id in actions.items():
            if task_id in assigned_before and assigned_before[task_id] != node_id:
                reroutes_attempted += 1
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)
        if done:
            break
    print(f"reroutes_attempted={reroutes_attempted} (expected 0 for a Review-1 'no migration' agent)")
    if reroutes_attempted == 0:
        print("PASS: agent never reassigns already-running tasks, matches README/stated scope\n")
    else:
        print("NOTE: agent DID attempt to move in-flight tasks -- contradicts README claim of "
              "'intentionally does not migrate in-flight work yet'\n")

    print("=" * 70)
    print("TEST 5: Frozen-on-dead-node check (the exact failure mode PS3 targets)")
    print("=" * 70)
    # Confirm: since this agent never reroutes, tasks stuck on a node that goes DOWN
    # will stall until deadline, same as baseline, for tasks *already* on that node
    # at time of failure. This is expected/scoped-out for Review 1, just confirming.
    env = make_sandbox_env(seed=3, debug=True)
    agent = MyAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
    obs = env.reset()
    agent.reset()
    stuck_examples = []
    node_true_state = {}
    for t in range(env.episode_length):
        actions = agent.act(obs)
        prev_tasks = {task["task_id"]: task for task in obs["tasks"]}
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)
        for nid, state in enumerate(info.get("node_true_states", [])):
            node_true_state[nid] = state
        for failed_node in info.get("failed_this_step", []):
            if node_true_state.get(failed_node) == "down":
                stuck_on_node = [tid for tid, tk in prev_tasks.items()
                                  if tk["node"] == failed_node]
                if stuck_on_node:
                    stuck_examples.append((t, failed_node, stuck_on_node[:3]))
        if done:
            break
    if stuck_examples:
        t0, node0, tasks0 = stuck_examples[0]
        print(f"Confirmed: at t={t0}, node {node0} went DOWN while tasks {tasks0} were "
              f"assigned to it -- since this agent doesn't migrate in-flight work, those "
              f"tasks will stall exactly like the baseline until they miss deadline.")
        print("This matches the agent's stated scope (admission-only, no migration) -- "
              "expected for Review 1, should be fixed in Review 2.\n")
    else:
        print("No down-while-assigned events observed in this seed run.\n")


if __name__ == "__main__":
    main()
