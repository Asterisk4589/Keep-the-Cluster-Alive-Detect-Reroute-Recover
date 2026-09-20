"""
Test suite for MyAgent (agent.py).

Drop this file into the same folder as agent.py, agent_interface.py,
baseline_agent.py, sandbox_env.py, and env/cluster_env.py, then run:

    python test_agent.py

What it checks:
  1. Interface compliance (subclasses BaseAgent, implements reset/act/update).
  2. Runs full episodes without crashing; every returned action is a valid
     task_id -> node_id pair.
  3. No capacity-overflow attempts: the agent never tries to place/move a
     task onto a node that is already at capacity (checked against the
     REAL load from obs, not the agent's internal bookkeeping).
  4. Performance vs. the health-blind round-robin baseline (sanity check).
  5. Migration behavior (if any): how much it reroutes, how much of that
     reroute activity is "churn" (moving a task off a node that was
     actually healthy at the time -- ground truth only available here in
     debug mode, never during real evaluation).
  6. Rescue behavior (if any): for tasks that end up stuck on a node that
     goes truly DOWN, does the agent ever move them to safety, and how
     quickly?
  7. Cooldown/flapping sanity: does any single task get rerouted more than
     once in a very short window (a sign of thrashing)?

Notes:
  - debug=True is used throughout so this script can see ground-truth node
    health for scoring/diagnosis purposes only. The agent itself is never
    given this information -- obs passed to agent.act()/update() is
    unchanged either way.
  - This script doesn't assume the agent does or doesn't migrate in-flight
    tasks. If it never reroutes, sections 4-6 will just report zero
    activity rather than failing.
"""
import traceback

from sandbox_env import make_sandbox_env
from agent import MyAgent
from baseline_agent import RoundRobinNoHealthCheck
from agent_interface import BaseAgent


def run_episode(agent_cls, seed, debug=True, **agent_kwargs):
    env = make_sandbox_env(seed=seed, debug=debug)
    agent = agent_cls(n_nodes=env.n_nodes, node_capacity=env.node_capacity, **agent_kwargs)
    obs = env.reset()
    agent.reset()

    total_reward = 0.0
    invalid_task_ids = 0
    invalid_node_ids = 0
    overflow_attempts = 0
    overflow_examples = []
    reroutes_total = 0
    reroutes_from_healthy = 0
    crashed = False
    crash_trace = None
    node_true_state = {}
    rescue_events = []       # list of delays (steps) until rescue
    stuck_since = {}         # task_id -> step it started sitting on a DOWN node

    t = 0
    for t in range(env.episode_length):
        valid_tasks = {task["task_id"]: task for task in obs["tasks"]}
        assigned_before = {tid: tk["node"] for tid, tk in valid_tasks.items() if tk["node"] is not None}

        try:
            actions = agent.act(obs)
        except Exception:
            crashed = True
            crash_trace = traceback.format_exc()
            break

        if not isinstance(actions, dict):
            raise AssertionError(f"act() returned {type(actions)}, expected dict")

        real_load = {}
        for task in obs["tasks"]:
            if task["node"] is not None:
                real_load[task["node"]] = real_load.get(task["node"], 0) + 1

        for task_id, node_id in actions.items():
            if task_id not in valid_tasks:
                invalid_task_ids += 1
                continue
            if not (isinstance(node_id, int) and 0 <= node_id < env.n_nodes):
                invalid_node_ids += 1
                continue
            task_obj = valid_tasks[task_id]
            if task_obj["node"] is not None and task_obj["node"] != node_id:
                reroutes_total += 1
                if node_true_state.get(task_obj["node"]) == "healthy":
                    reroutes_from_healthy += 1
            if task_obj["node"] != node_id and real_load.get(node_id, 0) >= env.node_capacity:
                overflow_attempts += 1
                if len(overflow_examples) < 5:
                    overflow_examples.append(
                        (t, task_id, task_obj["node"], node_id, real_load.get(node_id, 0), env.node_capacity)
                    )

        for tid, node in assigned_before.items():
            if node_true_state.get(node) == "down":
                stuck_since.setdefault(tid, t)

        try:
            obs, reward, done, info = env.step(actions)
        except Exception:
            crashed = True
            crash_trace = traceback.format_exc()
            break

        for nid, state in enumerate(info.get("node_true_states", [])):
            node_true_state[nid] = state

        still_present = {task["task_id"]: task for task in obs["tasks"]}
        for tid in list(stuck_since.keys()):
            if tid not in still_present:
                del stuck_since[tid]  # completed or failed, resolved either way
            elif still_present[tid]["node"] is not None and node_true_state.get(still_present[tid]["node"]) != "down":
                rescue_events.append(t + 1 - stuck_since[tid])
                del stuck_since[tid]

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
        "overflow_examples": overflow_examples,
        "reroutes_total": reroutes_total,
        "reroutes_from_healthy": reroutes_from_healthy,
        "rescue_events": rescue_events,
        "still_stuck_at_end": len(stuck_since),
        "crashed": crashed,
        "crash_trace": crash_trace,
        "steps_run": t + 1,
    }


def main():
    seeds = [0, 1, 2, 3, 4]

    print("=" * 70)
    print("TEST 1: Interface compliance")
    print("=" * 70)
    assert issubclass(MyAgent, BaseAgent), "MyAgent must subclass BaseAgent"
    probe = MyAgent(n_nodes=6, node_capacity=4)
    for method in ("reset", "act", "update"):
        assert hasattr(probe, method), f"missing method {method}"
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
              f"failed={r['failed']:4d}  invalid_ids={r['invalid_task_ids'] + r['invalid_node_ids']}  "
              f"overflow_attempts={r['overflow_attempts']}")
        if r["crashed"]:
            all_ok = False
            print("  --- crash trace ---")
            print(r["crash_trace"])
        if r["invalid_task_ids"] or r["invalid_node_ids"]:
            all_ok = False
        if r["overflow_attempts"]:
            all_ok = False
            for (ts, tid, src, dst, load_at_dst, cap) in r["overflow_examples"]:
                print(f"    e.g. t={ts}: task {tid} moved {src}->{dst}, "
                      f"but node {dst} real load was already {load_at_dst}/{cap}")
    print()
    print("PASS: no crashes, no invalid ids, no overflow attempts\n" if all_ok else
          "FAIL: see above (crashes / invalid ids / capacity overflow attempts)\n")

    print("=" * 70)
    print("TEST 3: Beats health-blind round-robin baseline?")
    print("=" * 70)
    for seed, my in results:
        rr = run_episode(RoundRobinNoHealthCheck, seed=seed, debug=True)
        delta = my["total_reward"] - rr["total_reward"]
        verdict = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "TIE")
        print(f"seed={seed:2d}  baseline={rr['total_reward']:8.2f}  agent={my['total_reward']:8.2f}  "
              f"delta={delta:+8.2f}  [{verdict}]  delta_completed={my['completed'] - rr['completed']:+4d}  "
              f"delta_failed={my['failed'] - rr['failed']:+4d}")
    print()

    print("=" * 70)
    print("TEST 4: Migration / reroute behavior")
    print("=" * 70)
    total_reroutes = sum(r["reroutes_total"] for _, r in results)
    total_churn = sum(r["reroutes_from_healthy"] for _, r in results)
    for seed, r in results:
        print(f"seed={seed:2d}  reroutes_total={r['reroutes_total']:4d}  "
              f"reroutes_from_healthy_node(churn)={r['reroutes_from_healthy']:3d}")
    print()
    if total_reroutes == 0:
        print("No reroutes observed -- this agent does not migrate in-flight tasks "
              "(admission/placement only). Not a failure by itself, just a scope note.\n")
    else:
        churn_rate = total_churn / total_reroutes
        print(f"Total reroutes: {total_reroutes}, of which {total_churn} ({churn_rate:.1%}) moved a "
              f"task off a node that was genuinely healthy at the time.")
        print("Low churn is good (cheap cold-restarts avoided); "
              f"{'looks conservative' if churn_rate < 0.15 else 'higher than ideal -- may be rerouting too eagerly'}.\n")

    print("=" * 70)
    print("TEST 5: Rescue behavior -- tasks stuck on a node that goes DOWN")
    print("=" * 70)
    any_rescue_activity = False
    for seed, r in results:
        events = r["rescue_events"]
        if events:
            any_rescue_activity = True
            avg_delay = sum(events) / len(events)
            print(f"seed={seed:2d}  rescued={len(events):3d}  avg_delay={avg_delay:5.1f} steps  "
                  f"max_delay={max(events):3d} steps  still_stuck_at_end={r['still_stuck_at_end']}")
        else:
            print(f"seed={seed:2d}  rescued=  0  still_stuck_at_end={r['still_stuck_at_end']}")
    print()
    if any_rescue_activity:
        print("Agent actively moves tasks off nodes that go down rather than letting them "
              "stall to their deadline.\n")
    else:
        print("No rescues observed in these seeds -- either no down-while-assigned events "
              "occurred, or the agent never moves tasks off failed nodes (check if this "
              "matches the agent's intended scope).\n")

    print("=" * 70)
    print("TEST 6: Flapping / cooldown sanity")
    print("=" * 70)
    # Re-run one seed with fine-grained per-task reroute timestamp tracking.
    env = make_sandbox_env(seed=1, debug=True)
    agent = MyAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
    obs = env.reset()
    agent.reset()
    last_reroute_step = {}
    quick_repeats = 0  # same task rerouted again within a short window
    WINDOW = 3
    for t in range(env.episode_length):
        assigned_before = {tk["task_id"]: tk["node"] for tk in obs["tasks"] if tk["node"] is not None}
        actions = agent.act(obs)
        for tid, nid in actions.items():
            if tid in assigned_before and assigned_before[tid] != nid:
                if tid in last_reroute_step and t - last_reroute_step[tid] < WINDOW:
                    quick_repeats += 1
                last_reroute_step[tid] = t
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)
        if done:
            break
    print(f"tasks rerouted again within {WINDOW} steps of their last reroute: {quick_repeats}")
    print("Looks stable, no rapid flapping.\n" if quick_repeats == 0 else
          "NOTE: some tasks were rerouted multiple times in quick succession -- "
          "possible flapping/thrashing, worth checking hysteresis/cooldown logic.\n")

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    avg_reward = sum(r["total_reward"] for _, r in results) / len(results)
    avg_completed = sum(r["completed"] for _, r in results) / len(results)
    avg_failed = sum(r["failed"] for _, r in results) / len(results)
    print(f"Average over {len(seeds)} seeds -- reward: {avg_reward:.2f}, "
          f"completed: {avg_completed:.1f}, failed: {avg_failed:.1f}")


if __name__ == "__main__":
    main()
