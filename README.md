# MM26AI02 — Keep the Cluster Alive
### Detect · Reroute · Recover

> **An autonomous, fault-aware scheduler for a simulated distributed cluster.**
>
> The agent observes only noisy node telemetry and task state. It infers hidden node health, prevents new work from entering unhealthy nodes, decides when an in-flight task is worth migrating despite a cold-restart penalty, and continuously adapts as nodes degrade and recover.

---

## 1. The Problem

The environment contains a distributed cluster in which:

- Nodes have **hidden health states**: `HEALTHY`, `DEGRADED`, or `DOWN`.
- Each node's health evolves independently over time.
- The agent **does not receive the true health state**.
- Instead, it sees noisy telemetry:
  - heartbeat status
  - latency
  - error rate
  - queue length
  - capacity
- Tasks continuously arrive with:
  - current node assignment
  - remaining work
  - deadline
- Moving an already-running task to another node causes a **cold restart**: its remaining work resets to its original duration.

The resulting decision is not simply:

> "Which node has the lowest load?"

It is:

> **"Given uncertain node health, task urgency, capacity, and the cost of restarting work, should I stay, place, or reroute?"**

That is the core of the agent.

---

# 2. What the Agent Actually Does

The system runs a continuous observe → infer → decide → act loop:

```text
             ┌──────────────────────────┐
             │     Cluster telemetry    │
             │ heartbeat / latency /    │
             │ error / queue / capacity │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │   Health Inference      │
             │                          │
             │ peer + self baselines    │
             │ robust MAD statistics    │
             │ 3-state belief filter    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │   Node State Machine     │
             │                          │
             │ ACTIVE → SUSPECT →       │
             │ QUARANTINED → PROBING    │
             └────────────┬─────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      New / Pending Work       In-flight Work
              │                       │
              ▼                       ▼
       Deadline-aware          Slack + EV-style
         placement              reroute decision
              │                       │
              └───────────┬───────────┘
                          ▼
                    Action Selection
                          │
                          ▼
                 ┌─────────────────┐
                 │ Cluster evolves │
                 └────────┬────────┘
                          │
                          └──────► repeat
```

The important architectural choice is that **new work and in-flight work are treated differently**.

---

# 3. Why New Work and In-Flight Work Need Different Policies

A common load balancer can simply stop sending new requests to an unhealthy node.

That is not enough here.

An already-running task may be sitting on a node that has just failed. Leaving it there can make it miss its deadline.

However, blindly moving it is also dangerous because **migration restarts the task**.

Therefore:

### New task

The agent can be conservative:

> "Don't send new work to a node I currently believe is suspicious."

### In-flight task

The agent asks a harder question:

> "Is this task sufficiently at risk that restarting it on another node is better than letting it continue?"

This asymmetric policy is one of the central design decisions in the project.

---

# 4. Hidden Health Inference

The agent never reads the environment's true node state.

It maintains a belief over three states:

```text
P(healthy)
P(degraded)
P(failed)
```

The belief is updated from observable telemetry and temporal evidence.

### Signals

#### 1. Heartbeat

A heartbeat is the node's periodic indication that it is alive and responding.

Consecutive heartbeat misses are treated as stronger evidence than one isolated miss.

#### 2. Latency

Latency is normalized against:

- the current cluster peers
- the node's own historical baseline

This makes the detector less dependent on a single hard-coded latency number.

#### 3. Error rate

Error rate is similarly compared against peer/self baselines.

#### 4. Queue length

Queue length is **not used as direct node-health evidence**.

Why?

Because in this environment, queue length is heavily influenced by the agent's own scheduling decisions. Using it as a health signal can create a feedback loop:

```text
agent sends work to node
        ↓
queue increases
        ↓
agent declares node unhealthy
        ↓
agent moves work away
```

Instead, queue/load is used where it belongs: **scheduling and capacity decisions**.

---

# 5. Robust Telemetry Normalization

The detector uses median/MAD-style robust statistics instead of assuming fixed absolute telemetry values.

This provides resilience against:

- outliers
- heavy-tailed measurements
- different telemetry scales
- missing latency/error values
- dropped node observations

Relative MAD floors also reduce dependence on the exact units of latency/error telemetry.

The final agent additionally treats an absent node observation as a heartbeat miss rather than silently keeping stale health information forever.

---

# 6. Bayesian-Style 3-State Belief

The agent combines current evidence with a sticky transition model:

```text
             ┌───────────────┐
             │    HEALTHY    │
             └───────┬───────┘
                     │
                degradation
                     ▼
             ┌───────────────┐
             │   DEGRADED    │
             └───────┬───────┘
                     │
                  failure
                     ▼
             ┌───────────────┐
             │     FAILED    │
             └───────────────┘
```

The model also allows recovery.

This matters because the environment does not define failures as permanently dead nodes. Nodes can recover, so the agent must be able to **trust a node again** rather than permanently blacklist it.

The belief is therefore not a binary:

```text
alive / dead
```

but a graded uncertainty model:

```text
healthy probability
degraded probability
failed probability
```

---

# 7. Circuit Breaker + Recovery Probing

The belief model feeds a state machine:

```text
ACTIVE
   │
   │ persistent suspicious evidence
   ▼
SUSPECT
   │
   │ stronger/persistent failure belief
   ▼
QUARANTINED
   │
   │ wait + recovery evidence
   ▼
PROBING
   │
   │ sustained healthy evidence
   ▼
ACTIVE
```

### Why this matters

Without hysteresis, a noisy node can oscillate:

```text
bad → good → bad → good → bad
```

and cause constant scheduler changes.

The circuit breaker creates **stateful behavior**:

- enter suspicion after persistent evidence
- quarantine after stronger evidence
- wait before probing
- require sustained healthy evidence before returning to active service

This is intended to reduce false reactions and thrashing.

---

# 8. Deadline Slack

The agent estimates how much breathing room a task has before its deadline.

Conceptually:

```text
slack =
    time until deadline
    − estimated queue wait
    − estimated time to finish remaining work
```

In the implementation, estimated completion time is adjusted by the node's inferred effective service rate.

Interpretation:

```text
large positive slack → task has room to wait

slack ≈ 0             → task is becoming urgent

negative slack        → task is already estimated to be infeasible
```

In-flight tasks are ordered by slack so that the most deadline-sensitive work is considered first.

This means the scheduler does not simply evacuate every task from a degraded node.

It prioritizes:

> **Which task is most likely to miss its deadline if I do nothing?**

---

# 9. Migration Is Not Free

This is one of the most important properties of the environment.

If:

```text
Task 47:
Node 2 → Node 4
```

the task is **restarted** on Node 4.

Its progress is not preserved.

Therefore this is intentionally avoided:

```text
Node A: slightly bad
Node B: slightly better

A → B
```

because the health improvement may not compensate for throwing away progress.

The final agent therefore estimates the value of:

```text
STAY
vs.
MOVE
```

using:

- current node failure belief
- destination failure belief
- remaining work
- estimated service rate
- queue wait
- deadline
- original task duration / restart work

The resulting gain is used as an **expected-value-style migration signal**.

This is substantially more principled than migrating merely because another node has a higher health score.

---

# 10. Why the Agent Does Not Reroute Everything

The migration policy contains several safeguards:

### Cooldown

A task cannot be rerouted repeatedly every step.

```text
migration
    ↓
3-step cooldown
    ↓
migration can be reconsidered
```

### Churn margin

A small expected benefit is not enough to justify a restart.

### Migration cap

Only a bounded number of tasks may be rerouted in one step.

The cap scales with cluster size:

```text
max_reroutes = round(0.75 × number_of_nodes)
```

with a minimum of one.

### Result

The agent is designed to avoid:

```text
A → B → C → B → A ...
```

when those moves do not provide enough expected benefit.

---

# 11. Target Selection

For a new assignment or migration, candidate nodes are filtered by:

1. available capacity
2. circuit-breaker state
3. health belief

Then the agent evaluates a placement score combining:

- deadline feasibility
- probability/quality of node health
- load penalty

The key design principle is:

> **A lightly loaded unhealthy node should not automatically beat a healthy node simply because it has more spare capacity.**

Health and deadline feasibility therefore matter directly in placement.

---

# 12. Deadline-Aware Admission

Pending tasks are ordered using **least laxity**:

```text
laxity =
deadline − current_step − remaining_work
```

Tasks with tighter deadlines are handled first.

Infeasible tasks are placed after tasks that are still feasible.

This avoids a scheduler that greedily fills capacity with easy/low-urgency tasks while allowing deadline-critical work to wait.

---

# 13. Continuous Recovery

The agent does not make one failure decision and stop.

Every step it:

1. receives fresh telemetry
2. updates node beliefs
3. updates circuit-breaker state
4. recalculates task urgency
5. evaluates reroutes
6. places new work
7. observes the result
8. repeats

Therefore, if a destination node later becomes unhealthy, it can be detected and reconsidered after the task's cooldown.

Likewise, a recovered node can eventually return to service through the probing state.

---

# 14. Explainability / Diagnostics

The agent records structured events such as:

```text
ACTIVE → SUSPECT
SUSPECT → QUARANTINED
QUARANTINED → PROBING
PROBING → ACTIVE

REROUTE:
  source node
  target node
  task
  failure belief
  task slack
  estimated gain
```

This gives the system an inspectable decision trail rather than an opaque stream of node assignments.

`get_diagnostics()` exposes the buffered event log.

---

# 15. Defensive Behavior

Evaluation environments can contain imperfect telemetry.

The agent explicitly handles:

- `None` latency
- invalid telemetry values
- NaN latency
- missing node observations
- telemetry/update exceptions
- saturated candidate nodes

If the main decision path encounters an unexpected edge case, a safe placement fallback is used rather than allowing the agent to crash.

This is important in an evaluation setting where one uncaught telemetry exception can terminate an entire episode.

---

# 16. What Makes This Agentic?

The project is not simply:

```text
if node_bad:
    move_task()
```

The agent maintains **state and beliefs over time** and repeatedly makes decisions under uncertainty.

Its loop is:

```text
OBSERVE
   ↓
Infer hidden node state
   ↓
Update belief + circuit state
   ↓
Estimate task urgency
   ↓
Estimate consequences of staying/moving
   ↓
Select action
   ↓
Observe new evidence
   ↓
Update again
```

The environment changes independently of the agent, and the agent must continuously adapt.

That is the central agentic behavior:

> **persistent state + uncertain observations + autonomous action + feedback-driven adaptation.**

---

# 17. Evaluation

The repository includes a benchmark harness comparing:

- `baseline`: round-robin scheduling with no health awareness
- `MyAgent`: the submitted autonomous scheduler
- `oracle`: a measurement-only upper-bound reference that can see true node state

The benchmark evaluates multiple cluster conditions rather than a single sandbox configuration.

### Metrics tracked

The harness measures:

| Metric | What it tells us |
|---|---|
| Reward | Overall task completion/failure outcome |
| Completed | Number of completed tasks |
| Failed | Deadline-missed tasks |
| On-time % | Fraction of completed/failed tasks completed successfully |
| Reroutes | Total migrations |
| Churn | Reroutes that were unnecessary relative to a healthy source |
| Detection latency | Time from true failure onset to agent suspicion/quarantine |
| Missed detection | Failures not detected during the episode |
| False quarantine | Healthy-node time incorrectly treated as non-active |
| Runtime | Agent computation cost |

---

# 18. Benchmark Results

The included verification report evaluated the canonical `agent.py` over **20 seeds across 10 supplied stress configurations**.

Reward comparison:

| Configuration | Baseline | Agent |
|---|---:|---:|
| sandbox | 566.5 | **753.8** |
| big_cluster | 2741.4 | **3476.2** |
| small_cluster | 276.5 | **320.3** |
| high_load | 961.3 | **978.0** |
| low_load | 229.1 | **314.4** |
| more_failures | 354.8 | **586.7** |
| rare_failures | 774.6 | **786.8** |
| tight_deadline | 451.0 | **690.6** |
| long_tasks | 240.6 | **297.5** |
| cap1 | 205.4 | **258.5** |

The supplied verification report states that the agent improved reward over the supplied baseline in all 10 tested configurations.

These are **local/reviewer-harness measurements**, not a guarantee of hidden organizer-evaluation performance.

---

# 19. Robustness Testing

The included robustness harness tested the agent against telemetry perturbations including:

- normal telemetry
- latency represented in seconds
- latency scaled ×10
- noisier telemetry
- flaky heartbeats
- missing error-rate values
- missing latency values
- dropped node observations

The verification report records that the agent completed all supplied perturbation modes without crashing.

---

# 20. Repository Structure

```text
.
├── agent.py                 # Canonical submission agent
├── agent_interface.py       # Required BaseAgent interface
├── env/
│   └── cluster_env.py       # Supplied cluster simulation
├── sandbox_env.py           # Participant-facing sandbox wrapper
├── baseline_agent.py        # No-health-awareness baseline
│
├── bench.py                 # Benchmark / evaluation harness
├── robust.py                # Telemetry robustness tests
├── ablate.py                # Baseline/final comparison
├── make_patched.py          # Validate/copy submission agent
│
├── VERIFICATION_REPORT.md   # Verification and benchmark summary
└── REVIEW_AND_HANDOFF.md    # Implementation/review handoff notes
```

### Submission file

For a single-file agent submission, the canonical implementation is:

```text
agent.py
```

It implements:

```python
MyAgent
```

against the supplied `BaseAgent` interface.

---

# 21. Running the Verification Suite

Install the required Python dependency:

```bash
pip install numpy
```

Compile/check the Python files:

```bash
python -m py_compile *.py env/*.py
```

Run the benchmark:

```bash
python bench.py 20
```

Run robustness checks:

```bash
python robust.py 5
```

Run the comparison/ablation harness:

```bash
python ablate.py 20
```

Validate/copy the submission:

```bash
python make_patched.py agent.py agent_submission.py
```

> The organizer-only hidden evaluation harness is not included in the archive, so official hidden-evaluation results cannot be reproduced locally from this ZIP alone.

---

# 22. Important Implementation Details for Judges

### The agent does NOT cheat

The final agent only consumes fields exposed through `agent_interface.py`.

It does **not** read:

```text
true node health
failure timing
debug ground truth
```

The true environment state is used only by the local benchmark/oracle for measurement.

### The environment intentionally makes rerouting expensive

Moving an in-flight task resets its duration.

Therefore, migration decisions are explicitly designed around the trade-off:

```text
benefit of escaping an unhealthy node
                VS
cost of restarting work
```

### Health is probabilistic, not binary

The agent does not pretend noisy telemetry gives certainty.

It maintains:

```text
P(healthy)
P(degraded)
P(failed)
```

and uses these beliefs throughout scheduling.

### Recovery is part of the design

A node is not permanently blacklisted after failure.

It moves through:

```text
QUARANTINED → PROBING → ACTIVE
```

when evidence supports recovery.

---

# 23. Known Evaluation Boundary

The supplied archive does not contain the organizer-only `eval_harness.py`.

Therefore:

- local benchmark results are reproducible with the supplied environment/harness
- hidden official evaluation configuration is not known
- local scores should not be presented as official competition scores

The sandbox itself also warns that its node count, capacity, arrival rate and failure dynamics are illustrative rather than guaranteed to match hidden evaluation.

The implementation therefore avoids depending on a single fixed sandbox configuration by using relative/self-normalized telemetry and stateful inference.

---

# 24. Design Summary

The system can be summarized as four layers:

```text
┌─────────────────────────────────────────────┐
│  1. SENSE                                   │
│  noisy heartbeat / latency / errors         │
├─────────────────────────────────────────────┤
│  2. BELIEVE                                 │
│  robust normalization + 3-state belief      │
│  + circuit breaker                          │
├─────────────────────────────────────────────┤
│  3. DECIDE                                  │
│  deadline slack + capacity + health         │
│  + stay-vs-move value + anti-churn          │
├─────────────────────────────────────────────┤
│  4. ACT + LEARN FROM FEEDBACK               │
│  place / reroute / probe / recover          │
│  then repeat every simulation step          │
└─────────────────────────────────────────────┘
```

### Core idea

> **Don't just detect that a node is bad. Decide what to do about it while accounting for uncertainty, deadlines, capacity, recovery, and the cost of moving work.**

That is the purpose of the agent in **MM26AI02 — Keep the Cluster Alive: Detect, Reroute, Recover**.
