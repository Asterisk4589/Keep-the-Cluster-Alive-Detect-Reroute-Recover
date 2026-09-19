# MM26AI02 — "Keep the Cluster Alive: Detect, Reroute, Recover"
## End-to-End Workflow, Design Options & Tradeoffs

> **Track:** Agentic AI | **Concepts:** Distributed Systems + Agentic AI + Fault Tolerance
> **Deliverables:** one importable agent class (`MyAgent(BaseAgent)` with `reset`, `act`, `update`), a ≤1-page README, optional interpretability log.

---

## 0. TL;DR — Recommended Strategy

1. **Don't start with RL.** Build a **probabilistic health filter per node** (Bayesian / HMM-style belief over `HEALTHY / DEGRADED / FAILED`) fed by robust, *self-normalizing* telemetry features.
2. **Make every threshold relative**, never absolute: compare a node to its own history *and* to its peers (median/MAD across the cluster). This is what makes the agent generalize to a different node count, capacity, arrival rate and failure timing.
3. **Schedule with health × capacity × deadline slack** (weighted least-loaded / power-of-two-choices, deadline-aware), not round-robin.
4. **Reroute only when expected value says so**: `EV(move) − EV(stay) > cold-restart cost + margin`, with hysteresis and a per-step reroute budget.
5. **Add a circuit-breaker with canary probes** so recovered nodes get traffic back gradually.
6. **Build your own fault-injecting simulator** with randomized parameters to test generalization — the practice sandbox alone is a trap for overfitting.
7. **Log a one-line "why" for every belief change/reroute** — cheap bonus points and invaluable for debugging.

Scoring priority is **(1) on-time completed work > (2) detection latency > (3) churn**. Every design decision below is judged against that ordering.

---

## 1. Problem Decomposition

| Sub-problem | Question the agent must answer | Scored via |
|---|---|---|
| **State estimation** | Which nodes are truly healthy right now, given noisy/incomplete telemetry? | Detection latency, false-positive churn |
| **Impact analysis** | Which in-flight tasks are sitting on a bad node and at risk? | Reward |
| **Placement (load balancing)** | Where should each *new* task go? | Reward |
| **Migration (reroute)** | Which *existing* tasks should move, and when is it worth the cold-restart? | Reward vs. churn |
| **Recovery** | When is a previously bad node safe to trust again? | Reward (capacity utilization) |
| **Generalization** | Does all of the above hold when N, capacity, arrival rate, failure timing change? | Final evaluation |

**Key insight:** this is a *partially observable, sequential decision problem* with an asymmetric cost structure — a missed failure silently kills every task on that node; a false alarm costs a cold-restart per task. The design goal is to make that tradeoff **explicit and tunable**, not implicit in a magic threshold.

---

## 2. Phase 0 — Recon (do this FIRST, ~1–2 hrs)

Before writing any logic, resolve the unknowns. Fill this table from the sandbox docs / demo script:

| Unknown | Why it matters | Where to look |
|---|---|---|
| Exact telemetry fields & types (heartbeat = bool? timestamp? age?) | Determines feature design | Docs, print one raw observation |
| Are missing heartbeats `None`, `0`, or absent keys? | "Incomplete signals" handling | Run baseline, dump obs |
| Noise model (Gaussian? spikes? dropped readings?) | Choice of robust vs. parametric stats | Plot telemetry from healthy nodes |
| Task model: size/work units, deadline formula, arrival process | Slack computation, EV for reroute | Docs |
| Is progress on a task observable (remaining work)? | Enables **stall detection** (huge signal) | Task list fields |
| Node capacity semantics: concurrent slots? queue length cap? | Assignment feasibility | Docs |
| Cold-restart cost: full restart or partial? Fixed penalty? | Reroute EV formula | Docs / experiment |
| Can a node's failure be *partial* (fraction of requests erroring)? | Need graded health, not binary | Docs / demo |
| Do failed nodes drop *queued* tasks or hold them? | Determines urgency | Baseline run |
| Reward definition (per-task on-time? partial credit? penalty for failed?) | Whether to shed hopeless tasks | Docs |
| Per-step compute budget / wall-clock episode budget | Bounds algorithm complexity | Docs |
| Is the environment seeded / deterministic? | Reproducible A/B testing | `reset()` signature |
| Are assignments to already-assigned tasks = reroute? Any reroute API quirks? | Correct action encoding | Base class |
| Can `act` see full task list including in-flight ones each step? | Impact analysis | Base class |

**Deliverable of Phase 0:** a 1-page "Interface cheat sheet" + a **telemetry dump** (CSV/JSONL) of a full baseline episode with ground-truth failure times if the sandbox exposes them for debugging (it won't in eval, but may in practice — use it *only* for offline evaluation, never as agent input).

---

## 3. Phase 1 — Evaluation Harness (before the agent!)

Build the measuring stick first; otherwise you can't tell whether changes help.

### 3.1 Metrics to compute per episode
- **Reward / completion rate** (primary): tasks completed on time ÷ total arrived (and raw reward if different).
- **Detection latency**: for each true failure, `t(agent stops sending new traffic) − t(true failure)`. Also track *reroute latency* (time until in-flight tasks move).
- **Churn rate**: reroutes triggered while the source node was actually healthy ÷ total tasks (or total reroutes).
- **Secondary**: false-positive node-quarantine count, time-in-quarantine of healthy nodes (lost capacity), max queue imbalance, compute time per step.

### 3.2 Scenario suite (your own, on top of the sandbox)
Create a parametrized scenario generator (wrap or re-implement the sandbox) with:

| Axis | Values to sweep |
|---|---|
| # nodes | 3, 5, 8, 16, 32 |
| Node capacity | low / medium / high; heterogeneous |
| Arrival rate | 50% / 80% / 100% / 120% of cluster capacity |
| Failure type | hard crash (silent), heartbeat-only loss, latency inflation, error-rate spike, intermittent flapping, gradual degradation |
| Failure timing | early / late / simultaneous multiple / staggered |
| Failure duration | short (recovers in a few steps) / long / permanent |
| Noise level | 0.5× / 1× / 2× sandbox noise |
| Cluster-wide events | all nodes slow at once (load spike — *not* a node failure!) |

> ⚠️ Include the **"everything got slow because load spiked"** scenario. It's the classic false-positive trap for absolute thresholds and the best justification for peer-relative features.

### 3.3 Experiment hygiene
- Fixed seed list (e.g., 30 seeds × N scenarios) → report **mean ± std/CI**, not a single run.
- Always run **baseline round-robin** and an **oracle agent** (has true health) as lower/upper bounds. The gap `oracle − yours` is your headroom.
- Keep a **held-out scenario set** you never tune on (mimics the hidden evaluation environment).
- Keep a regression script: one command → table of all metrics for all agents.

**Tradeoff — fidelity vs. speed of your own sim:** a faithful copy of the sandbox is quick to write but inherits its biases; a more abstract sim lets you randomize widely but may mis-model details. **Recommendation:** wrap the real sandbox where you can (parameters exposed), and supplement with a light custom sim for extreme randomization.

---

## 4. Phase 2 — Architecture

```
            ┌──────────────────────────────────────────────────────────┐
 telemetry  │  1. Feature extractor (per node, per step)                │
 + tasks ──▶│     • heartbeat age / miss streak                         │
            │     • latency (robust z vs self & vs peers)               │
            │     • error rate (smoothed)                               │
            │     • queue depth / drain rate                            │
            │     • task-progress stall (if observable)                 │
            └───────────────┬──────────────────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │  2. Health estimator (belief per node)                    │
            │     P(HEALTHY), P(DEGRADED), P(FAILED) + effective        │
            │     capacity multiplier + confidence                      │
            └───────────────┬──────────────────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │  3. Circuit breaker / node state machine                  │
            │     ACTIVE → SUSPECT → QUARANTINED → PROBING → ACTIVE     │
            └───────────────┬──────────────────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │  4. Task risk assessor                                    │
            │     per in-flight task: P(miss deadline | stay)           │
            └───────────────┬──────────────────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │  5. Planner: placement + reroute (EV-based, budgeted)     │
            └───────────────┬──────────────────────────────────────────┘
                            ▼
                    assignments ──▶ env      + explanation log
```

Mapping to the required interface:
- `reset()` → clear all per-node filters, breaker states, baselines, logs.
- `update(obs/feedback)` → run modules 1–3 (state estimation) using the new telemetry. *(Use whichever of `act`/`update` receives the new observation — confirm in Phase 0.)*
- `act(obs)` → run modules 4–5, return assignments.

**Design principle:** keep estimation (`update`) and decision (`act`) separable so you can unit-test the detector against recorded telemetry without the scheduler.

---

## 5. Phase 3 — Failure Detection (the core)

### 5.1 Signals & what each tells you

| Signal | Detects | Weakness | Robust handling |
|---|---|---|---|
| **Heartbeat** (miss streak / age) | Hard crash, network partition | Noisy drops → false alarms; useless for *degradation* | Phi-accrual-style: model inter-arrival stats, output a *suspicion level*, not a boolean |
| **Latency** | Slowdown, overload, partial failure | Load-dependent (rises with queue); noisy/heavy-tailed | Use **median/EWMA + MAD**; normalize by queue depth; compare to peers |
| **Error rate** | Partial failure ("errors on some fraction") | Small sample when few requests → high variance | Beta-Binomial / Wilson bound; require sample size |
| **Queue depth** | Stall (queue grows, drain rate ≈ 0) | Also grows with legit load | Use **drain rate** & **queue growth relative to peers**, not raw depth |
| **Task progress** (if visible) | Silent stalls; most direct evidence | May not be exposed | Progress velocity vs. expected; strongest single signal if available |
| **Stale/missing telemetry** | Node gone silent | "Incomplete signals" ≠ dead | Treat *absence* as evidence with a tunable weight, not proof |

### 5.2 Detection algorithm options

| # | Approach | How it works | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| A | **Fixed thresholds** (e.g., latency > X, 3 missed heartbeats) | Simple rules | Trivial, fast | **Overfits sandbox numbers — explicitly warned against** | ❌ Only as a fallback floor |
| B | **Heartbeat timeout + EWMA z-score** | Adaptive per-node baseline | Easy, decent | Baseline drifts to include the failure; slow on gradual degradation | ✅ Strong MVP |
| C | **CUSUM / Page–Hinkley** | Accumulates small persistent deviations | Detects subtle shifts; tunable ARL | Needs reference mean/variance; needs reset logic | ✅ Good for latency/error drift |
| D | **Bayesian Online Change-Point Detection (BOCPD)** | Posterior over run-length | Principled, gives probabilities | Heavier compute, hyperparameters | ⚠️ Stretch |
| E | **HMM / Bayesian forward filter** (3 hidden states) | Emission likelihoods per state; transition matrix encodes "failures are sticky, recoveries possible" | **Gives calibrated beliefs → plugs straight into EV decisions**; handles missing data natively (skip emission) | Need emission model; transition probs are priors | ✅✅ **Recommended core** |
| F | **Phi-accrual detector** | Suspicion φ from heartbeat inter-arrival distribution | Battle-tested (Cassandra/Akka); adaptive | Heartbeat-only | ✅ Use as the heartbeat feature |
| G | **Peer-comparison / outlier detection** (median-MAD across nodes) | Node is bad if it deviates from cluster consensus | **Immune to cluster-wide load shifts**, no absolute thresholds | Fails if >half the nodes bad; needs N≥3–4 | ✅✅ Use as a feature |
| H | **Supervised classifier** (GBM / logistic) trained on your sim | Learns feature→health | Can combine many signals | Sim-to-eval distribution shift; needs labels; risk of encoding env config | ⚠️ Optional, on relative features only |
| I | **RL (PPO/DQN) end-to-end** | Learn policy directly | Could discover non-obvious tradeoffs | Sample-hungry, brittle to param shifts, hard to interpret, compute-budget risk | ❌ Not for first submission |
| J | **Ensemble** (voting / weighted logistic over B,C,E,F,G) | Diverse signals | Robust; best of all | More tuning | ✅ Final form |

### 5.3 Recommended design: "Relative-feature Bayesian filter"

1. **Per-node features each step** (all unit-free):
   - `hb_suspicion` (phi-accrual or miss-streak / expected interval)
   - `lat_z_self` = (latency − own rolling median) / own rolling MAD
   - `lat_z_peer` = (latency − cluster median) / cluster MAD *(after adjusting for queue depth)*
   - `err_excess` = error-rate − cluster median error-rate (with confidence from sample size)
   - `drain_ratio` = tasks completed ÷ expected given queue (relative to peers)
   - `stall` = fraction of node's in-flight tasks with ~zero progress (if visible)
2. **Robust baseline updating:** update a node's "healthy baseline" **only while its belief is HEALTHY** (freeze during suspicion) — prevents baseline contamination ("boiling frog").
3. **Filter:** `belief_t = normalize( (belief_{t−1} · T) ⊙ L(features_t | state) )`
   - `T`: sticky transition matrix (FAILED→FAILED high; HEALTHY→DEGRADED small; recovery small but nonzero).
   - `L`: likelihoods from simple parametric forms (e.g., Gaussian/Student-t on z-scores per state, with wider variance for DEGRADED). Missing feature ⇒ likelihood 1 (or a mild penalty for missing heartbeat).
4. **Output:** `p_fail`, `p_degraded`, and `health_multiplier = Σ_s P(s)·capacity_factor(s)`.

### 5.4 Detection tradeoffs (the big ones)

| Tradeoff | Leaning aggressive | Leaning conservative | Guidance |
|---|---|---|---|
| **Sensitivity vs. false positives** | Faster detection, lower latency score, more churn | Slower detection, more tasks die on dead node | Reward > detection > churn ⇒ **bias toward sensitivity for *stopping new traffic* (cheap), toward conservatism for *rerouting in-flight tasks* (expensive)** |
| **Two-threshold asymmetry** | — | — | Use a **low** suspicion threshold to *stop new assignments* and a **higher** threshold (or EV test) to *migrate* existing work |
| **Window length** | Short window → fast but noisy | Long window → smooth but laggy | Multi-scale (fast + slow EWMA) and require agreement, or let the Bayesian filter accumulate evidence |
| **Self-baseline vs. peer-baseline** | Self: works with heterogeneous nodes | Peer: robust to global load shifts | Use **both**; alarm when consistent |
| **Absent telemetry = evidence?** | Treat as failure → fast crash detection | Treat as neutral → robust to dropped packets | Soft weight that grows with consecutive misses |
| **Binary vs. graded health** | Binary is simple | Graded (capacity multiplier) uses degraded nodes productively | Graded is better for throughput — a half-speed node is still useful |
| **Model complexity vs. compute budget** | BOCPD/particle filters | Simple EWMA/CUSUM | Filter per node is O(1)/step — cheap; avoid O(history) recomputation |
| **Prior on failure rate** | High prior → trigger-happy | Low prior → needs more evidence | Estimate online from observed failure frequency (adaptive prior), initialize moderately |

---

## 6. Phase 4 — Node State Machine (Circuit Breaker)

```
  ACTIVE ──(p_fail/p_degr > τ_suspect)──▶ SUSPECT ──(> τ_quarantine or persists)──▶ QUARANTINED
     ▲                                        │                                          │
     │                                (evidence fades)                          (backoff timer expires)
     └──────────────◀──────────────────────────┘                                          ▼
     ▲                                                                                PROBING (canary tasks)
     └───────(N consecutive healthy canary results)──────────────────────────────────────┘
```

| State | New task policy | In-flight policy |
|---|---|---|
| ACTIVE | Full weight | Leave alone |
| SUSPECT | Reduced weight (or only low-priority / high-slack tasks) | Evaluate per-task risk; migrate only urgent ones |
| QUARANTINED | **Zero** | Migrate at-risk tasks (EV rule); tasks with plenty of slack may be left if node is only degraded |
| PROBING | 1–2 canary tasks with high slack | — |

**Hysteresis:** entry threshold ≠ exit threshold (`τ_exit < τ_enter`), plus **minimum dwell time** in each state, to stop flapping (flapping causes churn — the third scoring criterion).

**Tradeoffs**
- **Probe frequency:** frequent probes recover capacity sooner but sacrifice a few tasks to still-bad nodes. Use **exponential backoff** per node (backoff resets after a successful recovery; grows on repeated failures).
- **Canary selection:** send *sacrificial-safe* tasks (long deadline slack) so a failed probe costs nothing. If nothing has slack, probe less.
- **Permanent vs. transient failure:** unknown; backoff handles both without assuming.
- **Cluster-collapse guard:** never quarantine *all* nodes — if every node looks bad, it's more likely a global event (load spike/telemetry issue). Cap quarantine to e.g. `N − 1` (or rank by health and keep the best).

---

## 7. Phase 5 — Task Placement (Load Balancing)

### 7.1 Options

| Strategy | Description | Pros | Cons |
|---|---|---|---|
| Round-robin (baseline) | Cycle nodes | Trivial | Health-blind — the failure mode to fix |
| Round-robin over healthy set | Skip quarantined | Easy win | Ignores load/capacity/degradation |
| **Least-loaded (JSQ)** | Pick shortest expected wait | Good throughput | Herding: everyone picks the same node in one step |
| **Power of two choices** | Sample 2, pick better | Robust to stale info, low herding | Slight suboptimality; randomness (use seeded RNG) |
| **Weighted by health × capacity** | Probability/score ∝ `health_multiplier × est_service_rate` | Uses degraded nodes proportionally | Needs service-rate estimate |
| **Deadline-aware (EDF / least-slack-first)** | Schedule tightest deadlines first onto best nodes | Maximizes on-time completion | Needs task-size/deadline info; tie-breaks matter |
| **Expected completion-time minimization** | For each task compute `E[finish]` per node = queue_wait/health + service_time/health; choose node maximizing `P(finish ≤ deadline)` | Best when models are decent | Highest complexity; sensitive to model error |
| RL policy | Learned | Adaptive | Same issues as §5.2-I |

**Recommended:** *expected-completion-time with health-discounting*, using **batch assignment within a step** (assign in order of least slack, update virtual queue lengths after each assignment to avoid herding). Fall back to power-of-two if telemetry is missing.

### 7.2 Placement tradeoffs
- **Balance vs. concentration:** spreading load minimizes queueing but exposes more tasks to any single node failure; concentrating on fewer, verified-healthy nodes reduces exposure but risks overload. → Prefer balance, weighted by health; add a **soft cap** per node (e.g., ≤ ~85% estimated capacity) to leave headroom for rerouted work.
- **Reserve headroom for reroutes:** if the cluster is running near 100% capacity, a failure has nowhere to send displaced tasks. Keeping ~10–20% slack trades a bit of steady-state throughput for resilience. **Make it adaptive**: raise headroom when recent failure frequency is high.
- **Shed hopeless tasks?** If reward comes only from on-time completion, a task with `P(on-time) ≈ 0` still consumes capacity. Options: (a) run it anyway (if partial credit or no penalty), (b) deprioritize it, (c) drop/deny it if the API allows. → **Check reward definition in Phase 0**; default to *deprioritize*, not drop.
- **Heterogeneous capacity:** estimate each node's throughput online (tasks/step completed when healthy) rather than assuming equal capacity — the eval env has "different node capacity."
- **Queue depth as feedback:** queue depth is both a *health signal* and a *load signal* — decorrelate them (drain rate vs. expected) so a busy node isn't mistaken for a sick one.

---

## 8. Phase 6 — Reroute Policy (Where the Score Is Won or Lost)

### 8.1 Identify affected tasks
For each in-flight task `i` on node `n`:
- `p_stuck_i` = probability the task won't progress = `p_fail(n)` + partial credit from `p_degraded(n)`; sharpened if the task itself shows stall (zero progress for k steps).
- `slack_i` = time to deadline − expected remaining time at *current* effective speed.

### 8.2 Expected-value reroute rule

```
stay_value   = P(node n survives/keeps pace) × P(finish on time | stay)
move_value   = P(target m healthy)           × P(finish on time | restart on m)
               └─ includes queue wait at m + FULL restart work (cold-restart cost)

reroute iff   (move_value − stay_value) × reward_i  >  λ_churn + margin
```

- `λ_churn` = explicit penalty for churn (tunable); `margin` = hysteresis.
- If the source node has **high `p_fail`**, `stay_value → 0` and the rule fires automatically — no separate "failed ⇒ move everything" special case needed (but include it as a sanity fallback).
- If the source is only **degraded**, tasks with **ample slack** stay; tasks with **tight slack** move first. This is the "don't disrupt tasks running fine" requirement.

### 8.3 Reroute tradeoffs

| Decision | Aggressive | Conservative | Guidance |
|---|---|---|---|
| **When to migrate** | Migrate as soon as suspect | Wait for near-certain failure | EV-based. Because deadlines bound the wait, **the latest safe reroute time = deadline − restart_time − safety** — delaying until then costs nothing *if* you're confident it's still recoverable; waiting also buys more evidence. Use **slack-driven "wait for evidence" for high-slack tasks**, immediate move for low-slack |
| **Order of migration** | — | — | Least-slack first (most at risk), because target capacity is finite |
| **Batch size / rate limit** | Move all at once | Trickle | **Per-step reroute budget** + stagger, to avoid overloading healthy nodes ("thundering herd" onto the best node) |
| **Target selection** | Best node | Spread | Use virtual-queue updates so displaced tasks don't all land on one node |
| **Re-reroute** | Allow repeated moves | Once per task (or cooldown) | Add **cooldown** per task; a task bouncing between nodes pays cold-restart each time |
| **Does moving reset progress fully?** | If yes, tasks nearly complete on a slow node should *stay* | — | Compare `remaining_work/slow_speed` vs `full_work/fast_speed + queue_wait` |
| **Speculative duplication** (if API allows) | Run copy on 2 nodes → no loss | Not available / doubles load | Check whether env allows; likely not (assignment = single node). If allowed, useful only for very-high-value, high-risk tasks |

### 8.4 New-task handling during incidents
When a failure is suspected, the *cheapest* action is to **stop feeding the node** (costs nothing but capacity). Do that at the low threshold (§5.4), long before migrating in-flight work.

---

## 9. Generalization Plan (Sandbox ≠ Evaluation)

The organizers explicitly warn: *different node count, capacity, arrival rate, failure timing/frequency.*

| Risk | Mitigation |
|---|---|
| Hard-coded latency threshold | Peer- and self-relative z-scores; MAD not σ |
| Assumes N nodes | Nothing indexed by `N`; handle N as small as 2–3 (peer stats degrade → fall back to self-baseline and heartbeat); handle N large without O(N²) |
| Assumes capacity | Estimate throughput per node online |
| Assumes arrival rate | Compute utilization online; adapt headroom |
| Assumes failure frequency | Adaptive prior on failure rate; don't hard-code T matrix beyond mild defaults |
| Assumes step length / timing | Express times in units of observed heartbeat interval / median service time |
| Cold start | First K steps: use conservative priors, run in "learning" mode (round-robin among nodes with heartbeat present) while baselines form |
| Failure at t=0 | Peer comparison + heartbeat handle it even without a healthy baseline |
| Overfitting hyper-parameters | **Domain randomization** in your sim; select hyper-params by **worst-case/mean over scenarios**, not best single scenario; check sensitivity (parameter ± 30% shouldn't collapse score) |

**Auto-tuning idea (optional):** run a small random/Bayesian search over `τ_suspect, τ_migrate, λ_churn, headroom, T` on your scenario suite; pick the **flat** optimum (robust region), not the sharpest peak.

---

## 10. Compute Budget

- Per-step cost target: **O(N + T)** where T = tasks in flight. Avoid per-step history rescans (use running/EWMA stats, small ring buffers).
- Keep the filter closed-form; no per-step optimization solvers. Greedy assignment sorted by slack is O((N + T) log T).
- Profile early: time `act()` on the largest scenario (e.g., 64 nodes, thousands of tasks). Add a **watchdog**: if elapsed budget > X%, switch to a cheap fallback (healthy-set weighted round-robin) — partial results are scored, so never risk a timeout.
- No heavyweight imports or model loading in `act`; do it in `__init__`/`reset`.
- Deterministic RNG (seeded in `reset`) for reproducibility.

---

## 11. Interpretability / Bonus Log

Emit a structured line whenever a belief crosses a threshold or a reroute happens:

```json
{"t": 214, "node": 4, "event": "SUSPECT→QUARANTINED",
 "p_fail": 0.93,
 "evidence": {"hb_miss_streak": 5, "lat_z_peer": 6.1, "err_excess": 0.32, "drain_ratio": 0.05},
 "top_signal": "heartbeat",
 "action": "stop_new_traffic; migrate 3 tasks (slack<8)",
 "tasks_moved": [881, 884, 890], "tasks_kept": [877, 879]}
```

- Keep a **rolling explanation buffer** (cap size) and write to file at episode end / on `reset` — avoid per-step I/O in `act` (compute budget).
- Include **"why NOT rerouted"** entries (e.g., "kept task 877: slack 40 > threshold, p_fail 0.3") — demonstrates the low-churn philosophy to judges.
- A small **timeline plot** (belief per node + true failure windows) makes the README/demo far more persuasive.

---

## 12. Master Tradeoff Matrix

| # | Axis | Option A | Option B | Recommendation |
|---|---|---|---|---|
| 1 | Detector family | Heuristic (EWMA/timeout) | Probabilistic (Bayes/HMM) | B, with A features as inputs |
| 2 | Absolute vs. relative signals | Absolute thresholds | Relative (self+peer) | Relative — required for generalization |
| 3 | Binary vs. graded health | Healthy/dead | Continuous capacity multiplier | Graded |
| 4 | Detection speed vs. FP | Trigger-happy | Patient | Asymmetric: quick to stop new traffic, patient to migrate |
| 5 | Migration trigger | Threshold on p_fail | EV vs. cold-restart cost | EV (+ threshold sanity floor) |
| 6 | Migration timing | Immediate | Slack-deferred | Slack-driven |
| 7 | Reroute volume | Bulk | Rate-limited & ordered by slack | Rate-limited |
| 8 | Recovery | Timer-only | Canary-probe w/ backoff | Canary + exponential backoff |
| 9 | Placement | RR-healthy | Completion-time-min w/ health & headroom | Latter, batch with virtual queues |
| 10 | Headroom | Max utilization | Reserve capacity | Adaptive reserve |
| 11 | Hopeless tasks | Run all | Deprioritize/shed | Deprioritize (check reward rules) |
| 12 | Learning | Pure rule/Bayes | RL / supervised | Rules+Bayes first; optional learned re-ranker on relative features |
| 13 | Complexity | Sophisticated | Simple & robust | Simple core + ablation-proven add-ons |
| 14 | Hyper-param selection | Best-peak | Flat/robust region | Robust region across randomized scenarios |
| 15 | Logging | Verbose | Buffered/light | Buffered structured events |
| 16 | Global-event handling | Treat as node failures | Detect via peer consensus & cap quarantines | Peer consensus + cap |
| 17 | Cold-start | Assume healthy | Assume unknown, learn | Learn w/ conservative priors |
| 18 | Compute | Rich models per step | O(1)/node filters | O(1)/node + watchdog fallback |

---

## 13. Edge-Case / Failure-Mode Checklist

- [ ] **Cluster-wide slowdown** (load spike) — must *not* quarantine everything.
- [ ] **Multiple simultaneous failures** — do remaining nodes have capacity? Shed/deprioritize gracefully.
- [ ] **Flapping node** (up/down/up) — hysteresis, dwell time, growing backoff.
- [ ] **Gray failure** (heartbeats fine, tasks stalling/erroring) — need progress/error/drain signals, not just heartbeat.
- [ ] **Heartbeat drops on a healthy node** (noise) — soft evidence, not proof.
- [ ] **Silent node with stale telemetry** — missing data escalates suspicion over time.
- [ ] **Slow-burn degradation** — CUSUM/multi-scale catches what z-score misses; baseline freeze prevents drift-absorption.
- [ ] **Recovery mid-migration** — don't move tasks *back*; cooldown per task.
- [ ] **N very small (2–3)** — peer stats unreliable; fallback to self-baseline.
- [ ] **All nodes suspect** — pick least-bad; never assign to nobody (unassigned = missed deadline).
- [ ] **Burst arrivals** — queues spike legitimately; queue-depth alone must not trigger failure.
- [ ] **Tasks with already-impossible deadlines** — deprioritize.
- [ ] **Task appears/disappears in list unexpectedly** — defensive coding; no KeyErrors killing the episode (an exception = partial score!). Wrap `act` in try/except with fallback assignments.
- [ ] **Non-determinism / RNG** — seed in `reset`.
- [ ] **Episode restarts** — `reset()` fully clears state.

---

## 14. Suggested Timeline (adapt to hackathon length)

| Block | Work | Exit criterion |
|---|---|---|
| **H0–2** | Phase 0 recon; run baseline demo; dump telemetry | Interface cheat sheet + telemetry dump |
| **H2–5** | Harness: metrics, scenario suite, oracle agent | One-command benchmark table |
| **H5–8** | **MVP agent**: heartbeat timeout + healthy-set weighted least-loaded; safe try/except fallback | Beats baseline substantially on reward |
| **H8–14** | Detector v2: relative features + Bayesian filter + baseline freeze | Detection latency ↓ with churn flat |
| **H14–18** | Circuit breaker + canary recovery + hysteresis | No flapping; capacity recovered |
| **H18–24** | EV-based reroute + slack ordering + rate limits + headroom | Churn ↓, reward ↑ vs. threshold reroute |
| **H24–30** | Generalization sweep, domain randomization, hyper-param robust search, ablations | Stable across held-out scenarios |
| **H30–34** | Compute profiling, watchdog, edge-case checklist | Fits budget on largest scenario |
| **H34–38** | Interpretability log + belief timeline plot | Readable "why" trace |
| **H38–40** | README (≤1 page), packaging, fresh-env import test | `import MyAgent` works from clean zip |

**If time is short**, priority order (highest value first):
1. Health-aware placement (stop feeding dead nodes) →
2. Heartbeat + relative latency/error features →
3. Reroute in-flight tasks from confirmed-dead nodes, least-slack-first →
4. Hysteresis + canary recovery →
5. EV-based reroute & headroom →
6. Bayesian filter / CUSUM upgrades →
7. Interpretability log.

---

## 15. Agent Skeleton (Pseudocode)

```python
class MyAgent(BaseAgent):
    def reset(self, *args, **kw):
        self.rng = np.random.default_rng(SEED)
        self.nodes = {}            # node_id -> NodeModel (filters, baselines, breaker state)
        self.task_cooldown = {}    # task_id -> last_reroute_step
        self.log = RingBuffer(5000)
        self.t = 0

    def update(self, telemetry):   # state estimation (confirm which method receives obs)
        self.t += 1
        peer = robust_peer_stats(telemetry)               # median/MAD across nodes
        for nid, obs in telemetry.items():
            m = self.nodes.setdefault(nid, NodeModel())
            feats = m.extract(obs, peer)                  # hb suspicion, lat_z_self/peer, err_excess, drain, stall
            m.belief = m.filter(feats)                    # Bayesian forward step (missing => neutral/soft)
            m.breaker.step(m.belief, self.t)              # ACTIVE/SUSPECT/QUARANTINED/PROBING w/ hysteresis
            if m.state == HEALTHY: m.update_baseline(obs) # freeze when suspicious
        self.log_events()

    def act(self, obs):
        try:
            tasks = obs.tasks
            eff_cap = {n: m.health_multiplier * m.est_rate for n, m in self.nodes.items()}
            vq = {n: obs.queue_depth[n] for n in self.nodes}   # virtual queues

            # 1) Reroute in-flight tasks (budgeted, least slack first)
            at_risk = sorted(inflight(tasks), key=slack)
            moves, budget = [], self.reroute_budget(obs)
            for task in at_risk:
                if budget == 0: break
                if on_cooldown(task): continue
                src, tgt = task.node, best_target(task, eff_cap, vq)
                if ev_gain(task, src, tgt) > self.lambda_churn + self.margin:
                    moves.append((task, tgt)); vq[tgt] += 1; budget -= 1
                    self.explain_move(task, src, tgt)

            # 2) Place new tasks (least slack first, completion-time minimizing, headroom cap)
            placements = []
            for task in sorted(unassigned(tasks), key=slack):
                tgt = argmax_p_on_time(task, eligible_nodes(), eff_cap, vq)
                placements.append((task, tgt)); vq[tgt] += 1

            # 3) Canary probes for PROBING nodes (high-slack tasks only)
            placements += self.canaries(tasks, vq)
            return build_actions(moves, placements)
        except Exception:
            return self.safe_fallback(obs)   # healthy-set weighted round-robin; never crash
```

---

## 16. README (≤1 page) Outline

1. **Idea in one paragraph** — probabilistic per-node health filter from *relative* telemetry features; graded capacity; EV-based rerouting.
2. **Detection signal** — heartbeat suspicion (phi-accrual-like), self- and peer-relative latency z-scores (median/MAD), error-rate excess, queue drain ratio, (task stall); Bayesian fusion; baseline freeze.
3. **Adaptation once suspected** — asymmetric thresholds (stop new traffic early / migrate on EV), slack-ordered, rate-limited rerouting, circuit breaker with canary probes and exponential backoff.
4. **Why it generalizes** — no absolute thresholds, online capacity/arrival estimation, adaptive failure prior, tested on randomized scenarios (N, capacity, load, failure patterns).
5. **Results table** — baseline vs. yours vs. oracle: completion rate, detection latency, churn.
6. **Interpretability** — sample log line + timeline figure.
7. **Limitations / future work** — e.g., learned re-ranker, BOCPD.

---

## 17. Open Questions to Answer from the Sandbox Docs

1. Is task progress observable per step? (Enables stall detection.)
2. What exactly does "reroute" look like in the action format — re-assigning an already-running task's node?
3. Is the cold-restart cost a fixed penalty or "remaining work resets to full"?
4. Reward function: per on-time completion only? Any penalty for failed/late tasks or reroutes?
5. Does telemetry arrive every step for every node, or can it be missing?
6. Are failures modeled as: crash, slowdown, error injection, or all three? Do degraded nodes reduce capacity or only add latency/errors?
7. Are there any hidden per-step time limits on `act`?
8. Is ground truth exposed in practice mode for offline scoring only?
9. Can tasks be left unassigned/queued at the controller (deferred assignment), or must every task be assigned immediately? (Deferral is a powerful lever if allowed.)
10. What are the `BaseAgent` return types/exceptions on invalid assignments (e.g., assigning to a dead node — silently dropped or penalized)?

---

*Bottom line:* win by (1) never sending new work into a suspected node, (2) migrating in-flight work only when the deadline-aware expected value beats the cold-restart cost, and (3) proving generalization with randomized scenarios — all backed by a probabilistic, relative-signal health model rather than tuned thresholds.
