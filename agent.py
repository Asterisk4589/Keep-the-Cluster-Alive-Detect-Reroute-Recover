"""
MM26AI02 - Keep the Cluster Alive: Detect, Reroute, Recover

Submission agent for the supplied BaseAgent interface.

Design:
- Relative/self-normalized telemetry rather than fixed environment thresholds.
- Per-node graded health belief with a small Bayesian-style 3-state filter.
- Asymmetric response: stop NEW traffic early; reroute IN-FLIGHT work only
  when deadline slack and estimated benefit justify the cold restart.
- Circuit breaker with hysteresis and recovery probing.
- Deadline/slack-aware placement with virtual queue accounting.
- Defensive fallback so a telemetry edge case never crashes an evaluation.

The agent only uses fields exposed by agent_interface.py. It never reads
debug ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional
import math

from agent_interface import BaseAgent


EPS = 1e-6


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _mad(values: List[float], center: Optional[float] = None) -> float:
    if not values:
        return 1.0
    c = _median(values) if center is None else center
    d = [abs(x - c) for x in values]
    return _median(d)


def _sigmoid(x: float) -> float:
    x = _clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class NodeModel:
    node_id: int
    capacity: float = 1.0

    # Robust healthy baselines. These are frozen while the node is suspicious.
    lat_samples: deque = field(default_factory=lambda: deque(maxlen=31))
    err_samples: deque = field(default_factory=lambda: deque(maxlen=31))
    load_samples: deque = field(default_factory=lambda: deque(maxlen=31))
    hb_samples: deque = field(default_factory=lambda: deque(maxlen=31))

    # Health belief: healthy, degraded, failed.
    p_healthy: float = 0.85
    p_degraded: float = 0.10
    p_failed: float = 0.05

    state: str = "ACTIVE"
    suspect_streak: int = 0
    healthy_streak: int = 0
    probe_backoff: int = 0
    probe_wait: int = 0
    last_reason: str = ""

    last_load: int = 0

    def belief_tuple(self):
        return self.p_healthy, self.p_degraded, self.p_failed

    @property
    def p_bad(self) -> float:
        return _clip(self.p_degraded + self.p_failed, 0.0, 1.0)

    @property
    def health_multiplier(self) -> float:
        # Healthy=1, degraded~0.55, failed~0.03. A tiny non-zero value lets
        # the planner keep a "least bad" node when the cluster is collapsing.
        return (
            self.p_healthy * 1.0
            + self.p_degraded * 0.55
            + self.p_failed * 0.03
        )

    @property
    def effective_rate(self) -> float:
        return max(0.10, self.health_multiplier)


class MyAgent(BaseAgent):
    """
    Autonomous fault-aware scheduler.

    Required interface:
        reset()
        act(obs) -> {task_id: node_id}
        update(obs, reward, done, info)
    """

    def __init__(self, n_nodes: int, node_capacity: int):
        super().__init__(n_nodes, node_capacity)
        self.reset()

    def reset(self) -> None:
        self.step_no = 0
        self.nodes: Dict[int, NodeModel] = {}
        self.hb_streak: Dict[int, int] = {}
        self.task_reroute_cooldown: Dict[int, int] = {}
        self.task_total: Dict[int, float] = {}
        self.prev_task: Dict[int, tuple] = {}
        self.prog_ref = 1.0
        self.node_prog: Dict[int, list] = {}
        self.events = deque(maxlen=1000)
        self.last_obs = None

        # Relative, dimensionless controls. They are deliberately broad rather
        # than tied to the sandbox's latency/error/capacity numbers.
        self.peer_weight = 0.55
        self.self_weight = 0.45
        self.new_task_bad_limit = 0.58
        self.quarantine_limit = 0.82
        self.recover_limit = 0.25

        # Cold restart is expensive in this environment because duration resets.
        self.churn_margin = 0.10
        self.max_reroutes = max(1, int(round(self.n_nodes * 0.75)))
        self.reroute_cooldown = 3

    # ------------------------------------------------------------------
    # Telemetry / health estimation
    # ------------------------------------------------------------------

    def _ensure_nodes(self, obs: dict) -> None:
        for raw in obs.get("nodes", []):
            nid = int(raw["node_id"])
            if nid not in self.nodes:
                self.nodes[nid] = NodeModel(
                    node_id=nid,
                    capacity=max(1.0, float(raw.get("capacity", self.node_capacity))),
                )

    def _peer_stats(self, nodes: List[dict]):
        lats = []
        errs = []
        queues = []
        for raw in nodes:
            lat = raw.get("latency_ms")
            try:
                lat = float(lat) if lat is not None else None
            except (TypeError, ValueError):
                lat = None
            if lat is not None and math.isfinite(lat):
                lats.append(lat)
            try:
                err = float(raw.get("error_rate", 0.0))
            except (TypeError, ValueError):
                err = 0.0
            errs.append(_clip(err, 0.0, 1.0))
            try:
                q = int(raw.get("queue_len", 0))
            except (TypeError, ValueError):
                q = 0
            queues.append(max(0, q))

        lat_med = _median(lats) if lats else 0.0
        err_med = _median(errs) if errs else 0.0
        q_med = _median(queues) if queues else 0.0

        # Floors are relative to the observed scale, so unit changes such as
        # milliseconds -> seconds do not silently disable the detector.
        lat_floor = max(EPS, 0.05 * max(abs(lat_med), EPS))
        err_floor = max(0.0025, 0.05 * max(abs(err_med), 0.05))
        q_floor = max(1.0, 0.10 * max(abs(q_med), 1.0))
        return {
            "lat_med": lat_med,
            "lat_mad": max(lat_floor, _mad(lats)),
            "err_med": err_med,
            "err_mad": max(err_floor, _mad(errs)),
            "queue_med": q_med,
            "queue_mad": max(q_floor, _mad(queues)),
        }

    def _z(self, value: float, center: float, scale: float) -> float:
        # MAD is robust to heavy tails and one bad node.
        return (value - center) / max(scale, EPS)

    def _update_one(self, raw: dict, peer: dict, tasks: List[dict]) -> None:
        nid = int(raw["node_id"])
        m = self.nodes[nid]

        hb = bool(raw.get("heartbeat_ok", False))
        try:
            lat = float(raw.get("latency_ms")) if raw.get("latency_ms") is not None else None
        except (TypeError, ValueError):
            lat = None
        try:
            err = _clip(float(raw.get("error_rate", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            err = 0.0
        try:
            queue = max(0, int(raw.get("queue_len", 0)))
        except (TypeError, ValueError):
            queue = 0
        if lat is not None and not math.isfinite(lat):
            lat = None

        # Per-node self baselines are only updated when the node is healthy
        # enough, avoiding "boiling frog" baseline contamination.
        if m.p_bad < 0.35:
            if lat is not None:
                m.lat_samples.append(lat)
            m.err_samples.append(err)
            m.load_samples.append(float(queue))
            m.hb_samples.append(1.0 if hb else 0.0)

        self_lat = _median(list(m.lat_samples)) if m.lat_samples else peer["lat_med"]
        self_lat_mad = max(2.0, _mad(list(m.lat_samples), self_lat))
        self_err = _median(list(m.err_samples)) if m.err_samples else peer["err_med"]
        self_err_mad = max(0.01, _mad(list(m.err_samples), self_err))

        lat_peer_z = 0.0
        lat_self_z = 0.0
        if lat is None:
            # Missing latency is evidence, but heartbeat remains the stronger
            # signal. Do not convert one missing field directly into failure.
            lat_peer_z = 2.0
            lat_self_z = 2.0
        else:
            lat_peer_z = self._z(lat, peer["lat_med"], peer["lat_mad"])
            lat_self_z = self._z(lat, self_lat, self_lat_mad)

        err_peer_z = self._z(err, peer["err_med"], peer["err_mad"])
        err_self_z = self._z(err, self_err, self_err_mad)

        # Consecutive heartbeat misses are more meaningful than one miss.
        if not hb:
            miss_streak = self.hb_streak.get(nid, 0) + 1
        else:
            miss_streak = 0
        self.hb_streak[nid] = miss_streak

        hb_score = _clip(miss_streak / 3.0, 0.0, 1.0)
        lat_score = _clip(max(lat_peer_z, lat_self_z) / 5.0, 0.0, 1.0)
        err_score = _clip(max(err_peer_z, err_self_z) / 5.0, 0.0, 1.0)

        # Queue length is mostly caused by our own placement decisions in this
        # environment, so it is deliberately excluded from health evidence.
        evidence = (
            0.47 * hb_score
            + 0.29 * lat_score
            + 0.24 * err_score
        )

        # Gray-failure amplification: multiple independent symptoms matter more
        # than one noisy spike.
        symptom_count = sum(
            x >= 0.55 for x in (hb_score, lat_score, err_score)
        )
        if symptom_count >= 2:
            evidence = min(1.0, evidence + 0.10)

        # Bayesian-style forward prediction with sticky states.
        ph, pd, pf = m.belief_tuple()
        prior_h = 0.985 * ph + 0.10 * pd + 0.02 * pf
        prior_d = 0.012 * ph + 0.85 * pd + 0.03 * pf
        prior_f = 0.003 * ph + 0.05 * pd + 0.95 * pf

        # State likelihoods are intentionally broad. The relative evidence
        # score is the main measurement, while the transition model supplies
        # temporal persistence.
        fail_l = 0.20 + 2.20 * evidence
        deg_l = 0.55 + 1.10 * evidence
        healthy_l = 1.25 - 0.95 * evidence

        # A clean heartbeat is evidence against failure, but not proof.
        if hb and lat is not None and err_score < 0.25:
            fail_l *= 0.65
            healthy_l *= 1.08

        prog = self.node_prog.get(nid, [])
        for x in prog[:3]:
            r = x / max(EPS, self.prog_ref)
            if r >= 0.9:
                healthy_l *= 3.0
                fail_l *= 0.15
                deg_l *= 0.3
            elif r <= 0.05:
                healthy_l *= 0.4
                fail_l *= 1.8
                deg_l *= 1.4
            else:
                healthy_l *= 0.2
                fail_l *= 0.3
                deg_l *= 3.0
        post_h = prior_h * healthy_l
        post_d = prior_d * deg_l
        post_f = prior_f * fail_l
        total = max(EPS, post_h + post_d + post_f)
        m.p_healthy = post_h / total
        m.p_degraded = post_d / total
        m.p_failed = post_f / total


        self._update_breaker(m, evidence, hb, lat)

        return

    def _update_breaker(self, m: NodeModel, evidence: float, hb: bool, lat) -> None:
        old = m.state
        bad = m.p_bad

        if bad >= self.quarantine_limit:
            m.suspect_streak += 1
        elif bad >= self.new_task_bad_limit or evidence >= 0.55:
            m.suspect_streak += 1
        else:
            m.suspect_streak = max(0, m.suspect_streak - 1)

        healthy_signal = (
            hb and lat is not None and bad <= self.recover_limit
        )
        if healthy_signal:
            m.healthy_streak += 1
        else:
            m.healthy_streak = 0

        if m.state == "ACTIVE":
            if m.p_bad >= self.new_task_bad_limit or m.suspect_streak >= 2:
                m.state = "SUSPECT"
                m.last_reason = "multi-signal suspicion"
        elif m.state == "SUSPECT":
            if m.p_bad >= self.quarantine_limit or m.suspect_streak >= 3:
                m.state = "QUARANTINED"
                m.probe_wait = max(2, m.probe_backoff)
                m.last_reason = "persistent/high failure belief"
            elif healthy_signal and m.healthy_streak >= 3:
                m.state = "ACTIVE"
        elif m.state == "QUARANTINED":
            if m.probe_wait > 0:
                m.probe_wait -= 1
            elif healthy_signal:
                m.state = "PROBING"
                m.healthy_streak = 0
        elif m.state == "PROBING":
            if not hb or lat is None or bad > 0.45:
                m.state = "QUARANTINED"
                m.probe_backoff = min(32, max(2, 2 * max(1, m.probe_backoff)))
                m.probe_wait = m.probe_backoff
            elif m.healthy_streak >= 3:
                m.state = "ACTIVE"
                m.probe_backoff = 0
                m.probe_wait = 0

        if m.state == 'ACTIVE' and old != 'ACTIVE':
            m.suspect_streak = 0
        m.suspect_streak = min(m.suspect_streak, 6)
        if old != m.state:
            self.events.append({
                "t": self.step_no,
                "node": m.node_id,
                "event": old + "->" + m.state,
                "p_fail": round(m.p_failed, 3),
                "p_degraded": round(m.p_degraded, 3),
                "reason": m.last_reason,
            })

    def _update_from_obs(self, obs: dict) -> None:
        self._ensure_nodes(obs)
        self.node_prog = {}
        cur = {}
        for t in obs.get("tasks", []):
            tid = int(t["task_id"]); node = t.get("node"); rem = float(t.get("duration_remaining", 0.0))
            cur[tid] = (node, rem)
            pv = self.prev_task.get(tid)
            if node is not None and pv is not None and pv[0] == node:
                self.node_prog.setdefault(int(node), []).append(pv[1] - rem)
        for lst in self.node_prog.values():
            for x in lst: self.prog_ref = max(self.prog_ref, x)
        self.prev_task = cur
        for tid in list(self.task_total):
            if tid not in cur: self.task_total.pop(tid, None)
        observed = obs.get("nodes", [])
        peer = self._peer_stats(observed)
        tasks = obs.get("tasks", [])
        seen = set()
        for raw in observed:
            try:
                nid = int(raw["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            seen.add(nid)
            try:
                self._update_one(raw, peer, tasks)
            except Exception as exc:
                self.events.append({
                    "t": self.step_no, "node": nid, "event": "TELEMETRY_ERROR",
                    "reason": type(exc).__name__,
                })
        # An absent node is treated as a heartbeat miss rather than silently
        # retaining stale health indefinitely.
        for nid in self.nodes:
            if nid not in seen:
                try:
                    self._update_one(
                        {"node_id": nid, "heartbeat_ok": False,
                         "latency_ms": None, "error_rate": 1.0,
                         "queue_len": self.nodes[nid].last_load},
                        peer, tasks,
                    )
                except Exception as exc:
                    self.events.append({
                        "t": self.step_no, "node": nid, "event": "TELEMETRY_ERROR",
                        "reason": type(exc).__name__,
                    })

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def _current_loads(self, obs: dict) -> Dict[int, int]:
        loads = {nid: 0 for nid in self.nodes}
        for t in obs.get("tasks", []):
            node = t.get("node")
            if node is not None and int(node) in loads:
                loads[int(node)] += 1
        return loads

    def _task_slack(self, task: dict, node_id: Optional[int] = None, loads=None) -> float:
        # Environment step units are the only reliable time unit exposed.
        # Estimate completion using remaining work and node effective rate.
        now = self.step_no
        deadline = int(task.get("deadline", now))
        rem = max(0.0, float(task.get("duration_remaining", 0.0)))
        if node_id is None or node_id not in self.nodes:
            speed = 1.0
        else:
            speed = self.nodes[node_id].effective_rate
        queue_wait = 0.0
        if loads is not None and node_id in loads:
            cap = max(1.0, self.nodes[node_id].capacity)
            queue_wait = max(0.0, loads[node_id] / cap)
        return (deadline - now) - (rem / max(0.10, speed)) - queue_wait

    def _eligible_nodes(self, loads: Dict[int, int], allow_suspect: bool = False) -> List[int]:
        candidates = []
        for nid, m in self.nodes.items():
            if loads.get(nid, 0) >= int(round(m.capacity)):
                continue
            if m.state == "QUARANTINED":
                continue
            if m.state == "SUSPECT" and not allow_suspect:
                continue
            candidates.append(nid)
        if candidates:
            return candidates

        fallback = [
            nid for nid, m in self.nodes.items()
            if loads.get(nid, 0) < int(round(m.capacity))
        ]
        return fallback

    def _placement_score(self, task: dict, nid: int, loads: Dict[int, int]) -> float:
        m = self.nodes[nid]
        rem = max(0.0, float(task.get("duration_remaining", 0.0)))
        deadline = int(task.get("deadline", self.step_no))
        cap = max(1.0, m.capacity)
        virtual_wait = loads[nid] / cap
        service = rem / max(0.10, m.effective_rate)
        finish = self.step_no + virtual_wait + service
        slack_after = deadline - finish

        # Main objective: deadline probability proxy + health + load balance.
        deadline_score = _sigmoid(slack_after / 2.0)
        health_score = 0.60 * m.p_healthy + 0.30 * (1.0 - m.p_failed) + 0.10 * m.health_multiplier
        load_penalty = loads[nid] / cap
        return 2.2 * deadline_score + 1.0 * health_score - 0.55 * load_penalty

    def _best_target(self, task: dict, loads: Dict[int, int], exclude: Optional[int] = None) -> Optional[int]:
        candidates = self._eligible_nodes(loads)
        if exclude is not None:
            other = [n for n in candidates if n != exclude]
            if other:
                candidates = other
        if not candidates:
            return None
        return max(candidates, key=lambda n: self._placement_score(task, n, loads))

    def _reroute_value(self, task: dict, src: int, tgt: int, loads: Dict[int, int]) -> float:
        sm = self.nodes[src]
        tm = self.nodes[tgt]

        rem = max(0.0, float(task.get("duration_remaining", 0.0)))
        total = max(rem, self.task_total.get(int(task['task_id']), rem))
        deadline = int(task.get("deadline", self.step_no))

        # Stay: probability current node remains usable times chance to finish.
        stay_speed = max(0.10, sm.effective_rate)
        stay_finish = self.step_no + rem / stay_speed + loads.get(src, 0) / max(1.0, sm.capacity)
        stay_slack = deadline - stay_finish
        p_stay_on_time = _sigmoid(stay_slack / 1.8)
        stay_value = (1.0 - sm.p_failed) * p_stay_on_time

        # Move: cold restart means FULL original work, which is exactly what
        # the supplied environment does when assigning to a different node.
        restart_work = max(rem, total)
        move_speed = max(0.10, tm.effective_rate)
        move_wait = loads.get(tgt, 0) / max(1.0, tm.capacity)
        move_finish = self.step_no + move_wait + restart_work / move_speed
        move_slack = deadline - move_finish
        p_move_on_time = _sigmoid(move_slack / 1.8)
        move_value = (1.0 - tm.p_failed) * p_move_on_time

        # The gain is normalized. The explicit source-failure term makes a
        # strongly failed node evacuate even if its task has some slack.
        gain = move_value - stay_value
        if sm.p_failed > 0.80:
            gain += 0.20 * sm.p_failed
        return gain

    def _safe_fallback(self, obs: dict) -> dict:
        try:
            loads = self._current_loads(obs)
            actions = {}
            for task in obs.get("tasks", []):
                if task.get("node") is not None:
                    continue
                target = self._best_target(task, loads)
                if target is not None:
                    actions[int(task["task_id"])] = target
                    loads[target] += 1
            return actions
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    def act(self, obs: dict) -> dict:
        try:
            self._ensure_nodes(obs)
            if not self.nodes:
                return {}

            loads = self._current_loads(obs)
            tasks = list(obs.get("tasks", []))
            for _t in tasks:
                _id = int(_t['task_id']); _r = float(_t.get('duration_remaining', 0.0))
                self.task_total[_id] = max(self.task_total.get(_id, 0.0), _r)
            actions: Dict[int, int] = {}

            # 1) Reroute only the most deadline-sensitive tasks first.
            #    This preserves progress on tasks that can safely wait.
            inflight = [t for t in tasks if t.get("node") is not None]
            inflight.sort(key=lambda t: self._task_slack(t, int(t["node"]), loads))

            moved = 0
            for task in inflight:
                if moved >= self.max_reroutes:
                    break

                tid = int(task["task_id"])
                src = int(task["node"])
                if src not in self.nodes:
                    continue

                if self.task_reroute_cooldown.get(tid, -999999) + self.reroute_cooldown > self.step_no:
                    continue

                sm = self.nodes[src]

                # Healthy/low-risk nodes are left alone. This is important
                # because moving resets the task to full duration.
                if sm.p_bad < 0.48 and sm.state == "ACTIVE":
                    continue

                tgt = self._best_target(task, loads, exclude=src)
                if tgt is None:
                    continue

                gain = self._reroute_value(task, src, tgt, loads)
                slack = self._task_slack(task, src, loads)

                # Asymmetric migration policy:
                # - very bad source -> migrate even with moderate gain
                # - otherwise require EV gain, especially for high-slack work
                urgent = slack <= 1.5
                very_bad = sm.p_failed >= 0.80 or sm.state == "QUARANTINED"

                should_move = (
                    very_bad and (urgent or gain > 0.04)
                ) or (
                    urgent and gain > 0.06
                ) or (
                    gain > (self.churn_margin + 0.04)
                )

                if should_move:
                    actions[tid] = tgt
                    loads[src] = max(0, loads.get(src, 0) - 1)
                    loads[tgt] = loads.get(tgt, 0) + 1
                    self.task_reroute_cooldown[tid] = self.step_no
                    moved += 1
                    self.events.append({
                        "t": self.step_no,
                        "node": src,
                        "event": "REROUTE",
                        "task": tid,
                        "target": tgt,
                        "p_fail": round(sm.p_failed, 3),
                        "slack": round(slack, 2),
                        "gain": round(gain, 3),
                    })

            # 2) New/pending tasks: assign tight deadlines first. Never feed
            #    SUSPECT/QUARANTINED nodes unless the cluster has no alternative.
            pending = [t for t in tasks if t.get("node") is None]
            def _admission_key(t):
                dl = int(t.get("deadline", self.step_no))
                rem = max(0.0, float(t.get("duration_remaining", 0.0)))
                laxity = dl - self.step_no - rem
                infeasible = 1 if laxity < 0 else 0
                return (infeasible, laxity, int(t.get("task_id", 0)))
            pending.sort(key=_admission_key)

            for task in pending:
                target = self._best_target(task, loads)
                if target is None:
                    # If all nodes are saturated, leave the task pending; the
                    # environment will apply only a tiny holding cost.
                    continue
                actions[int(task["task_id"])] = target
                loads[target] += 1

            return actions

        except Exception:
            return self._safe_fallback(obs)

    def update(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        try:
            self.step_no = int(obs.get("step", self.step_no + 1))
            self._update_from_obs(obs)

            # Expire cooldown entries to keep memory bounded.
            cutoff = self.step_no - 20
            for tid, ts in list(self.task_reroute_cooldown.items()):
                if ts < cutoff:
                    del self.task_reroute_cooldown[tid]

            self.last_obs = obs

            if done:
                self.events.append({
                    "t": self.step_no,
                    "event": "EPISODE_END",
                    "reward": round(float(reward), 3),
                })
        except Exception as exc:
            self.events.append({
                "t": self.step_no,
                "event": "UPDATE_ERROR",
                "reason": type(exc).__name__,
            })

    def get_diagnostics(self) -> List[dict]:
        """Optional: inspect buffered explanations after an episode."""
        return list(self.events)
