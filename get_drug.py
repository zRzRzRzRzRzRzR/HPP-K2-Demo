import os
import json
import math
import itertools
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class Edge:
    def __init__(self, e: dict):
        self.id = e.get("edge_id")
        self.src = e.get("from")
        self.dst = e.get("to")
        self.effect = e.get("effect") or {}
        self.scale = self.effect.get("scale")
        self.estimate = self.effect.get("estimate")
        self.ts = e.get("time_semantics") or {}
        self.horizon = self.ts.get("time_horizon")
        self.raw = e


class Graph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Edge] = []
        self.succ: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, n: dict):
        nid = n.get("node_id")
        if nid and nid not in self.nodes:
            self.nodes[nid] = n

    def add_edge(self, e: dict, numeric_only: bool = True):
        ed = Edge(e)
        if not ed.src or not ed.dst:
            return
        if numeric_only:
            if ed.estimate is None:
                return
            if isinstance(ed.estimate, (int, float)):
                val = float(ed.estimate)
            elif isinstance(ed.estimate, str):
                s = ed.estimate.strip()
                if not s:
                    return
                for c in s:
                    if c not in "0123456789+-.eE":
                        return
                val = float(s)
            else:
                return
            ed.estimate = val
        self.edges.append(ed)
        self.succ[ed.src].append(ed)


def load_graph(node_file: str, edge_file: str) -> Graph:
    g = Graph()
    node_data = load_json(node_file)
    if isinstance(node_data, dict):
        nodes_iter = node_data.get("node") or node_data.get("nodes") or node_data.get("data") or []
    else:
        nodes_iter = node_data
    for n in nodes_iter:
        g.add_node(n)
    edge_data = load_json(edge_file)
    if isinstance(edge_data, dict):
        edges_iter = edge_data.get("edge") or edge_data.get("edges") or edge_data.get("data") or []
    else:
        edges_iter = edge_data
    for e in edges_iter:
        g.add_edge(e, numeric_only=True)
    return g


def parse_targets(step6: dict) -> Dict[str, dict]:
    tmeta: Dict[str, dict] = {}
    for t in step6.get("targets", []):
        nid = t.get("node_id")
        if not nid:
            continue
        tr = None
        tr_raw = t.get("target_range")
        if isinstance(tr_raw, dict):
            tr = [tr_raw.get("lowerBound"), tr_raw.get("upperBound")]
        tmeta[nid] = {
            "label": t.get("label") or nid,
            "unit": t.get("unit"),
            "current": t.get("current"),
            "target_range": tr,
            "priority": float(t.get("priority", t.get("priority_score", 0.5))),
        }
    return tmeta


def parse_regimens(step7: dict, topk: int) -> List[dict]:
    regs = step7.get("regimens", []) or []
    regs = sorted(
        regs,
        key=lambda r: (
            -float((r.get("score") or {}).get("overall", 0.0) or 0.0),
            -float((r.get("score") or {}).get("efficacy", 0.0) or 0.0),
        ),
    )
    return regs[:topk]


def to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        for c in s:
            if c not in "0123456789+-.eE":
                return None
        return float(s)
    return None


def regimen_class_of(drug_name: str, regimen: dict) -> Optional[str]:
    for d in regimen.get("drugs", []) or []:
        nm = d.get("name") or d.get("drug")
        if nm == drug_name:
            cls = d.get("class")
            if cls:
                return str(cls).upper()
    return None


def read_drug_kb_to_actions(kb: dict) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    entries: List[dict] = []
    if isinstance(kb.get("drugs"), list):
        entries = kb["drugs"]
    elif isinstance(kb.get("entries"), list):
        entries = kb["entries"]
    else:
        for k, v in kb.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        u = dict(it)
                        if "_name" not in u:
                            u["_name"] = k
                        entries.append(u)
            elif isinstance(v, dict):
                u = dict(v)
                if "_name" not in u:
                    u["_name"] = k
                entries.append(u)
    for it in entries:
        nm = it.get("name") or it.get("_name") or it.get("drugName")
        if not nm:
            continue
        cls = it.get("class") or it.get("drugClass")
        acts: List[dict] = out.setdefault(nm, [])
        payloads: List[dict] = []
        for key in ("targets", "effects", "indicators", "actions"):
            if isinstance(it.get(key), list):
                payloads.extend(it[key])
        if not payloads:
            payloads = [it]
        sfx_list = []
        for sx in it.get("sideEffects", []) or []:
            sfx_list.append(
                {
                    "node": sx.get("affectedNodeId") or sx.get("node_id"),
                    "trend": (sx.get("expectedTrend") or sx.get("trend") or "").lower(),
                    "mean": sx.get("expectedMeanChange") or sx.get("mean"),
                    "unit": sx.get("unit"),
                }
            )
        for p in payloads:
            node = p.get("node_id") or p.get("indicator") or p.get("target")
            delta = p.get("delta") or p.get("expectedMeanChange") or p.get("mean")
            tf = p.get("timeframe") or p.get("window")
            dv = to_float(delta)
            if node and dv is not None:
                acts.append(
                    {
                        "node_id": node,
                        "delta": dv,
                        "timeframe": tf,
                        "class": cls,
                        "side_effects": list(sfx_list),
                        "ttb_days_hint": None,
                    }
                )
    return {k: v for k, v in out.items() if v}


def build_on_target_from_step7_and_kb(regimen: dict, step6: dict, drug_kb: dict) -> Dict[str, List[dict]]:
    class_defaults = {
        "ARB": {"node": "CARDIO:SBP", "delta": -10.0, "timeframe": "P90D"},
        "ACEI": {"node": "CARDIO:SBP", "delta": -10.0, "timeframe": "P90D"},
        "CCB": {"node": "CARDIO:SBP", "delta": -9.0, "timeframe": "P90D"},
        "THIAZIDE": {"node": "CARDIO:SBP", "delta": -8.0, "timeframe": "P30D"},
        "BETA-BLOCKER": {"node": "CARDIO:SBP", "delta": -7.0, "timeframe": "P30D"},
        "STATIN": {"node": "CARDIO:LDL_C", "delta": -1.2, "timeframe": "P90D"},
        "PCSK9I": {"node": "CARDIO:LDL_C", "delta": -2.0, "timeframe": "P90D"},
        "SGLT2I": {"node": "METABOLIC_ENDO:HbA1c", "delta": -0.7, "timeframe": "P90D"},
        "METFORMIN": {"node": "METABOLIC_ENDO:HbA1c", "delta": -1.0, "timeframe": "P90D"},
        "GLP1RA": {"node": "METABOLIC_ENDO:Weight", "delta": -5.0, "timeframe": "P90D"},
    }

    from_step7 = defaultdict(list)
    seen_7 = set()
    for path in regimen.get("treatment_paths", []) or []:
        drug = path.get("drug") or path.get("drug_name") or path.get("name")
        node = path.get("on_target_node") or path.get("target_node")
        delta = path.get("on_target_delta") or path.get("delta")
        if not drug or node is None or delta is None:
            continue
        key = (drug, node)
        if key in seen_7:
            continue
        seen_7.add(key)
        dv = to_float(delta)
        if dv is None:
            continue
        from_step7[drug].append(
            {
                "node_id": node,
                "delta": dv,
                "timeframe": path.get("timeframe"),
                "class": path.get("drug_class"),
                "side_effects": [],
                "ttb_days_hint": path.get("ttb_days"),
            }
        )

    from_step6 = defaultdict(list)
    for t in step6.get("targets", []) or []:
        tnode = t.get("node_id")
        for dr in t.get("recommended_drugs", []) or []:
            name = dr.get("drugName") or dr.get("name")
            if not name:
                continue
            calc = (dr.get("expectedIndicatorChange") or {}).get("calculation") or {}
            delta = calc.get("expectedMeanChange")
            tf = (dr.get("expectedIndicatorChange") or {}).get("timeframe")
            direct = dr.get("directNodeId") or dr.get("direct_node_id") or tnode
            cls = dr.get("drugClass") or dr.get("class")
            dv = to_float(delta)
            sfx = []
            for sx in dr.get("sideEffects", []) or []:
                sfx.append(
                    {
                        "node": sx.get("affectedNodeId") or sx.get("node_id"),
                        "trend": (sx.get("expectedTrend") or sx.get("trend") or "").lower(),
                        "mean": sx.get("expectedMeanChange") or sx.get("mean"),
                        "unit": sx.get("unit"),
                    }
                )
            if direct and dv is not None:
                from_step6[name].append(
                    {
                        "node_id": direct,
                        "delta": dv,
                        "timeframe": tf,
                        "class": cls,
                        "side_effects": sfx,
                        "ttb_days_hint": None,
                    }
                )

    kb_actions = read_drug_kb_to_actions(drug_kb)

    combo = [d.get("name") or d.get("drug") for d in regimen.get("drugs", []) or []]
    actions = defaultdict(list)
    for nm in combo:
        has = False
        if from_step7.get(nm):
            actions[nm].extend(from_step7[nm])
            has = True
        if not has and from_step6.get(nm):
            for a in from_step6[nm]:
                if a["delta"] is not None:
                    actions[nm].append(a)
            has = bool(actions[nm])
        if not has and kb_actions.get(nm):
            actions[nm].extend(kb_actions[nm])
            has = True
        if not has:
            clazz = regimen_class_of(nm, regimen)
            if clazz:
                cfg = class_defaults.get(clazz.upper())
                if cfg:
                    actions[nm].append(
                        {
                            "node_id": cfg["node"],
                            "delta": cfg["delta"],
                            "timeframe": cfg["timeframe"],
                            "class": clazz,
                            "side_effects": [],
                            "ttb_days_hint": None,
                        }
                    )
    return actions


DUR_RE = re.compile(
    r"P(?:(?P<y>\d+)Y)?(?:(?P<m>\d+)M)?(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?"
    r"(?:T(?:(?P<h>\d+)H)?(?:(?P<mi>\d+)M)?(?:(?P<s>\d+)S)?)?$"
)


def dur_to_days(d: Optional[str]) -> float:
    if not d or not isinstance(d, str):
        return 0.0
    m = DUR_RE.match(d.strip().upper())
    if not m:
        return 0.0
    y = int(m.group("y") or 0)
    mo = int(m.group("m") or 0)
    w = int(m.group("w") or 0)
    dd = int(m.group("d") or 0)
    h = int(m.group("h") or 0)
    mi = int(m.group("mi") or 0)
    s = int(m.group("s") or 0)
    return y * 365 + mo * 30 + w * 7 + dd + h / 24.0 + mi / 1440.0 + s / 86400.0


SRC_NODE_SCALER = {
    "METABOLIC_ENDO:Steps": 1.0 / 1000.0,
    "CARDIO:BNP": 1.0 / 10.0,
}

DEFAULT_EDGE_CAP = 0.03


def within_target(cur: Optional[float], tr: Optional[List[Optional[float]]]) -> bool:
    if cur is None or not tr:
        return False
    lo, hi = tr
    if lo is not None and cur < lo:
        return False
    if hi is not None and cur > hi:
        return False
    return True


def towards_score(cur: Optional[float], delta: float, tr: Optional[List[Optional[float]]]) -> float:
    if cur is None:
        gap = abs(delta)
        return min(0.6, 1.0 - math.exp(-gap / (1.0 + gap)))
    if tr is None or all(v is None for v in tr):
        gap = abs(delta)
        return min(0.6, 1.0 - math.exp(-gap / (1.0 + gap)))
    lo, hi = tr
    goal = None
    if hi is not None and cur > hi:
        goal = hi
    elif lo is not None and cur < lo:
        goal = lo
    if goal is None:
        if within_target(cur, tr):
            return 0.2
        return 0.0
    gap = abs(cur - goal)
    new_gap = abs((cur + delta) - goal)
    if gap <= 1e-6:
        return 0.2
    return max(0.0, min(1.0, (gap - new_gap) / (gap + 1e-6)))


def exp_ramp(total_delta: float, days_since_start: int, ttb_days: float) -> float:
    t = max(0, days_since_start)
    base = ttb_days if ttb_days and ttb_days > 0 else 30.0
    tau = max(1.0, 0.7 * max(5.0, base))
    cum_t = total_delta * (1.0 - math.exp(-t / tau))
    cum_t_1 = total_delta * (1.0 - math.exp(-(t - 1) / tau)) if t > 0 else 0.0
    return cum_t - cum_t_1


def propagate_numeric(graph: Graph, src_delta: Dict[str, float], edge_cap: Optional[float]) -> Tuple[Dict[str, float], Dict[str, dict], List[str]]:
    out = defaultdict(float)
    risk_log = defaultdict(float)
    used = []
    for src, ds in src_delta.items():
        scale = SRC_NODE_SCALER.get(src, 1.0)
        ds_scaled = ds * scale
        for e in graph.succ.get(src, []):
            est = e.estimate
            if est is None:
                continue
            beta = float(est)
            if e.scale == "BETA":
                out[e.dst] += beta * ds_scaled
                used.append(e.id or f"{e.src}->{e.dst}")
            elif e.scale in ("LOG(HR)", "LOG(OR)"):
                b = beta * ds_scaled
                if edge_cap is not None:
                    if b > edge_cap:
                        b = edge_cap
                    elif b < -edge_cap:
                        b = -edge_cap
                risk_log[e.dst] += b
                used.append(e.id or f"{e.src}->{e.dst}")
    risk = {}
    for k, v in risk_log.items():
        risk[k] = {"hr_multiplier": round(math.exp(v), 4)}
    used_dedup = list(dict.fromkeys(used))
    return dict(out), risk, used_dedup


def generate_schedules(regimen: dict, horizon_days: int) -> List[dict]:
    drugs = [d.get("name") or d.get("drug") for d in regimen.get("drugs", []) or []]
    if not drugs:
        return []
    schedules = []
    schedules.append({"starts": {d: 0 for d in drugs}, "label": "parallel", "horizon": horizon_days})
    for perm in itertools.permutations(drugs):
        starts = {perm[0]: 0}
        if len(perm) >= 2:
            starts[perm[1]] = 7
        if len(perm) >= 3:
            starts[perm[2]] = 14
        schedules.append({"starts": starts, "label": "stagger7@" + ",".join(perm), "horizon": horizon_days})
    for perm in itertools.permutations(drugs):
        starts = {perm[0]: 0}
        if len(perm) >= 2:
            starts[perm[1]] = 14
        if len(perm) >= 3:
            starts[perm[2]] = 28
        schedules.append({"starts": starts, "label": "stagger14@" + ",".join(perm), "horizon": horizon_days})
    uniq = {}
    for s in schedules:
        key = tuple(sorted(s["starts"].items()))
        if key not in uniq:
            uniq[key] = s
    return list(uniq.values())


def simulate_schedule(
    regimen: dict,
    schedule: dict,
    on_targets: Dict[str, List[dict]],
    tmeta: Dict[str, dict],
    graph: Graph,
    horizon_days: int,
    edge_cap: float,
) -> dict:
    cur = {}
    for nid, m in tmeta.items():
        cur[nid] = m.get("current")
    traj = []
    risk_traj = []
    activated = set()
    drug_actions = {}
    for d, acts in on_targets.items():
        items = []
        for a in acts:
            node = a.get("node_id")
            delta = a.get("delta")
            tf = a.get("timeframe")
            ttb_hint = a.get("ttb_days_hint")
            if node is None or delta is None:
                continue
            dv = float(delta)
            if isinstance(ttb_hint, (int, float)) and ttb_hint > 0:
                ttb_days = float(ttb_hint)
            else:
                days = dur_to_days(tf) if tf else 0.0
                if days <= 0:
                    days = 30.0
                ttb_days = days
            items.append(
                {
                    "node": node,
                    "delta": dv,
                    "ttb_days": ttb_days,
                    "side_effects": a.get("side_effects") or [],
                }
            )
        if items:
            drug_actions[d] = items
    reached_day = {nid: None for nid in tmeta.keys()}
    for day in range(horizon_days + 1):
        daily_src_delta = defaultdict(float)
        for d, acts in drug_actions.items():
            start = schedule["starts"].get(d, 10**9)
            if day < start:
                continue
            offset = day - start
            for a in acts:
                inc = exp_ramp(a["delta"], offset, a["ttb_days"])
                if inc != 0.0:
                    daily_src_delta[a["node"]] += inc
        diff, risk, used = propagate_numeric(graph, daily_src_delta, edge_cap=edge_cap)
        for eid in used:
            activated.add(eid)
        for nid, dv in diff.items():
            if nid in cur and cur[nid] is not None:
                cur[nid] = cur[nid] + dv
        day_state = {}
        for nid in tmeta.keys():
            v = cur.get(nid)
            if v is not None:
                day_state[nid] = round(float(v), 4)
        traj.append({"day": day, "nodes": day_state})
        risk_traj.append({"day": day, "risk": risk})
        for nid, meta in tmeta.items():
            if reached_day[nid] is None:
                if within_target(cur.get(nid), meta.get("target_range")):
                    reached_day[nid] = day
    times = []
    wsum = 0.0
    for nid, meta in tmeta.items():
        w = max(0.4, float(meta.get("priority", 0.5)))
        wsum += w
        t = reached_day[nid] if reached_day[nid] is not None else (horizon_days + 30)
        base = 1.0 / (1.0 + t / 30.0)
        times.append((w, base))
    time_score = 0.0
    if wsum > 0:
        acc = 0.0
        for w, base in times:
            acc += w * base
        time_score = acc / wsum
    risk_pen = 0.0
    for rec in risk_traj:
        for _, v in (rec.get("risk") or {}).items():
            hr = float(v.get("hr_multiplier", 1.0) or 1.0)
            if hr > 1.0:
                risk_pen += min(1.0, hr - 1.0) * 0.01
    risk_pen = round(risk_pen, 4)
    change_pen = 0.0
    overall = round(0.65 * time_score - 0.25 * risk_pen - 0.10 * change_pen, 4)
    return {
        "label": schedule["label"],
        "overall_score": overall,
        "time_score": round(time_score, 4),
        "risk_penalty": risk_pen,
        "reached_day": reached_day,
        "traj": traj,
        "risk_traj": risk_traj,
        "activated_edges": sorted(list(activated)),
        "starts": dict(schedule["starts"]),
        "horizon": horizon_days,
    }


def main():
    args = {
        "node_file": "hpp_data/node.json",
        "edge_file": "hpp_data/edge.json",
        "targets": "example/case1/edge_select.json",
        "drug_kb": "hpp_data/drug.json",
        "step7": "example/case1/candidates.json",
        "topk_regimens": 5,
        "horizon_days": 90,
        "edge_cap": DEFAULT_EDGE_CAP,
        "out": "",
        "debug_out": "",
    }

    cache_dir = os.path.dirname(args["targets"]) or "."
    if not args["out"]:
        args["out"] = os.path.join(cache_dir, "plan.json")
    if not args["debug_out"]:
        args["debug_out"] = os.path.join(cache_dir, "rx.json")

    for p in [args["node_file"], args["edge_file"], args["targets"], args["drug_kb"], args["step7"]]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    step6 = load_json(args["targets"])
    step7 = load_json(args["step7"])
    kb = load_json(args["drug_kb"])
    g = load_graph(args["node_file"], args["edge_file"])
    tmeta = parse_targets(step6)
    regimens = parse_regimens(step7, topk=args["topk_regimens"])

    all_candidates = []
    best = None
    best_key = None

    for reg in regimens:
        actions = build_on_target_from_step7_and_kb(reg, step6, kb)
        schedules = generate_schedules(reg, horizon_days=args["horizon_days"])
        reg_sims = []
        for sch in schedules:
            sim = simulate_schedule(
                reg,
                sch,
                actions,
                tmeta,
                g,
                horizon_days=args["horizon_days"],
                edge_cap=args["edge_cap"],
            )
            reg_sims.append({"regimen": reg.get("drugs", []), "schedule": sch, "sim": sim})
            all_candidates.append({"regimen": reg.get("drugs", []), "schedule": sch, "sim": sim})
        if not reg_sims:
            continue
        reg_sims.sort(
            key=lambda x: (
                -x["sim"]["overall_score"],
                -x["sim"]["time_score"],
                x["sim"]["risk_penalty"],
            )
        )
        top = reg_sims[0]
        key = (
            -top["sim"]["overall_score"],
            -top["sim"]["time_score"],
            top["sim"]["risk_penalty"],
        )
        if best is None or best_key is None or key < best_key:
            best = {
                "regimen": top["regimen"],
                "schedule": top["sim"]["label"],
                "starts": top["sim"]["starts"],
                "scores": {
                    "overall": top["sim"]["overall_score"],
                    "time": top["sim"]["time_score"],
                    "risk_penalty": top["sim"]["risk_penalty"],
                },
                "reached_day": top["sim"]["reached_day"],
                "activated_edges": top["sim"]["activated_edges"],
                "traj": top["sim"]["traj"],
                "risk_traj": top["sim"]["risk_traj"],
                "horizon": top["sim"]["horizon"],
            }
            best_key = key

    out = {
        "best_plan": best,
        "parameters": {
            "horizon_days": args["horizon_days"],
            "edge_cap": args["edge_cap"],
            "src_scaler": SRC_NODE_SCALER,
        },
    }
    save_json(out, args["out"])

    all_candidates.sort(
        key=lambda c: (
            -c["sim"]["overall_score"],
            -c["sim"]["time_score"],
            c["sim"]["risk_penalty"],
        )
    )
    dbg = []
    limit = 20 if len(all_candidates) > 20 else len(all_candidates)
    for i in range(limit):
        c = all_candidates[i]
        sim = c["sim"]
        dbg.append(
            {
                "regimen": c["regimen"],
                "schedule": sim["label"],
                "starts": sim["starts"],
                "overall": sim["overall_score"],
                "time_score": sim["time_score"],
                "risk_penalty": sim["risk_penalty"],
                "reached_day": sim["reached_day"],
                "traj_sample": sim["traj"][:5],
            }
        )
    save_json({"candidates": dbg}, args["debug_out"])


if __name__ == "__main__":
    main()