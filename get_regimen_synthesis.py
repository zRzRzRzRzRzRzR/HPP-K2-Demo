import collections
import itertools
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from utils import load_json, save_json


class Edge:
    def __init__(self, e: dict):
        self.id = e.get("edge_id")
        self.src = e.get("from")
        self.dst = e.get("to")
        self.eff = e.get("effect") or {}
        self.scale = self.eff.get("scale")
        self.est = self.eff.get("estimate")
        self.ts = e.get("time_semantics") or {}
        self.tag = e.get("ident") or e.get("source_tag")
        self.raw = e

    @property
    def numeric(self) -> bool:
        if self.est is None:
            return False
        if isinstance(self.est, (int, float)):
            return True
        if isinstance(self.est, str):
            s = self.est.strip()
            if not s:
                return False
            for c in s:
                if c not in "0123456789+-.eE":
                    return False
            return True
        return False


class Graph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Edge] = []
        self.succ_all: Dict[str, List[Edge]] = collections.defaultdict(list)
        self.succ_beta: Dict[str, List[Edge]] = collections.defaultdict(list)
        self.succ_risk: Dict[str, List[Edge]] = collections.defaultdict(list)

    def add_node(self, n: dict):
        nid = n.get("node_id")
        if nid and nid not in self.nodes:
            self.nodes[nid] = n

    def add_edge(self, e: dict, allow_plausible_seed: bool = False):
        ed = Edge(e)
        if not ed.src or not ed.dst:
            return
        if not ed.numeric:
            if allow_plausible_seed:
                if ed.scale == "BETA":
                    ed.est = 0.01
                elif ed.scale in ("LOG(HR)", "LOG(OR)"):
                    ed.est = 0.005
                else:
                    return
            else:
                return
        self.edges.append(ed)
        self.succ_all[ed.src].append(ed)
        if ed.scale == "BETA":
            self.succ_beta[ed.src].append(ed)
        elif ed.scale in ("LOG(HR)", "LOG(OR)"):
            self.succ_risk[ed.src].append(ed)


def load_graph(
    node_file: str, edge_file: str, allow_plausible_seed: bool = False
) -> Graph:
    g = Graph()
    node_data = load_json(node_file)
    if isinstance(node_data, dict):
        nodes_iter = (
            node_data.get("node")
            or node_data.get("nodes")
            or node_data.get("data")
            or []
        )
    else:
        nodes_iter = node_data
    for n in nodes_iter:
        g.add_node(n)
    edge_data = load_json(edge_file)
    if isinstance(edge_data, dict):
        edges_iter = (
            edge_data.get("edge")
            or edge_data.get("edges")
            or edge_data.get("data")
            or []
        )
    else:
        edges_iter = edge_data
    for e in edges_iter:
        g.add_edge(e, allow_plausible_seed=allow_plausible_seed)
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
        elif isinstance(tr_raw, (list, tuple)) and tr_raw:
            lo = tr_raw[0]
            hi = tr_raw[1] if len(tr_raw) > 1 else None
            tr = [lo, hi]
        tmeta[nid] = {
            "label": t.get("label") or nid,
            "unit": t.get("unit"),
            "current": t.get("current"),
            "target_range": tr,
            "priority": float(t.get("priority_score", 0.5)),
        }
    return tmeta


def read_drug_kb(kb: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    def ensure_drug(
        name: str, clazz: Optional[str], adherence: float = 0.5, cost: float = 0.5
    ) -> dict:
        if name not in out:
            out[name] = {
                "name": name,
                "class": clazz,
                "adherence": float(adherence),
                "cost": float(cost),
                "actions": [],
            }
        else:
            if clazz and not out[name].get("class"):
                out[name]["class"] = clazz
        return out[name]

    def to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return None

    def normalize_rec(rec: dict) -> Optional[dict]:
        if not rec:
            return None
        name = rec.get("name") or rec.get("drugName")
        if not name:
            return None
        clazz = rec.get("class") or rec.get("drugClass")
        eic = rec.get("expectedIndicatorChange") or {}
        tf = eic.get("timeframe") or eic.get("window") or rec.get("timeframe")
        calc = eic.get("calculation") or {}
        delta = (
            rec.get("delta")
            or rec.get("on_target_mean_delta")
            or calc.get("expectedMeanChange")
            or rec.get("expectedMeanChange")
            or rec.get("mean")
        )
        delta_val = to_float(delta)
        if delta_val is None:
            return None
        sfx: List[dict] = []
        for sx in rec.get("sideEffects", []) or []:
            sfx.append(
                {
                    "node": sx.get("affectedNodeId") or sx.get("node_id"),
                    "trend": (sx.get("expectedTrend") or sx.get("trend") or "").lower(),
                    "mean": sx.get("expectedMeanChange") or sx.get("mean"),
                    "unit": sx.get("unit"),
                }
            )
        return {
            "name": name,
            "class": clazz,
            "delta": delta_val,
            "timeframe": tf,
            "side_effects": sfx,
        }

    def add_action(
        drug_name: str,
        clazz: Optional[str],
        node_id: Optional[str],
        delta: float,
        timeframe: Optional[str],
        sfx: List[dict],
    ):
        if not drug_name or not node_id:
            return
        drv = ensure_drug(drug_name, clazz)
        drv["actions"].append(
            {
                "node_id": node_id,
                "delta": float(delta),
                "timeframe": timeframe,
                "side_effects": sfx,
            }
        )

    indicators = kb.get("indicators")
    if isinstance(indicators, list):
        for ind in indicators:
            node_meta = ind.get("indicatorNode") or {}
            node_id = node_meta.get("nodeId") or node_meta.get("node_id")
            if not node_id:
                continue
        indicators = kb.get("indicators")
        if isinstance(indicators, list):
            for ind in indicators:
                node_meta = ind.get("indicatorNode") or {}
                node_id = node_meta.get("nodeId") or node_meta.get("node_id")
                if not node_id:
                    continue
                diseases = (node_meta.get("subIndicators") or {}).get(
                    "potentialDiseases"
                ) or []
                for dz in diseases:
                    for rec in dz.get("recommendedDrugs", []) or []:
                        norm = normalize_rec(rec)
                        if norm:
                            add_action(
                                norm["name"],
                                norm["class"],
                                node_id,
                                norm["delta"],
                                norm.get("timeframe"),
                                norm.get("side_effects") or [],
                            )

    entries: List[dict] = []
    if isinstance(kb.get("drugs"), list):
        entries = kb["drugs"]
    elif isinstance(kb.get("entries"), list):
        entries = kb["entries"]
    else:
        for k, v in kb.items():
            if k == "indicators":
                continue
            if isinstance(v, dict):
                vv = v.copy()
                vv["_name"] = k
                entries.append(vv)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        tmp = it.copy()
                        tmp["_name"] = k
                        entries.append(tmp)

    for it in entries:
        nm = it.get("name") or it.get("_name") or it.get("drugName")
        if not nm:
            continue
        clazz = it.get("class") or it.get("drugClass")
        adh = to_float(it.get("adherence"))
        cost = to_float(it.get("cost"))
        drv = ensure_drug(
            nm,
            clazz,
            adh if adh is not None else 0.5,
            cost if cost is not None else 0.5,
        )
        payloads: List[dict] = []
        for key in ("targets", "effects", "indicators", "actions"):
            if isinstance(it.get(key), list):
                payloads.extend(it[key])
        if not payloads:
            payloads = [it]
        sfx_list = it.get("sideEffects", []) or []
        side_effects = [
            {
                "node": sx.get("affectedNodeId") or sx.get("node_id"),
                "trend": (sx.get("expectedTrend") or sx.get("trend") or "").lower(),
                "mean": sx.get("expectedMeanChange") or sx.get("mean"),
                "unit": sx.get("unit"),
            }
            for sx in sfx_list
        ]
        for p in payloads:
            node = p.get("node_id") or p.get("indicator") or p.get("target")
            delta = p.get("delta") or p.get("expectedMeanChange") or p.get("mean")
            delta_val = to_float(delta)
            if not node or delta_val is None:
                continue
            drv["actions"].append(
                {
                    "node_id": node,
                    "delta": delta_val,
                    "timeframe": p.get("timeframe") or p.get("window"),
                    "side_effects": side_effects,
                }
            )

    return {k: v for k, v in out.items() if v.get("actions")}


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


def towards_score(
    cur: Optional[float], delta: float, tr: Optional[List[Optional[float]]]
) -> float:
    if tr is None or all(v is None for v in tr):
        gap = abs(delta)
        return min(0.6, 1.0 - math.exp(-gap / (1.0 + gap)))
    if cur is None:
        gap = abs(delta)
        return min(0.6, 1.0 - math.exp(-gap / (1.0 + gap)))
    lo, hi = tr
    if within_target(cur, tr):
        center = (
            (lo if lo is not None else cur) + (hi if hi is not None else cur)
        ) / 2.0
        if (cur < center and delta < 0) or (cur > center and delta > 0):
            return 0.1
        return 0.2
    goal = None
    if hi is not None and cur > hi:
        goal = hi
    elif lo is not None and cur < lo:
        goal = lo
    if goal is None:
        return 0.5
    gap = abs(cur - goal)
    new_gap = abs((cur + delta) - goal)
    if gap <= 1e-6:
        return 0.2
    return max(0.0, min(1.0, (gap - new_gap) / (gap + 1e-6)))


def k_paths_to_targets(
    g: Graph, src: str, targets: List[str], max_hops: int = 2, k_per_target: int = 3
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {t: [] for t in targets}
    frontier = [(src, [], 1.0)]
    visited = {(src, 0): 1.0}
    for depth in range(1, max_hops + 1):
        new_frontier = []
        for node, edges, coef in frontier:
            for e in g.succ_beta.get(node, []):
                if e.est is None:
                    continue
                be = float(e.est)
                new_coef = coef * be
                if abs(new_coef) == 0.0:
                    continue
                path_edges = edges + [e]
                if e.dst in out:
                    out[e.dst].append(
                        {
                            "edges": [
                                ed.id or f"{ed.src}->{ed.dst}" for ed in path_edges
                            ],
                            "coef": new_coef,
                            "nodes": [src] + [ed.dst for ed in path_edges],
                        }
                    )
                if depth < max_hops:
                    key = (e.dst, depth)
                    if key not in visited:
                        visited[key] = new_coef
                        new_frontier.append((e.dst, path_edges, new_coef))
        frontier = new_frontier
    for t in targets:
        paths = out[t]
        paths.sort(key=lambda p: -abs(p["coef"]))
        out[t] = paths[:k_per_target]
    return out


def calc_risk_penalty(risk_map: dict) -> float:
    pen = 0.0
    for _, v in (risk_map or {}).items():
        hr = float(v.get("hr_multiplier", 1.0) or 1.0)
        if hr > 1.0:
            pen += min(1.0, hr - 1.0) * 0.05
    return round(pen, 4)


def overall_score(s: dict) -> float:
    eff = s.get("efficacy", 0.0)
    sp = s.get("safety_penalty", 0.0)
    adh = s.get("adherence", 0.5)
    cost = s.get("cost", 0.5)
    return round(0.70 * eff - 0.20 * sp + 0.06 * adh + 0.04 * (1.0 - cost), 4)


def build_treatment_paths(
    drug_name: str, dk: dict, tmeta: Dict[str, dict], g: Graph, max_hops: int = 2
) -> List[dict]:
    out: List[dict] = []
    targets = list(tmeta.keys())
    for a in dk.get("actions", []):
        src = a["node_id"]
        delta = float(a.get("delta") or 0.0)
        scale = SRC_NODE_SCALER.get(src, 1.0)
        delta_scaled = delta * scale
        paths = k_paths_to_targets(g, src, targets, max_hops=max_hops, k_per_target=3)
        for tgt, plist in paths.items():
            for p in plist:
                out.append(
                    {
                        "drug": drug_name,
                        "drug_class": dk.get("class"),
                        "on_target_node": src,
                        "on_target_delta": delta,
                        "dst": tgt,
                        "edges": p["edges"],
                        "path_coef_linear": round(p["coef"], 6),
                        "ttb_days": None,
                        "ttb_score": None,
                    }
                )
        for e in g.succ_risk.get(src, []):
            if e.est is None:
                continue
            b = float(e.est) * delta_scaled
            if b > DEFAULT_EDGE_CAP:
                b = DEFAULT_EDGE_CAP
            if b < -DEFAULT_EDGE_CAP:
                b = -DEFAULT_EDGE_CAP
            out.append(
                {
                    "drug": drug_name,
                    "drug_class": dk.get("class"),
                    "on_target_node": src,
                    "on_target_delta": delta,
                    "dst": e.dst,
                    "edges": [e.id or f"{e.src}->{e.dst}"],
                    "path_risk_log_coef": round(b, 6),
                }
            )
    return out


def simulate_single_drug(
    drug_name: str, dk: dict, tmeta: Dict[str, dict], g: Graph, max_hops: int = 2
) -> dict:
    actions = dk.get("actions", [])
    targets = list(tmeta.keys())
    activated = set()
    node_delta = collections.defaultdict(float)
    risk_log = collections.defaultdict(float)
    for a in actions:
        src = a["node_id"]
        delta = float(a.get("delta") or 0.0)
        if delta == 0.0:
            continue
        scale = SRC_NODE_SCALER.get(src, 1.0)
        delta_scaled = delta * scale
        paths = k_paths_to_targets(g, src, targets, max_hops=max_hops, k_per_target=3)
        for tgt, plist in paths.items():
            for p in plist:
                node_delta[tgt] += p["coef"] * delta_scaled
                for eid in p["edges"]:
                    activated.add(eid)
        for e in g.succ_risk.get(src, []):
            if e.est is None:
                continue
            b = float(e.est) * delta_scaled
            if b > DEFAULT_EDGE_CAP:
                b = DEFAULT_EDGE_CAP
            if b < -DEFAULT_EDGE_CAP:
                b = -DEFAULT_EDGE_CAP
            risk_log[e.dst] += b
            activated.add(e.id or f"{e.src}->{e.dst}")
    per_target: Dict[str, dict] = {}
    for nid, meta in tmeta.items():
        d = node_delta.get(nid, 0.0)
        ts = towards_score(meta.get("current"), d, meta.get("target_range"))
        per_target[nid] = {
            "delta_mean": round(d, 6),
            "unit": meta.get("unit"),
            "towards_score": round(ts, 4),
        }
    risk: Dict[str, dict] = {}
    for k, v in risk_log.items():
        risk[k] = {"hr_multiplier": round(math.exp(v), 4)}
    wsum = 0.0
    ssum = 0.0
    for nid, meta in tmeta.items():
        w = max(0.4, float(meta.get("priority", 0.5)))
        wsum += w
        ssum += w * per_target[nid]["towards_score"]
    eff = (ssum / wsum) if wsum > 0 else 0.0
    score = {
        "efficacy": round(eff, 4),
        "safety_penalty": calc_risk_penalty(risk),
        "adherence": float(dk.get("adherence", 0.5)),
        "cost": float(dk.get("cost", 0.5)),
    }
    score["overall"] = overall_score(score)
    return {
        "drugs": [{"name": drug_name, "class": dk.get("class")}],
        "score": score,
        "predicted_effects": per_target,
        "risk_predictions": risk,
        "activated_edges": sorted(list(activated)),
        "treatment_paths": build_treatment_paths(
            drug_name, dk, tmeta, g, max_hops=max_hops
        ),
    }


def merge_two(r1: dict, r2: dict, tmeta: Optional[Dict[str, dict]] = None) -> dict:
    drugs = r1["drugs"] + r2["drugs"]
    pe: Dict[str, dict] = {}
    nids = set(
        list(r1["predicted_effects"].keys()) + list(r2["predicted_effects"].keys())
    )
    for nid in nids:
        d1 = r1["predicted_effects"].get(nid, {"delta_mean": 0.0})
        d2 = r2["predicted_effects"].get(nid, {"delta_mean": 0.0})
        dm = float(d1.get("delta_mean", 0.0) or 0.0) + float(
            d2.get("delta_mean", 0.0) or 0.0
        )
        unit = (
            d1.get("unit")
            or d2.get("unit")
            or ((tmeta or {}).get(nid) or {}).get("unit")
        )
        ts = 0.0
        if tmeta and nid in tmeta:
            ts = towards_score(
                tmeta[nid].get("current"), dm, tmeta[nid].get("target_range")
            )
        pe[nid] = {
            "delta_mean": round(dm, 6),
            "unit": unit,
            "towards_score": round(ts, 4),
        }
    risk: Dict[str, dict] = {}
    r_keys = set(
        list(r1["risk_predictions"].keys()) + list(r2["risk_predictions"].keys())
    )
    for k in r_keys:
        hr1 = float(r1["risk_predictions"].get(k, {}).get("hr_multiplier", 1.0) or 1.0)
        hr2 = float(r2["risk_predictions"].get(k, {}).get("hr_multiplier", 1.0) or 1.0)
        risk[k] = {"hr_multiplier": round(hr1 * hr2, 4)}
    paths = (r1.get("treatment_paths") or []) + (r2.get("treatment_paths") or [])
    act_edges = sorted(
        list({*r1.get("activated_edges", []), *r2.get("activated_edges", [])})
    )
    wsum = 0.0
    ssum = 0.0
    if tmeta:
        for nid, meta in tmeta.items():
            w = max(0.4, float(meta.get("priority", 0.5)))
            wsum += w
            ssum += w * pe.get(nid, {}).get("towards_score", 0.0)
    eff = (ssum / wsum) if wsum > 0 else 0.0
    s_pen = calc_risk_penalty(risk)
    c1 = max(1, len(r1.get("drugs", [])))
    c2 = max(1, len(r2.get("drugs", [])))
    total = c1 + c2
    adh = (
        (r1["score"].get("adherence", 0.5) * c1)
        + (r2["score"].get("adherence", 0.5) * c2)
    ) / total
    cost = (
        (r1["score"].get("cost", 0.5) * c1) + (r2["score"].get("cost", 0.5) * c2)
    ) / total
    sc = {
        "efficacy": round(eff, 4),
        "safety_penalty": s_pen,
        "adherence": adh,
        "cost": cost,
    }
    sc["overall"] = overall_score(sc)
    return {
        "drugs": drugs,
        "score": sc,
        "predicted_effects": pe,
        "risk_predictions": risk,
        "activated_edges": act_edges,
        "treatment_paths": paths,
    }


def merge_regimens(
    base_regs: List[dict],
    size: int,
    tmeta: Optional[Dict[str, dict]] = None,
    max_combos: Optional[int] = None,
) -> List[dict]:
    combos: List[dict] = []
    if size < 2 or len(base_regs) < size:
        return combos
    count = 0
    for idxs in itertools.combinations(range(len(base_regs)), size):
        if max_combos is not None and count >= max_combos:
            break
        merged = base_regs[idxs[0]]
        for idx in idxs[1:]:
            merged = merge_two(merged, base_regs[idx], tmeta)
        combos.append(merged)
        count += 1
    combos.sort(
        key=lambda r: (
            -r["score"]["overall"],
            -r["score"]["efficacy"],
            r["score"]["safety_penalty"],
        )
    )
    return combos


def main():
    args = {
        "node_file": "hpp_data/node.json",
        "edge_file": "hpp_data/edge.json",
        "targets": "example/case1/edge_select.json",
        "drug_kb": "hpp_data/drug.json",
        "max_hops": 2,
        "allow_plausible_seed": False,
        "top_drugs": 12,
        "max_combo_size": 3,
        "combo_pool_limit": 15,
        "max_combos_per_size": 500,
        "focus_nodes": "",
        "focus_top_each": 3,
        "make_pairs": True,
        "make_triples": False,
        "out": "",
        "pre_out": "",
    }

    cache_dir = os.path.dirname(args["targets"]) or "."
    if not args["out"]:
        args["out"] = os.path.join(cache_dir, "candidates.json")
    if not args["pre_out"]:
        args["pre_out"] = os.path.join(cache_dir, "pre_rx.json")

    for p in [args["node_file"], args["edge_file"], args["targets"], args["drug_kb"]]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    g = load_graph(
        args["node_file"],
        args["edge_file"],
        allow_plausible_seed=args["allow_plausible_seed"],
    )
    tmeta = parse_targets(load_json(args["targets"]))
    kb = read_drug_kb(load_json(args["drug_kb"]))

    singles: List[dict] = []
    pre_paths: List[dict] = []
    for name, dk in kb.items():
        reg = simulate_single_drug(name, dk, tmeta, g, max_hops=args["max_hops"])
        singles.append(reg)
        pre_paths.extend(reg.get("treatment_paths") or [])

    singles_sorted = sorted(
        singles, key=lambda r: (-r["score"]["overall"], -r["score"]["efficacy"])
    )

    combo_pool_limit = max(2, args["combo_pool_limit"])
    combo_pool: List[dict] = []

    focus_nodes_str = (args["focus_nodes"] or "").strip()
    if focus_nodes_str:
        focus_nodes = [x.strip() for x in focus_nodes_str.split(",") if x.strip()]
        top_each = max(1, int(args["focus_top_each"]))
        ranked: List[Tuple[float, dict]] = []
        for node in focus_nodes:
            scored: List[Tuple[float, dict]] = []
            for reg in singles_sorted:
                pe = (reg.get("predicted_effects") or {}).get(node)
                if not pe:
                    continue
                delta = pe.get("delta_mean")
                if delta is None:
                    continue
                score = abs(float(delta) or 0.0)
                if score <= 0.0:
                    continue
                scored.append((score, reg))
            scored.sort(key=lambda x: -x[0])
            ranked.extend(scored[:top_each])
        ranked.sort(key=lambda x: -x[0])
        for _, reg in ranked:
            if reg not in combo_pool:
                combo_pool.append(reg)
            if len(combo_pool) >= combo_pool_limit:
                break

    if len(combo_pool) < combo_pool_limit:
        for reg in singles_sorted:
            if reg not in combo_pool:
                combo_pool.append(reg)
            if len(combo_pool) >= combo_pool_limit:
                break

    combos: List[dict] = []
    combo_sizes = set()
    max_combo = args["max_combo_size"] or 0
    if max_combo >= 2:
        for s in range(2, max_combo + 1):
            combo_sizes.add(s)
    if args["make_pairs"]:
        combo_sizes.add(2)
    if args["make_triples"]:
        combo_sizes.add(3)
    combo_sizes = sorted([s for s in combo_sizes if s >= 2])

    combo_cap = (
        args["max_combos_per_size"]
        if args["max_combos_per_size"] and args["max_combos_per_size"] > 0
        else None
    )
    for size in combo_sizes:
        combos += merge_regimens(
            combo_pool, size=size, tmeta=tmeta, max_combos=combo_cap
        )

    regimens = singles + combos
    regimens.sort(
        key=lambda r: (
            -r["score"]["overall"],
            -r["score"]["efficacy"],
            r["score"]["safety_penalty"],
        )
    )

    out = {
        "parameters": {
            "max_hops": args["max_hops"],
            "allow_plausible_seed": args["allow_plausible_seed"],
            "edge_cap_risk": DEFAULT_EDGE_CAP,
            "src_scaler": SRC_NODE_SCALER,
            "max_combo_size": args["max_combo_size"],
            "combo_sizes_used": combo_sizes,
            "combo_pool_limit": combo_pool_limit,
            "combo_pool_size": len(combo_pool),
            "max_combos_per_size": args["max_combos_per_size"],
        },
        "targets": [
            {
                "node_id": nid,
                "label": meta["label"],
                "unit": meta.get("unit"),
                "current": meta.get("current"),
                "target_range": meta.get("target_range"),
                "priority": meta.get("priority"),
            }
            for nid, meta in tmeta.items()
        ],
        "regimens": regimens[:120],
    }

    save_json(out, args["out"])
    save_json({"treatment_paths": pre_paths}, args["pre_out"])


if __name__ == "__main__":
    main()
