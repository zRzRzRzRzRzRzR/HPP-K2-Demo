import json
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from utils import build_json


class Edge:
    __slots__ = ("edge_id", "src", "dst", "raw")

    def __init__(self, edge_id: str, src: str, dst: str, raw: dict):
        self.edge_id = edge_id
        self.src = src
        self.dst = dst
        self.raw = raw


class Graph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Edge] = []
        self.succ: Dict[str, List[Edge]] = defaultdict(list)
        self.pred: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, node: dict):
        nid = node.get("node_id")
        if nid and nid not in self.nodes:
            self.nodes[nid] = node

    def add_edge(self, e: dict):
        src = e.get("from")
        dst = e.get("to")
        if not src or not dst:
            return
        eid = e.get("edge_id") or f"{src}->{dst}#{len(self.edges)}"
        edge = Edge(eid, src, dst, e)
        self.edges.append(edge)
        self.succ[src].append(edge)
        self.pred[dst].append(edge)

    def has_node(self, nid: str) -> bool:
        return nid in self.nodes

    def get_label(self, nid: str) -> str:
        n = self.nodes.get(nid) or {}
        return n.get("label") or nid

    def reverse_bfs_to_druggable(
        self,
        abnormal_node: str,
        druggable_nodes: Set[str],
        max_hops: int = 3,
        prefer_strong_edges: bool = True,
    ) -> List[Tuple[str, List[Edge]]]:
        if abnormal_node not in self.nodes:
            return []

        def is_strong(edge_raw: dict) -> bool:
            if not prefer_strong_edges:
                return True
            eff = (edge_raw or {}).get("effect") or {}
            ident = (edge_raw or {}).get("ident") or ""
            return (eff.get("estimate") is not None) or (ident != "plausible_seed")

        visited: Set[str] = {abnormal_node}
        q: Deque[Tuple[str, List[Edge], int]] = deque()
        q.append((abnormal_node, [], 0))
        results: List[Tuple[str, List[Edge]]] = []

        while q:
            node, path, hop = q.popleft()
            if hop > max_hops:
                continue

            if node in druggable_nodes and node != abnormal_node:
                results.append((node, path))

            if hop == max_hops:
                continue

            for e in self.pred.get(node, []):
                if not is_strong(e.raw):
                    continue
                prev_node = e.src
                if prev_node in visited:
                    continue
                visited.add(prev_node)
                new_path = [e] + path
                q.append((prev_node, new_path, hop + 1))
        return results


def load_graph(node_file: str, edge_file: str) -> Graph:
    g = Graph()

    node_data = build_json(node_file)
    if isinstance(node_data, dict):
        nodes_iter = node_data.get("nodes") or node_data.get("data") or []
    else:
        nodes_iter = node_data
    for n in nodes_iter:
        g.add_node(n)

    edge_data = build_json(edge_file)
    if isinstance(edge_data, dict):
        edges_iter = edge_data.get("edges") or edge_data.get("data") or []
    else:
        edges_iter = edge_data
    for e in edges_iter:
        g.add_edge(e)

    return g


def load_drug_kb(drug_kb_file: str) -> Dict[str, List[dict]]:
    kb = build_json(drug_kb_file)

    node_to_drugs: Dict[str, List[dict]] = defaultdict(list)
    indicators = kb.get("indicators") or []
    for ind in indicators:
        ind_node = ind.get("indicatorNode") or {}
        node_id = ind_node.get("nodeId") or ind_node.get("node_id")
        if not node_id:
            continue
        diseases = (ind_node.get("subIndicators") or {}).get("potentialDiseases") or []
        for dz in diseases:
            for rec in dz.get("recommendedDrugs", []) or []:
                entry = normalize_drug_entry(rec)
                if entry:
                    node_to_drugs[node_id].append(entry)
    return node_to_drugs


def normalize_drug_entry(rec: dict) -> Optional[dict]:
    name = rec.get("drugName") or rec.get("name")
    dcls = rec.get("drugClass") or rec.get("class") or ""
    eic = rec.get("expectedIndicatorChange") or {}
    tf = eic.get("timeframe") or eic.get("window") or ""
    calc = eic.get("calculation") or {}
    emc = calc.get("expectedMeanChange")
    tr = calc.get("targetValueRange") or {}
    lb, ub = tr.get("lowerBound"), tr.get("upperBound")
    target_range = [lb, ub] if (lb is not None or ub is not None) else None
    sfx = []
    for sx in rec.get("sideEffects", []) or []:
        sfx.append(
            {
                "node": sx.get("affectedNodeId") or sx.get("node_id"),
                "trend": sx.get("expectedTrend") or sx.get("trend"),
                "mean": sx.get("expectedMeanChange") or sx.get("mean"),
                "unit": sx.get("unit"),
            }
        )
    if not name:
        return None
    return {
        "name": name,
        "class": dcls,
        "timeframe": str(tf) if tf is not None else "",
        "on_target_mean_delta": emc,
        "target_range": target_range,
        "side_effects": sfx,
        "raw": rec,
    }


def edges_to_path_dict(path_edges: List[Edge]) -> List[dict]:
    out = []
    for e in path_edges:
        eff = (e.raw or {}).get("effect") or {}
        out.append(
            {
                "edge_id": e.edge_id,
                "from": e.src,
                "to": e.dst,
                "effect_scale": eff.get("scale"),
                "effect_estimate": eff.get("estimate"),
                "raw_effect": eff,
            }
        )
    return out


def add_or_update_target(
    targets_map: Dict[str, dict],
    node_id: str,
    t_type: str,
    distance: int,
    graph_paths: List[List[dict]],
    drug_index: Dict[str, List[dict]],
    abnormal_map: Dict[str, dict],
    graph: Graph,
):
    cur_val, cur_unit = None, None
    if node_id in abnormal_map:
        cur_val = abnormal_map[node_id].get("value")
        cur_unit = abnormal_map[node_id].get("unit")
    else:
        cur_unit = (graph.nodes.get(node_id) or {}).get("unit")

    recs = drug_index.get(node_id, [])
    target_range = None
    for r in recs:
        if r.get("target_range") is not None:
            target_range = r["target_range"]
            break

    base = (
        1.0
        if (t_type == "direct" or distance == 0)
        else {1: 0.8, 2: 0.6, 3: 0.4}.get(distance, 0.2)
    )
    numeric_edges = sum(
        1
        for p in graph_paths
        for e in p
        if (e.get("raw_effect") or {}).get("estimate") is not None
    )
    score = round(base + 0.2 * numeric_edges, 3)

    if node_id not in targets_map:
        targets_map[node_id] = {
            "node_id": node_id,
            "label": graph.get_label(node_id),
            "type": t_type,
            "distance_hops": distance,
            "priority_score": score,
            "current": cur_val,
            "unit": cur_unit,
            "target_range": target_range,
            "recommended_drugs": recs,
            "graph_paths": graph_paths,
            "sources": {
                "graph_nodes": "hpp_data/node.json",
                "graph_edges": "hpp_data/edge.json",
                "drug_kb": "hpp_data/drug.json",
            },
        }
    else:
        tgt = targets_map[node_id]
        tgt["type"] = (
            "direct" if (tgt["type"] == "direct" or t_type == "direct") else "indirect"
        )
        tgt["distance_hops"] = min(tgt.get("distance_hops", 99), distance)
        tgt["priority_score"] = round(
            max(tgt.get("priority_score", 0.0), score) + 0.1, 3
        )

        existed = set(
            (e["edge_id"], e["from"], e["to"])
            for p in tgt.get("graph_paths", [])
            for e in p
        )
        for p in graph_paths:
            sig = tuple((e["edge_id"], e["from"], e["to"]) for e in p)
            if not all(x in existed for x in sig):
                tgt.setdefault("graph_paths", []).append(p)
                for x in sig:
                    existed.add(x)

        seen = {
            (d["name"], d.get("class", "")) for d in tgt.get("recommended_drugs", [])
        }
        for d in recs:
            key = (d["name"], d.get("class", ""))
            if key not in seen:
                tgt["recommended_drugs"].append(d)
                seen.add(key)


def build_targets_from_input(
    abnormal_input: dict,
    graph: Graph,
    drug_index: Dict[str, List[dict]],
    max_hops: int = 3,
    prefer_strong_edges: bool = True,
) -> dict:
    t_eval = abnormal_input.get("t_eval_weeks")
    abns = abnormal_input.get("abnormal_points", [])
    abnormal_map = {a.get("node_id"): a for a in abns if a.get("node_id")}

    druggable_nodes: Set[str] = set(drug_index.keys())
    targets_map: Dict[str, dict] = {}
    unmatched: List[str] = []

    for ab in abns:
        nid = ab.get("node_id")
        if not nid or not graph.has_node(nid):
            if nid:
                unmatched.append(nid)
            continue

        if nid in druggable_nodes:
            add_or_update_target(
                targets_map=targets_map,
                node_id=nid,
                t_type="direct",
                distance=0,
                graph_paths=[],
                drug_index=drug_index,
                abnormal_map=abnormal_map,
                graph=graph,
            )

        hits = graph.reverse_bfs_to_druggable(
            abnormal_node=nid,
            druggable_nodes=druggable_nodes,
            max_hops=max_hops,
            prefer_strong_edges=prefer_strong_edges,
        )
        for cand_node, path_edges in hits:
            add_or_update_target(
                targets_map=targets_map,
                node_id=cand_node,
                t_type="indirect",
                distance=len(path_edges),
                graph_paths=[edges_to_path_dict(path_edges)],
                drug_index=drug_index,
                abnormal_map=abnormal_map,
                graph=graph,
            )

    targets = list(targets_map.values())
    targets.sort(key=lambda x: (-x.get("priority_score", 0.0), x["node_id"]))

    return {
        "t_eval_weeks": t_eval,
        "targets": targets,
        "unmatched_abnormal_nodes": unmatched,
    }


def run_target_node_selection(
    abnormal_input_path: str = "example/case1/abnormal_node.json",
    node_path: str = "hpp_data/node.json",
    edge_path: str = "hpp_data/edge.json",
    drug_kb_path: str = "hpp_data/drug.json",
    output_path: Optional[str] = None,
    max_hops: int = 3,
    prefer_strong_edges: bool = True,
) -> dict:
    """
    Run target node selection based on abnormal nodes.

    Args:
        abnormal_input_path: Path to abnormal node input JSON file
        node_path: Path to graph node data file
        edge_path: Path to graph edge data file
        drug_kb_path: Path to drug knowledge base file
        output_path: Optional path to save output JSON (if None, won't save to file)
        max_hops: Maximum hops for reverse BFS search
        prefer_strong_edges: Whether to prefer edges with strong evidence

    Returns:
        dict: Result containing targets, unmatched nodes, and metadata

    Raises:
        FileNotFoundError: If any required input file does not exist
    """
    # Validate input files
    for path in [node_path, edge_path, drug_kb_path, abnormal_input_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load graph and drug knowledge base
    graph = load_graph(node_file=node_path, edge_file=edge_path)
    drug_index = load_drug_kb(drug_kb_file=drug_kb_path)
    abnormal_input = build_json(abnormal_input_path)

    # Build targets from input
    result = build_targets_from_input(
        abnormal_input=abnormal_input,
        graph=graph,
        drug_index=drug_index,
        max_hops=max_hops,
        prefer_strong_edges=prefer_strong_edges,
    )

    # Save to file if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        n_targets = len(result.get("targets", []))
        n_unmatched = len(result.get("unmatched_abnormal_nodes", []))
        print(f"[Info] Targets: {n_targets}, Unmatched abnormal nodes: {n_unmatched}")
        print(f"[Info] Wrote output to: {output_path}")

    return result


def main():
    """Command-line entry point for target node selection."""
    result = run_target_node_selection(
        abnormal_input_path="example/case1/abnormal_node.json",
        node_path="hpp_data/node.json",
        edge_path="hpp_data/edge.json",
        drug_kb_path="hpp_data/drug.json",
        output_path="example/case1/edge_select.json",
        max_hops=3,
        prefer_strong_edges=True,
    )
    return result


if __name__ == "__main__":
    main()
