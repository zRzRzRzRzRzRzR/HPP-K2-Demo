import json
import os
from typing import Optional

VIEW_HEIGHT = 420

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>3D Causal Network</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      margin: 0;
      padding: 0;
      background: #f9fafb;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .viewer {
      width: 100%;
      height: {VIEW_HEIGHT}px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      background: #000000;
      position: relative;
      box-sizing: border-box;
    }
    .viewer iframe {
      width: 100%;
      height: 100%;
      border: 0;
      display: block;
    }
    .fullscreen-toggle {
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 4px 10px;
      font-size: 10px;
      border-radius: 999px;
      border: none;
      background: rgba(15,23,42,0.9);
      color: #f9fafb;
      cursor: pointer;
      z-index: 20;
    }
    .viewer-fullscreen {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      border-radius: 0;
      border: none;
      margin: 0;
      z-index: 99999;
      background: #000000;
    }
  </style>
</head>
<body>
  <div class="viewer" id="viewer">
    <button class="fullscreen-toggle" id="fs-btn">Fullscreen</button>
    <iframe
      sandbox="allow-scripts allow-same-origin"
      srcdoc='<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { height: 100%; margin: 0; background: #000000; overflow: hidden; }
    #3d-graph { position: fixed; inset: 0; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3d-force-graph"></script>
</head>
<body>
  <div id="3d-graph"></div>
  <script type="application/json" id="graph-data">{GRAPH_JSON}</script>
  <script type="module">
    import SpriteText from "https://esm.sh/three-spritetext";

    const highlightNodes = new Set();
    const highlightLinks = new Set();
    let hoverNode = null;

    const Graph = new ForceGraph3D(document.getElementById("3d-graph"))
      .backgroundColor("#000000")
      .linkDirectionalArrowLength(3.5)
      .linkDirectionalArrowRelPos(1)
      .linkCurvature(0.25)
      .linkDirectionalParticles(link => highlightLinks.has(link) ? 4 : 0)
      .linkDirectionalParticleWidth(4)
      .nodeAutoColorBy("group")
      .linkWidth(link => highlightLinks.has(link) ? 4 : 1)
      .nodeThreeObject(node => {
        const sprite = new SpriteText(String(node.id));
        sprite.material.depthWrite = false;
        sprite.color = node.color || "#ffffff";
        sprite.textHeight = 8;
        sprite.center.y = -0.6;
        return sprite;
      })
      .nodeThreeObjectExtend(true);

    Graph.onNodeHover(node => {
      if ((!node && !highlightNodes.size) || node === hoverNode) return;
      highlightNodes.clear();
      highlightLinks.clear();
      if (node) {
        highlightNodes.add(node);
        (node.neighbors || []).forEach(nei => highlightNodes.add(nei));
        (node.links || []).forEach(l => highlightLinks.add(l));
      }
      hoverNode = node || null;
      updateHighlight();
    });

    Graph.onLinkHover(link => {
      highlightNodes.clear();
      highlightLinks.clear();
      if (link) {
        highlightLinks.add(link);
        highlightNodes.add(link.source);
        highlightNodes.add(link.target);
      }
      updateHighlight();
    });

    function updateHighlight() {
      Graph
        .nodeColor(node => {
          if (highlightNodes.has(node)) {
            return node === hoverNode ? "#ff0000" : "#ffa000";
          }
          return node.color || "#ffffff";
        })
        .linkWidth(Graph.linkWidth())
        .linkDirectionalParticles(Graph.linkDirectionalParticles());
    }

    Graph.onNodeClick(node => {
      const distance = 300;
      const dx = node.x || 0;
      const dy = node.y || 0;
      const dz = node.z || 0;
      const dist = Math.hypot(dx, dy, dz) || 1;
      const distRatio = 1 + distance / dist;
      const newPos = { x: dx * distRatio, y: dy * distRatio, z: dz * distRatio };
      Graph.cameraPosition(newPos, node, 3000);
    });

    const raw = document.getElementById("graph-data").textContent || "";
    let data = { nodes: [], links: [] };
    try {
      data = JSON.parse(raw);
    } catch (e) {
      console.error("Invalid graph JSON:", e);
    }

    const nodeById = new Map((data.nodes || []).map(n => [n.id, n]));
    (data.nodes || []).forEach(n => {
      n.neighbors = [];
      n.links = [];
    });
    (data.links || []).forEach(l => {
      const a = typeof l.source === "object" ? l.source : nodeById.get(l.source);
      const b = typeof l.target === "object" ? l.target : nodeById.get(l.target);
      if (!a || !b) return;
      a.neighbors.push(b);
      b.neighbors.push(a);
      a.links.push(l);
      b.links.push(l);
    });

    Graph.graphData(data);
    Graph.d3Force("charge").strength(-400);
  </script>
</body>
</html>'>
    </iframe>
  </div>
  <script>
    const viewer = document.getElementById("viewer");
    const btn = document.getElementById("fs-btn");
    if (viewer && btn) {
      let isFullscreen = false;
      btn.addEventListener("click", () => {
        isFullscreen = !isFullscreen;
        if (isFullscreen) {
          viewer.classList.add("viewer-fullscreen");
          btn.textContent = "Exit";
        } else {
          viewer.classList.remove("viewer-fullscreen");
          btn.textContent = "Fullscreen";
        }
      });
    }
  </script>
</body>
</html>
"""


def resolve_case_dir(case_dir_raw: Optional[str]) -> str:
    if not case_dir_raw:
        return ""
    return case_dir_raw.strip()


def find_diagnosis_json(case_dir: str) -> Optional[str]:
    if not case_dir:
        return None
    path_exam = os.path.join(case_dir, "diagnosis_exam.json")
    path_default = os.path.join(case_dir, "diagnosis.json")
    if os.path.exists(path_exam):
        return path_exam
    if os.path.exists(path_default):
        return path_default
    return None


def find_exam_source(case_dir: str) -> Optional[str]:
    if not case_dir:
        return None
    exam_json = os.path.join(case_dir, "diagnosis_exam.json")
    if os.path.exists(exam_json):
        return exam_json
    exam_pdf = os.path.join(case_dir, "diagnosis_exam.pdf")
    if os.path.exists(exam_pdf):
        return exam_pdf
    full_json = os.path.join(case_dir, "diagnosis.json")
    if os.path.exists(full_json):
        return full_json
    full_pdf = os.path.join(case_dir, "diagnosis.pdf")
    if os.path.exists(full_pdf):
        return full_pdf
    return None


def find_edge_select(case_dir: str) -> Optional[str]:
    if not case_dir:
        return None
    edge_json = os.path.join(case_dir, "edge_select.json")
    if os.path.exists(edge_json):
        return edge_json
    return None


def build_diagnosis_preview_html(diagnosis_path: str) -> str:
    if not diagnosis_path or not os.path.exists(diagnosis_path):
        return ""
    with open(diagnosis_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preview = json.dumps(data, ensure_ascii=False, indent=2)
    if len(preview) > 2200:
        preview = preview[:2200] + "\n... (truncated)"
    return (
        "<details style='margin-top:6px;'>"
        "<summary style='cursor:pointer; font-size:11px; color:#2563eb; font-weight:600;'>"
        "View parsed report structure (diagnosis.json)</summary>"
        "<pre style='margin-top:6px; padding:6px 8px; border-radius:10px; background:#f9fafb;"
        "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
        f"{preview}"
        "</pre>"
        "</details>"
    )
