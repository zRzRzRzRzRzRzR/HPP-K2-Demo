import json
import os

import gradio as gr

from get_abnormal_node import run_abnormal_node_selection
from get_exam_node import run_examination_node_selection
from pdf_converter import convert_pdf_to_json
from utils_gradio import (
    HTML_TEMPLATE,
    VIEW_HEIGHT,
    build_diagnosis_preview_html,
    find_exam_source,
    find_json,
    resolve_case_dir,
)


def get_graph_html(step_state: int) -> str:
    json_path = "hpp_data/causal_graph.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data["_current_step"] = step_state
    graph_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("{GRAPH_JSON}", graph_json).replace(
        "{VIEW_HEIGHT}", str(VIEW_HEIGHT)
    )
    return html


def build_step2_block(abnormal_output, previous_html: str) -> str:
    abnormal_data = abnormal_output
    points = (
        abnormal_data.get("abnormal_points")
        or abnormal_data.get("abnormal_nodes")
        or abnormal_data.get("abnormals")
        or abnormal_data.get("items")
    )

    if not isinstance(points, list):
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 2 · Highlighted abnormal indicators (raw)</summary>"
            "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
            "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
            "<pre style='margin:0; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{abnormal_output}"
            "</pre>"
            "</div>"
            "</details>"
        )

    rows_html = ""
    for item in points:
        if not isinstance(item, dict):
            continue

        node_id = str(item.get("node_id", item.get("id", ""))).strip()
        val = (
            item.get("abnormal_value")
            or item.get("value")
            or item.get("observed_value")
            or item.get("result")
            or ""
        )
        unit = item.get("unit") or item.get("units") or ""
        highlight = bool(item.get("highlight") or item.get("is_abnormal"))

        val_text = "-" if val in (None, "", []) else str(val)
        unit_text = "-" if unit in (None, "", []) else str(unit)

        val_style = (
            "color:#dc2626; font-weight:700;"
            if highlight and val_text != "-"
            else "color:#111827; font-weight:500;"
        )

        rows_html += (
            "<tr>"
            f"<td style='padding:4px 6px; font-size:10px; color:#111827; border-bottom:1px solid #e5e7eb;'>{node_id}</td>"
            f"<td style='padding:4px 6px; font-size:10px; {val_style} border-bottom:1px solid #e5e7eb;'>{val_text}</td>"
            f"<td style='padding:4px 6px; font-size:10px; color:#6b7280; border-bottom:1px solid #e5e7eb;'>{unit_text}</td>"
            "</tr>"
        )

    if not rows_html:
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 2 · Highlighted abnormal indicators</summary>"
            "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
            "background:#fef2f2; border:1px solid #ef4444; font-size:12px; color:#991b1b;'>"
            "<div>No abnormal indicators parsed. Check the abnormal-node script output format.</div>"
            "</div>"
            "</details>"
        )

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 2 · Highlighted abnormal indicators</summary>"
        "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
        "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<thead>"
        "<tr>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>node_id</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>abnormal value</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>unit</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def build_step3_block(edge_output, previous_html: str) -> str:
    if isinstance(edge_output, dict):
        data = edge_output
    else:
        data = {}

    targets = (
        data.get("targets") or data.get("causal_targets") or data.get("items") or []
    )

    if not isinstance(targets, list) or not targets:
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 3 · Causal targets & strategies (raw)</summary>"
            "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
            "background:#eef2ff; border:1px solid #6366f1; font-size:12px; color:#312e81;'>"
            "<pre style='margin:0; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{edge_output}"
            "</pre>"
            "</div>"
            "</details>"
        )

    rows_html = ""
    for t in targets:
        if not isinstance(t, dict):
            continue

        node_id = str(t.get("node_id", t.get("id", ""))).strip()
        label = str(t.get("label", t.get("indicator", ""))).strip()

        current = t.get("current") or t.get("current_value") or t.get("value")

        target = t.get("target") or t.get("target_value")
        target_range = t.get("target_range") or t.get("range")
        if (
            not target
            and isinstance(target_range, (list, tuple))
            and len(target_range) == 2
        ):
            target = f"{target_range[0]}–{target_range[1]}"

        drugs = (
            t.get("drugs") or t.get("recommended_drugs") or t.get("medications") or ""
        )
        if isinstance(drugs, (list, tuple)):
            drug_names = []
            for d in drugs:
                if isinstance(d, dict):
                    name = d.get("name") or d.get("drugName") or d.get("label")
                    if name:
                        drug_names.append(str(name))
                else:
                    drug_names.append(str(d))
            drugs = ", ".join(drug_names)

        current_text = "-" if current in (None, "", []) else str(current)
        target_text = "-" if target in (None, "", []) else str(target)
        drugs_text = "-" if drugs in (None, "", []) else str(drugs)

        rows_html += (
            "<tr>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{node_id}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{label}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#dc2626; font-weight:600; border-bottom:1px solid #e5e7eb;'>{current_text}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#16a34a; font-weight:600; border-bottom:1px solid #e5e7eb;'>{target_text}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#4b5563; border-bottom:1px solid #e5e7eb;'>{drugs_text}</td>"
            "</tr>"
        )

    if not rows_html:
        return previous_html

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 3 · Causal targets & strategies</summary>"
        "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
        "background:#eef2ff; border:1px solid #6366f1; font-size:12px; color:#312e81;'>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<thead>"
        "<tr>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>node_id</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>indicator</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>current</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>target</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>recommended drugs</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def build_step4_block(candidates_output, previous_html: str) -> str:
    if isinstance(candidates_output, dict):
        data = candidates_output
    else:
        data = {}

    regimens = data.get("regimens") or []
    if not isinstance(regimens, list) or not regimens:
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 4 · Candidate regimens</summary>"
            "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
            "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
            "<div>No regimen candidates found. Please run get_regimen_synthesis.py to generate candidates.json.</div>"
            "</div>"
            "</details>"
        )

    rows_html = ""
    for idx, reg in enumerate(regimens[:20], start=1):
        drugs = reg.get("drugs") or []
        if isinstance(drugs, list):
            drug_labels = []
            for d in drugs:
                if isinstance(d, dict):
                    name = (d.get("name") or d.get("drug") or "").strip()
                    clazz = (d.get("class") or d.get("drug_class") or "").strip()
                    if name and clazz:
                        drug_labels.append(f"{name} ({clazz})")
                    elif name:
                        drug_labels.append(name)
                    elif clazz:
                        drug_labels.append(clazz)
                else:
                    drug_labels.append(str(d))
            regimen_str = " + ".join([s for s in drug_labels if s])
        else:
            regimen_str = str(drugs)

        score = reg.get("score") or {}
        overall = score.get("overall")
        eff = score.get("efficacy")
        safety = score.get("safety_penalty") or score.get("risk_penalty")
        adh = score.get("adherence")
        cost = score.get("cost")

        overall_txt = "-" if overall is None else f"{float(overall):.3f}"
        eff_txt = "-" if eff is None else f"{float(eff):.3f}"
        safety_txt = "-" if safety is None else f"{float(safety):.3f}"
        adh_txt = "-" if adh is None else f"{float(adh):.2f}"
        cost_txt = "-" if cost is None else f"{float(cost):.2f}"

        rows_html += (
            "<tr>"
            f"<td style='padding:4px 6px; font-size:9px; color:#6b7280; border-bottom:1px solid #e5e7eb;'>{idx}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; font-weight:600; border-bottom:1px solid #e5e7eb;'>{regimen_str}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{overall_txt}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{eff_txt}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#b91c1c; border-bottom:1px solid #fee2e2;'>{safety_txt}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{adh_txt}</td>"
            f"<td style='padding:4px 6px; font-size:9px; color:#111827; border-bottom:1px solid #e5e7eb;'>{cost_txt}</td>"
            "</tr>"
        )

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 4 · Candidate regimens (from get_regimen_synthesis)</summary>"
        "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
        "background:#fefce8; border:1px solid #facc15; font-size:12px; color:#78350f;'>"
        "<div style='font-size:10px; color:#92400e; margin-bottom:4px;'>"
        "Higher overall score indicates better balance of efficacy, safety, adherence, and cost."
        "</div>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<thead>"
        "<tr>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>rank</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>regimen</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>overall</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>efficacy</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>risk</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>adherence</th>"
        "<th style='text-align:left; padding:4px 6px; font-size:9px; color:#6b7280; font-weight:500;'>cost</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def render_graph(step_state):
    html = get_graph_html(step_state)
    return gr.update(value=html, visible=True)


def run_flow(pdf_file, case_dir_raw, current_case_dir, previous_html, step_state):
    if step_state is None:
        step_state = 1
    if previous_html is None:
        previous_html = ""

    if isinstance(pdf_file, str):
        pdf_path = pdf_file
    elif pdf_file is not None and hasattr(pdf_file, "name"):
        pdf_path = pdf_file.name
    else:
        pdf_path = None

    pdf_update = gr.update()
    case_dir_input_update = gr.update()
    graph_btn_update = gr.update(visible=False)
    graph_html_value = None

    if step_state == 1:
        case_dir = resolve_case_dir(case_dir_raw) or current_case_dir or ""

        if pdf_path:
            diagnosis_path = convert_pdf_to_json(pdf_path)
            if not case_dir:
                case_dir = os.path.dirname(diagnosis_path)
        else:
            diagnosis_path = find_json(case_dir, "diagnosis")

        if not diagnosis_path or not os.path.exists(diagnosis_path):
            error_html = (
                "<details open class='step-card'>"
                "<summary>Step 1 · Initial Diagnosis</summary>"
                "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
                "background:#fef2f2; border:1px solid #ef4444; font-size:12px; color:#991b1b;'>"
                "<div style='font-weight:600; margin-bottom:4px;'>Step 1 · Failed</div>"
                "<div>No valid diagnosis JSON/PDF found. Please upload a valid report or specify the correct case directory.</div>"
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(1)
            return (
                pdf_update,
                gr.update(value=case_dir),
                case_dir,
                error_html,
                error_html,
                "Step 1 · Initial Diagnosis",
                gr.update(value="Run Step 1"),
                1,
                graph_btn_update,
                graph_html_value,
            )

        exam_nodes = run_examination_node_selection(
            diagnosis_path,
            "hpp_data/node.json",
        )
        if isinstance(exam_nodes, dict):
            node_str = json.dumps(exam_nodes, ensure_ascii=False, indent=2)
        else:
            node_str = str(exam_nodes)

        diagnosis_preview = build_diagnosis_preview_html(diagnosis_path)

        step1_html = (
            "<details open class='step-card'>"
            "<summary>Step 1 · Examination node suggestions</summary>"
            "<div style='margin-top:4px; padding:10px 12px; border-radius:14px;"
            "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
            "<pre style='margin:0; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{node_str}"
            "</pre>"
            "</div>"
            f"{diagnosis_preview}"
            "</details>"
        )

        case_dir_input_update = gr.update(value=case_dir)
        graph_html_value = get_graph_html(2)

        return (
            pdf_update,
            case_dir_input_update,
            case_dir,
            step1_html,
            step1_html,
            "Step 2 · Full Examination",
            gr.update(value="Run Step 2"),
            2,
            graph_btn_update,
            graph_html_value,
        )

    if step_state == 2:
        if not previous_html:
            previous_html = (
                "<details open class='step-card'>"
                "<summary>Step 2 · Full Examination</summary>"
                "<div style='margin-top:4px; padding:8px 12px; border-radius:12px; background:#f9fafb;"
                "border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "Run Step 1 first."
                "</div>"
                "</details>"
            )

        resolved_case_dir = resolve_case_dir(current_case_dir)

        if pdf_path:
            exam_json_path = convert_pdf_to_json(pdf_path)
            if not resolved_case_dir:
                resolved_case_dir = os.path.dirname(exam_json_path)
        else:
            exam_json_path = find_exam_source(resolved_case_dir)

        if not exam_json_path or not os.path.exists(exam_json_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 2 · Full Examination</summary>"
                "<div style='margin-top:4px; padding:8px 12px; border-radius:12px;"
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "No valid full examination JSON/PDF found."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(2)
            return (
                pdf_update,
                case_dir_input_update,
                resolved_case_dir,
                previous_html,
                html,
                "Step 2 · Full Examination",
                gr.update(value="Run Step 2"),
                2,
                graph_btn_update,
                graph_html_value,
            )

        abnormal_output = run_abnormal_node_selection(
            exam_json_path,
            "hpp_data/node.json",
        )

        html = build_step2_block(abnormal_output, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(3)

        return (
            pdf_update,
            case_dir_input_update,
            resolved_case_dir,
            html,
            html,
            "Step 3 · Causal Targets & Strategies",
            gr.update(value="Run Step 3"),
            3,
            graph_btn_update,
            graph_html_value,
        )

    if step_state == 3:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        edge_path = find_json(resolved_case_dir, "edge_select")

        if not edge_path or not os.path.exists(edge_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 3 · Causal Targets & Strategies</summary>"
                "<div style='margin-top:4px; padding:8px 12px; border-radius:12px;"
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "No edge_select.json found."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(3)
            return (
                pdf_update,
                case_dir_input_update,
                resolved_case_dir,
                previous_html,
                html,
                "Step 3 · Causal Targets & Strategies",
                gr.update(value="Run Step 3"),
                3,
                graph_btn_update,
                graph_html_value,
            )

        with open(edge_path, "r", encoding="utf-8") as f:
            edge_output = json.load(f)

        final_html = build_step3_block(edge_output, previous_html)

        pdf_update = gr.update(value=None, visible=True)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(3)

        return (
            pdf_update,
            case_dir_input_update,
            resolved_case_dir,
            final_html,
            final_html,
            "Step 3 · Causal Targets & Strategies",
            gr.update(value="Run Step 4"),
            4,
            graph_btn_update,
            graph_html_value,
        )

    if step_state == 4:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        candidates_path = find_json(resolved_case_dir, "candidates")

        if not candidates_path or not os.path.exists(candidates_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 4 · Candidate regimens</summary>"
                "<div style='margin-top:4px; padding:8px 12px; border-radius:12px;"
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "candidates.json not found in the case directory. Please run get_regimen_synthesis.py."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(4)
            return (
                pdf_update,
                case_dir_input_update,
                resolved_case_dir,
                html,
                html,
                "Step 4 · Candidate regimens",
                gr.update(value="Run Step 4"),
                4,
                graph_btn_update,
                graph_html_value,
            )

        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)

        html = build_step4_block(candidates_data, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(4)

        return (
            pdf_update,
            case_dir_input_update,
            resolved_case_dir,
            html,
            html,
            "Step 4 · Candidate regimens",
            gr.update(value="Run Step 4"),
            4,
            graph_btn_update,
            graph_html_value,
        )

    graph_html_value = get_graph_html(1)
    return (
        pdf_update,
        case_dir_input_update,
        current_case_dir or "",
        previous_html,
        previous_html,
        "Step 1 · Initial Diagnosis",
        gr.update(value="Run Step 1"),
        1,
        graph_btn_update,
        graph_html_value,
    )


def reset_all():
    initial_html = (
        "<details open class='step-card'>"
        "<summary>Workflow</summary>"
        "<div style='margin-top:4px; padding:8px 12px; border-radius:12px; background:#f9fafb;"
        "border:1px solid #e5e7eb; font-size:12px; color:#9ca3af;'>"
        "Waiting for analysis..."
        "</div>"
        "</details>"
    )
    graph_html_value = get_graph_html(1)
    return (
        gr.update(value=None, visible=True),
        "example/case1",
        "example/case1",
        "",
        initial_html,
        "Step 1 · Initial Diagnosis",
        gr.update(value="Run Step 1"),
        1,
        gr.update(visible=False),
        graph_html_value,
    )


def create_interface():
    initial_case_dir = "example/case1"
    initial_step = 1
    initial_graph_html = get_graph_html(initial_step)

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .btn-sm button {
            padding: 6px 16px !important;
            font-size: 12px !important;
            border-radius: 10px !important;
        }
        details.step-card {
            border-radius: 12px;
            margin-top: 6px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 4px 8px 8px 8px;
        }
        details.step-card > summary {
            list-style: none;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            color: #374151;
            outline: none;
        }
        details.step-card > summary::-webkit-details-marker {
            display: none;
        }
        details.step-card[open] > summary {
            color: #111827;
        }
        """,
    ) as demo:
        gr.Markdown(
            "<h1 style='text-align:center; font-size:22px; font-weight:700; margin-bottom:4px;'>"
            "HPP-BioHealth"
            "</h1>"
        )

        case_dir_state = gr.State(initial_case_dir)
        step_html_state = gr.State(
            "<details open class='step-card'>"
            "<summary>Workflow</summary>"
            "<div style='margin-top:4px; padding:8px 12px; border-radius:12px; background:#f9fafb;"
            "border:1px solid #e5e7eb; font-size:12px; color:#9ca3af;'>"
            "Waiting for analysis..."
            "</div>"
            "</details>"
        )
        step_state = gr.State(initial_step)
        graph_html_state = gr.State(initial_graph_html)

        with gr.Row():
            with gr.Column(scale=1):
                graph_btn = gr.Button(
                    "Render 3D Graph",
                    variant="secondary",
                    elem_classes=["btn-sm"],
                    visible=False,
                )
                graph_html = gr.HTML(
                    value=initial_graph_html,
                    visible=True,
                )
                step_title = gr.Markdown(
                    "Step 1 · Initial Diagnosis",
                    elem_id="step-title",
                )
                pdf_input = gr.File(
                    label="PDF",
                    file_types=[".pdf"],
                    type="filepath",
                    height=110,
                )
                case_dir_input = gr.Textbox(
                    label="Case directory",
                    value=initial_case_dir,
                )
                with gr.Row():
                    main_btn = gr.Button(
                        "Run Step 1",
                        variant="primary",
                        elem_classes=["btn-sm"],
                    )
                    clear_btn = gr.Button(
                        "Reset",
                        variant="secondary",
                        elem_classes=["btn-sm"],
                    )

            with gr.Column(scale=1):
                render_html = gr.HTML(
                    value=step_html_state.value,
                )

        run_event = main_btn.click(
            fn=run_flow,
            inputs=[
                pdf_input,
                case_dir_input,
                case_dir_state,
                step_html_state,
                step_state,
            ],
            outputs=[
                pdf_input,
                case_dir_input,
                case_dir_state,
                step_html_state,
                render_html,
                step_title,
                main_btn,
                step_state,
                graph_btn,
                graph_html_state,
            ],
        )

        run_event.then(
            fn=lambda html: gr.update(value=html, visible=True),
            inputs=graph_html_state,
            outputs=graph_html,
        )

        reset_event = clear_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[
                pdf_input,
                case_dir_input,
                case_dir_state,
                step_html_state,
                render_html,
                step_title,
                main_btn,
                step_state,
                graph_btn,
                graph_html_state,
            ],
        )

        reset_event.then(
            fn=lambda html: gr.update(value=html, visible=True),
            inputs=graph_html_state,
            outputs=graph_html,
        )

        graph_btn.click(
            fn=render_graph,
            inputs=[step_state],
            outputs=graph_html,
        )

    return demo


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False,
    )
