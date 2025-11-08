import json
import os

import gradio as gr

from get_abnormal_node import run_abnormal_node_selection
from get_exam_node import run_examination_node_selection
from pdf_converter import convert_pdf_to_json
from utils_gradio import (
    HTML_TEMPLATE,
    PLACEHOLDER_HTML,
    VIEW_HEIGHT,
    build_diagnosis_preview_html,
    find_diagnosis_json,
    find_edge_select,
    find_exam_source,
    resolve_case_dir,
)


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
            previous_html
            + "<div style='margin-top:8px; padding:10px 12px; border-radius:14px;"
            "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>Step 2 · Highlighted abnormal indicators (raw)</div>"
            "<pre style='margin-top:4px; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{abnormal_output}"
            "</pre>"
            "</div>"
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
            previous_html
            + "<div style='margin-top:8px; padding:10px 12px; border-radius:14px;"
            "background:#fef2f2; border:1px solid #ef4444; font-size:12px; color:#991b1b;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>Step 2 · No abnormal indicators parsed</div>"
            "<div>Check the abnormal-node script output format.</div>"
            "</div>"
        )

    return (
        previous_html
        + "<div style='margin-top:8px; padding:10px 12px; border-radius:14px;"
        "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
        "<div style='font-weight:600; margin-bottom:4px;'>Step 2 · Highlighted abnormal indicators</div>"
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
            previous_html
            + "<div style='margin-top:8px; padding:10px 12px; border-radius:14px;"
            "background:#eef2ff; border:1px solid #6366f1; font-size:12px; color:#312e81;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>Step 3 · Causal targets & strategies (raw)</div>"
            "<pre style='margin-top:4px; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{edge_output}"
            "</pre>"
            "</div>"
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
        previous_html
        + "<div style='margin-top:8px; padding:10px 12px; border-radius:14px;"
        "background:#eef2ff; border:1px solid #6366f1; font-size:12px; color:#312e81;'>"
        "<div style='font-weight:600; margin-bottom:4px;'>Step 3 · Causal targets & strategies</div>"
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
    )


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
    graph_html_update = gr.update(value=PLACEHOLDER_HTML, visible=False)

    if step_state == 1:
        case_dir = resolve_case_dir(case_dir_raw) or current_case_dir or ""

        if pdf_path:
            diagnosis_path = convert_pdf_to_json(pdf_path)
            if not case_dir:
                case_dir = os.path.dirname(diagnosis_path)
        else:
            diagnosis_path = find_diagnosis_json(case_dir)

        if not diagnosis_path or not os.path.exists(diagnosis_path):
            error_html = (
                "<div style='margin-top:6px; padding:10px 12px; border-radius:14px;"
                "background:#fef2f2; border:1px solid #ef4444; font-size:12px; color:#991b1b;'>"
                "<div style='font-weight:600; margin-bottom:4px;'>Step 1 · Failed</div>"
                "<div>No valid diagnosis JSON/PDF found. Please upload a valid report or specify the correct case directory.</div>"
                "</div>"
            )
            return (
                pdf_update,
                case_dir_input_update,
                case_dir,
                error_html,
                error_html,
                "Step 1 · Initial Diagnosis",
                gr.update(value="Run Step 1"),
                1,
                graph_btn_update,
                graph_html_update,
            )

        node_output = run_examination_node_selection(
            diagnosis_path,
            "hpp_data/node.json",
        )
        if isinstance(node_output, dict):
            node_str = json.dumps(node_output, ensure_ascii=False, indent=2)
        else:
            node_str = str(node_output)

        diagnosis_preview = build_diagnosis_preview_html(diagnosis_path)

        step1_html = (
            "<div style='margin-top:6px; padding:10px 12px; border-radius:14px;"
            "background:#ecfdf5; border:1px solid #22c55e; font-size:12px; color:#064e3b;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>Step 1 · Examination node suggestions</div>"
            "<pre style='margin:0; padding:6px 8px; border-radius:10px; background:#f9fafb;"
            "border:1px solid #d1d5db; font-size:10px; white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{node_str}"
            "</pre>"
            "</div>"
            f"{diagnosis_preview}"
        )

        case_dir_input_update = gr.update(value=case_dir)

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
            graph_html_update,
        )

    if step_state == 2:
        if not previous_html:
            previous_html = (
                "<div style='padding:8px 12px; border-radius:12px; background:#f9fafb;"
                "border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "Run Step 1 first."
                "</div>"
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
                previous_html
                + "<div style='margin-top:8px; padding:8px 12px; border-radius:12px;"
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "No valid full examination JSON/PDF found."
                "</div>"
            )
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
                graph_html_update,
            )

        abnormal_output = run_abnormal_node_selection(
            exam_json_path,
            "hpp_data/node.json",
        )

        html = build_step2_block(abnormal_output, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)

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
            graph_html_update,
        )

    if step_state == 3:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        edge_path = find_edge_select(resolved_case_dir)

        if not edge_path or not os.path.exists(edge_path):
            html = (
                previous_html
                + "<div style='margin-top:8px; padding:8px 12px; border-radius:12px;"
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:12px; color:#6b7280;'>"
                "No edge_select.json found."
                "</div>"
            )
            graph_btn_update = gr.update(visible=False)
            graph_html_update = gr.update(value=PLACEHOLDER_HTML, visible=False)
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
                graph_html_update,
            )

        with open(edge_path, "r", encoding="utf-8") as f:
            edge_output = json.load(f)

        final_html = build_step3_block(edge_output, previous_html)

        pdf_update = gr.update(value=None, visible=False)
        case_dir_input_update = gr.update(value=resolved_case_dir)

        # hide the extra button, auto render graph
        graph_btn_update = gr.update(visible=False)
        graph_html_update = render_graph(resolved_case_dir, 3)

        return (
            pdf_update,
            case_dir_input_update,
            resolved_case_dir,
            final_html,
            final_html,
            "Step 3 · Causal Targets & Strategies",
            gr.update(value="Run Step 3"),
            3,
            graph_btn_update,
            graph_html_update,
        )

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
        graph_html_update,
    )


def reset_all():
    initial_html = (
        "<div style='padding:8px 12px; border-radius:12px; background:#f9fafb;"
        "border:1px solid #e5e7eb; font-size:12px; color:#9ca3af;'>"
        "Waiting for analysis..."
        "</div>"
    )
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
        gr.update(value=PLACEHOLDER_HTML, visible=False),
    )


def render_graph(case_dir, step_state):
    if step_state != 3:
        return gr.update(value=PLACEHOLDER_HTML, visible=False)
    resolved = resolve_case_dir(case_dir) or "example/case1"
    json_path = os.path.join(resolved, "causal_graph.json")
    if not os.path.exists(json_path):
        json_path = "example/case1/causal_graph.json"
    if not os.path.exists(json_path):
        html = (
            "<div style='width:100%;height:{h}px;border:1px solid #fecaca;"
            "border-radius:8px;display:flex;align-items:center;justify-content:center;"
            "background:#fef2f2;color:#b91c1c;font-size:12px;'>"
            "causal_graph.json not found.</div>"
        ).format(h=VIEW_HEIGHT)
        return gr.update(value=html, visible=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("{GRAPH_JSON}", graph_json).replace(
        "{VIEW_HEIGHT}", str(VIEW_HEIGHT)
    )
    return gr.update(value=html, visible=True)


def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .btn-sm button {
            padding: 6px 16px !important;
            font-size: 12px !important;
            border-radius: 10px !important;
        }
        """,
    ) as demo:
        gr.Markdown(
            "<h1 style='text-align:center; font-size:22px; font-weight:700; margin-bottom:4px;'>"
            "HPP-BioHealth"
            "</h1>"
        )

        case_dir_state = gr.State("example/case1")
        step_html_state = gr.State("")
        step_state = gr.State(1)

        with gr.Row():
            with gr.Column(scale=1):
                step_title = gr.Markdown(
                    "Step 1 · Initial Diagnosis",
                    elem_id="step-title",
                )
                pdf_input = gr.File(
                    label="PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                case_dir_input = gr.Textbox(
                    label="Case directory",
                    value="example/case1",
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
                gr.Markdown(
                    "<div style='margin-top:8px; font-size:11px; font-weight:600; color:#111827;'>"
                    "3D Causal Network</div>"
                )
                graph_btn = gr.Button(
                    "Render 3D Graph",
                    variant="secondary",
                    elem_classes=["btn-sm"],
                    visible=False,
                )
                graph_html = gr.HTML(
                    value=PLACEHOLDER_HTML,
                    visible=False,
                )

            with gr.Column(scale=2):
                render_html = gr.HTML(
                    "<div style='padding:8px 12px; border-radius:12px; background:#f9fafb;"
                    "border:1px solid #e5e7eb; font-size:12px; color:#9ca3af;'>"
                    "Waiting for analysis..."
                    "</div>"
                )

        main_btn.click(
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
                graph_html,
            ],
        )

        clear_btn.click(
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
                graph_html,
            ],
        )

        graph_btn.click(
            fn=render_graph,
            inputs=[case_dir_state, step_state],
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
