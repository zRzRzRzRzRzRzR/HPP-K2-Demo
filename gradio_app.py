import json
import os

import gradio as gr

from get_abnormal_node import run_abnormal_node_selection
from get_exam_node import run_examination_node_selection
from get_report import run_therapy_record_selection
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


def render_graph(step_state):
    html = get_graph_html(step_state or 1)
    return gr.update(value=html, visible=True)


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_step2_block(abnormal_output, previous_html: str) -> str:
    abnormal_data = abnormal_output or {}
    points = (
        abnormal_data.get("abnormal_points")
        or abnormal_data.get("abnormal_nodes")
        or abnormal_data.get("abnormals")
        or abnormal_data.get("items")
    )

    if not isinstance(points, list):
        raw_str = (
            json.dumps(abnormal_output, ensure_ascii=False, indent=2)
            if isinstance(abnormal_output, (dict, list))
            else str(abnormal_output)
        )
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 2 · Highlighted abnormal indicators (raw)</summary>"
            "<div class='step-body green'>"
            "<pre class='code-small'>"
            f"{html_escape(raw_str)}"
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
            f"<td class='cell'>{html_escape(node_id)}</td>"
            f"<td class='cell' style='{val_style}'>{html_escape(val_text)}</td>"
            f"<td class='cell muted'>{html_escape(unit_text)}</td>"
            "</tr>"
        )

    if not rows_html:
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 2 · Highlighted abnormal indicators</summary>"
            "<div class='step-body red'>"
            "No abnormal indicators parsed. Check the abnormal-node script output format."
            "</div>"
            "</details>"
        )

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 2 · Highlighted abnormal indicators</summary>"
        "<div class='step-body green'>"
        "<table class='table'>"
        "<thead>"
        "<tr>"
        "<th class='th'>node_id</th>"
        "<th class='th'>abnormal value</th>"
        "<th class='th'>unit</th>"
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
        raw_str = (
            json.dumps(edge_output, ensure_ascii=False, indent=2)
            if isinstance(edge_output, (dict, list))
            else str(edge_output)
        )
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 3 · Causal targets & strategies (raw)</summary>"
            "<div class='step-body indigo'>"
            "<pre class='code-small'>"
            f"{html_escape(raw_str)}"
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
                    dn = d.get("name") or d.get("drugName") or d.get("label")
                    if dn:
                        drug_names.append(str(dn))
                else:
                    drug_names.append(str(d))
            drugs = ", ".join(drug_names)
        current_text = "-" if current in (None, "", []) else str(current)
        target_text = "-" if target in (None, "", []) else str(target)
        drugs_text = "-" if drugs in (None, "", []) else str(drugs)
        rows_html += (
            "<tr>"
            f"<td class='cell'>{html_escape(node_id)}</td>"
            f"<td class='cell'>{html_escape(label)}</td>"
            f"<td class='cell bad'>{html_escape(current_text)}</td>"
            f"<td class='cell good'>{html_escape(target_text)}</td>"
            f"<td class='cell'>{html_escape(drugs_text)}</td>"
            "</tr>"
        )

    if not rows_html:
        return previous_html

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 3 · Causal targets & strategies</summary>"
        "<div class='step-body indigo'>"
        "<table class='table'>"
        "<thead>"
        "<tr>"
        "<th class='th'>node_id</th>"
        "<th class='th'>indicator</th>"
        "<th class='th'>current</th>"
        "<th class='th'>target</th>"
        "<th class='th'>recommended drugs</th>"
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
            "<div class='step-body gray'>"
            "No regimen candidates found. Please run get_regimen_synthesis.py to generate candidates.json."
            "</div>"
            "</details>"
        )

    rows_html = ""
    for idx, reg in enumerate(regimens[:30], start=1):
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
            f"<td class='cell muted'>{idx}</td>"
            f"<td class='cell strong'>{html_escape(regimen_str)}</td>"
            f"<td class='cell'>{overall_txt}</td>"
            f"<td class='cell'>{eff_txt}</td>"
            f"<td class='cell bad'>{safety_txt}</td>"
            f"<td class='cell'>{adh_txt}</td>"
            f"<td class='cell'>{cost_txt}</td>"
            "</tr>"
        )

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 4 · Candidate regimens (from get_regimen_synthesis)</summary>"
        "<div class='step-body yellow'>"
        "<div class='note'>Higher overall score indicates better balance of efficacy, safety, adherence, and cost.</div>"
        "<table class='table'>"
        "<thead>"
        "<tr>"
        "<th class='th'>rank</th>"
        "<th class='th'>regimen</th>"
        "<th class='th'>overall</th>"
        "<th class='th'>efficacy</th>"
        "<th class='th'>risk</th>"
        "<th class='th'>adherence</th>"
        "<th class='th'>cost</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def build_step5_block(pre_rx_output, previous_html: str) -> str:
    if isinstance(pre_rx_output, dict):
        data = pre_rx_output
    else:
        data = {}
    paths = data.get("treatment_paths") or data.get("paths") or []
    if not isinstance(paths, list) or not paths:
        return (
            previous_html + "<details open class='step-card'>"
            "<summary>Step 5 · Final regimen pathways</summary>"
            "<div class='step-body gray'>"
            "No treatment paths found. Please run get_drug.py to generate pre_rx.json."
            "</div>"
            "</details>"
        )

    rows_html = ""
    for idx, p in enumerate(paths[:80], start=1):
        drug = str(p.get("drug") or "").strip()
        drug_class = str(p.get("drug_class") or "").strip()
        on_target = str(p.get("on_target_node") or "").strip()
        delta = p.get("on_target_delta")
        dst = str(p.get("dst") or "").strip()
        edges = p.get("edges") or []
        if isinstance(edges, list):
            edges_str = " → ".join([str(e) for e in edges[:4]])
            if len(edges) > 4:
                edges_str += " ..."
        else:
            edges_str = str(edges)
        coef = (
            p.get("path_coef_linear")
            if p.get("path_coef_linear") is not None
            else p.get("path_risk_log_coef")
        )
        delta_txt = "-" if delta is None else f"{float(delta):.3g}"
        impact_txt = "-" if coef is None else f"{float(coef):.3g}"
        rows_html += (
            "<tr>"
            f"<td class='cell muted'>{idx}</td>"
            f"<td class='cell strong'>{html_escape(drug)}</td>"
            f"<td class='cell muted'>{html_escape(drug_class)}</td>"
            f"<td class='cell'>{html_escape(on_target)}</td>"
            f"<td class='cell'>{delta_txt}</td>"
            f"<td class='cell'>{html_escape(dst)}</td>"
            f"<td class='cell'>{html_escape(edges_str)}</td>"
            f"<td class='cell'>{impact_txt}</td>"
            "</tr>"
        )

    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 5 · Final regimen pathways (from pre_rx.json)</summary>"
        "<div class='step-body teal'>"
        "<div class='note'>Each row shows a drug, its key target, expected change, and causal path towards outcomes.</div>"
        "<table class='table'>"
        "<thead>"
        "<tr>"
        "<th class='th'>#</th>"
        "<th class='th'>drug</th>"
        "<th class='th'>class</th>"
        "<th class='th'>on-target node</th>"
        "<th class='th'>Δ on-target</th>"
        "<th class='th'>downstream outcome</th>"
        "<th class='th'>path (edges)</th>"
        "<th class='th'>path impact</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


def find_report_txt(case_dir: str):
    if not case_dir:
        return None
    candidates = [
        "therapy_record.txt",
        "final_report.txt",
        "report.txt",
        "pre_rx_report.txt",
        "treatment_report.txt",
    ]
    for name in candidates:
        path = os.path.join(case_dir, name)
        if os.path.exists(path):
            return path
    return None


def build_step6_block(report_text: str, previous_html: str) -> str:
    escaped = html_escape(report_text)
    return (
        previous_html + "<details open class='step-card'>"
        "<summary>Step 6 · Final clinical report</summary>"
        "<div class='step-body dark'>"
        "<div class='note'>This is the final synthesized recommendation based on all previous steps.</div>"
        "<pre class='code-small' style='background:#111827; color:#f9fafb; border-color:#111827;'>"
        f"{escaped}"
        "</pre>"
        "</div>"
        "</details>"
    )


def auto_run_report(case_dir: str) -> str:
    diagnosis_path = (
        find_json(case_dir, "diagnosis_exam")
        or find_exam_source(case_dir)
        or find_json(case_dir, "diagnosis")
    )
    if not diagnosis_path:
        return ""
    node_path = "hpp_data/node.json"
    plan_path = (
        os.path.join(case_dir, "plan.json")
        if os.path.exists(os.path.join(case_dir, "plan.json"))
        else (
            find_json(case_dir, "pre_rx")
            or find_json(case_dir, "candidates")
            or os.path.join(case_dir, "pre_rx.json")
        )
    )
    if not plan_path or not os.path.exists(plan_path):
        return ""
    try:
        resp = run_therapy_record_selection(
            diagnosis_path=diagnosis_path,
            node_path=node_path,
            plan_path=plan_path,
        )
    except Exception:
        return ""
    if not isinstance(resp, str):
        resp = str(resp)
    out_path = os.path.join(case_dir, "therapy_record.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp)
    except Exception:
        pass
    return resp


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
    final_report_html_update = gr.update(value="", visible=False)
    final_report_file_update = gr.update(value=None, visible=False)

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
                "<div class='step-body red'>"
                "No valid diagnosis JSON/PDF found. Please upload a valid report or specify the correct case directory."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(1)
            return (
                gr.update(),
                gr.update(value=case_dir),
                case_dir,
                error_html,
                error_html,
                "Step 1 · Initial Diagnosis",
                gr.update(value="Run Step 1"),
                1,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        exam_nodes = run_examination_node_selection(
            diagnosis_path,
            "hpp_data/node.json",
        )
        node_str = (
            json.dumps(exam_nodes, ensure_ascii=False, indent=2)
            if isinstance(exam_nodes, dict)
            else str(exam_nodes)
        )
        diagnosis_preview = build_diagnosis_preview_html(diagnosis_path)
        step1_html = (
            "<details open class='step-card'>"
            "<summary>Step 1 · Examination node suggestions</summary>"
            "<div class='step-body green'>"
            "<pre class='code-small'>"
            f"{html_escape(node_str)}"
            "</pre>"
            f"{diagnosis_preview}"
            "</div>"
            "</details>"
        )
        case_dir_input_update = gr.update(value=case_dir)
        graph_html_value = get_graph_html(2)
        return (
            gr.update(),
            case_dir_input_update,
            case_dir,
            step1_html,
            step1_html,
            "Step 2 · Full Examination",
            gr.update(value="Run Step 2"),
            2,
            graph_btn_update,
            graph_html_value,
            final_report_html_update,
            final_report_file_update,
        )

    if step_state == 2:
        if not previous_html:
            previous_html = (
                "<details open class='step-card'>"
                "<summary>Step 1 · Initial Diagnosis</summary>"
                "<div class='step-body gray'>Run Step 1 first.</div>"
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
                "<div class='step-body gray'>No valid full examination JSON/PDF found.</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(2)
            return (
                gr.update(),
                case_dir_input_update,
                resolved_case_dir,
                previous_html,
                html,
                "Step 2 · Full Examination",
                gr.update(value="Run Step 2"),
                2,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        abnormal_output = run_abnormal_node_selection(
            exam_json_path,
            "hpp_data/node.json",
        )
        html = build_step2_block(abnormal_output, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(3)
        pdf_update = gr.update(value=None, visible=False)
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
            final_report_html_update,
            final_report_file_update,
        )

    if step_state == 3:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        edge_path = find_json(resolved_case_dir, "edge_select")
        if not edge_path or not os.path.exists(edge_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 3 · Causal Targets & Strategies</summary>"
                "<div class='step-body gray'>No edge_select.json found.</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(3)
            return (
                gr.update(value=None, visible=False),
                case_dir_input_update,
                resolved_case_dir,
                previous_html,
                html,
                "Step 3 · Causal Targets & Strategies",
                gr.update(value="Run Step 3"),
                3,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        with open(edge_path, "r", encoding="utf-8") as f:
            edge_output = json.load(f)
        final_html = build_step3_block(edge_output, previous_html)
        pdf_update = gr.update(value=None, visible=False)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(3)
        return (
            pdf_update,
            case_dir_input_update,
            resolved_case_dir,
            final_html,
            final_html,
            "Step 4 · Candidate regimens",
            gr.update(value="Run Step 4"),
            4,
            graph_btn_update,
            graph_html_value,
            final_report_html_update,
            final_report_file_update,
        )

    if step_state == 4:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        candidates_path = find_json(resolved_case_dir, "candidates")
        if not candidates_path or not os.path.exists(candidates_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 4 · Candidate regimens</summary>"
                "<div class='step-body gray'>"
                "candidates.json not found in the case directory. Please run get_regimen_synthesis.py."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(4)
            return (
                gr.update(value=None, visible=False),
                case_dir_input_update,
                resolved_case_dir,
                html,
                html,
                "Step 4 · Candidate regimens",
                gr.update(value="Run Step 4"),
                4,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
        html = build_step4_block(candidates_data, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(4)
        return (
            gr.update(value=None, visible=False),
            case_dir_input_update,
            resolved_case_dir,
            html,
            html,
            "Step 5 · Final regimen pathways",
            gr.update(value="Run Step 5"),
            5,
            graph_btn_update,
            graph_html_value,
            final_report_html_update,
            final_report_file_update,
        )

    if step_state == 5:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        pre_rx_path = find_json(resolved_case_dir, "pre_rx")
        if not pre_rx_path or not os.path.exists(pre_rx_path):
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 5 · Final regimen pathways</summary>"
                "<div class='step-body gray'>"
                "pre_rx.json not found in the case directory. Please run get_drug.py."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(5)
            return (
                gr.update(value=None, visible=False),
                case_dir_input_update,
                resolved_case_dir,
                html,
                html,
                "Step 5 · Final regimen pathways",
                gr.update(value="Run Step 5"),
                5,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        with open(pre_rx_path, "r", encoding="utf-8") as f:
            pre_rx_data = json.load(f)
        html = build_step5_block(pre_rx_data, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(5)
        return (
            gr.update(value=None, visible=False),
            case_dir_input_update,
            resolved_case_dir,
            html,
            html,
            "Step 6 · Final clinical report",
            gr.update(value="Run Step 6"),
            6,
            graph_btn_update,
            graph_html_value,
            final_report_html_update,
            final_report_file_update,
        )

    if step_state == 6:
        resolved_case_dir = resolve_case_dir(current_case_dir)
        report_path = find_report_txt(resolved_case_dir)
        report_text = ""
        if report_path and os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
        if not report_text:
            generated = auto_run_report(resolved_case_dir)
            if generated:
                report_text = generated
                gen_path = os.path.join(resolved_case_dir, "therapy_record.txt")
                if os.path.exists(gen_path):
                    report_path = gen_path
        if not report_text:
            html = (
                previous_html + "<details open class='step-card'>"
                "<summary>Step 6 · Final clinical report</summary>"
                "<div class='step-body gray'>"
                "Failed to generate final report. Please check get_report.py configuration."
                "</div>"
                "</details>"
            )
            graph_html_value = get_graph_html(6)
            return (
                gr.update(value=None, visible=False),
                case_dir_input_update,
                resolved_case_dir,
                html,
                html,
                "Step 6 · Final clinical report",
                gr.update(value="Run Step 6"),
                6,
                graph_btn_update,
                graph_html_value,
                final_report_html_update,
                final_report_file_update,
            )
        left_html = (
            "<div style='margin-top:10px; padding:10px 12px; border-radius:12px;"
            "background:#111827; color:#f9fafb; font-size:11px; line-height:1.6;"
            "white-space:pre-wrap; border:1px solid #111827;'>"
            f"{html_escape(report_text)}"
            "</div>"
        )
        final_report_html_update = gr.update(value=left_html, visible=True)
        if report_path and os.path.exists(report_path):
            final_report_file_update = gr.update(value=report_path, visible=True)
        html = build_step6_block(report_text, previous_html)
        case_dir_input_update = gr.update(value=resolved_case_dir)
        graph_html_value = get_graph_html(6)
        return (
            gr.update(value=None, visible=False),
            case_dir_input_update,
            resolved_case_dir,
            html,
            html,
            "Step 6 · Final clinical report",
            gr.update(value="Run Step 6"),
            6,
            graph_btn_update,
            graph_html_value,
            final_report_html_update,
            final_report_file_update,
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
        final_report_html_update,
        final_report_file_update,
    )


def reset_all():
    initial_html = (
        "<details open class='step-card'>"
        "<summary>Workflow</summary>"
        "<div class='step-body gray'>Waiting for analysis...</div>"
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
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
    )


def create_interface():
    initial_case_dir = "example/case1"
    initial_step = 1
    initial_graph_html = get_graph_html(initial_step)
    initial_workflow_html = (
        "<details open class='step-card'>"
        "<summary>Workflow</summary>"
        "<div class='step-body gray'>Waiting for analysis...</div>"
        "</details>"
    )

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
        .step-body {
            margin-top: 4px;
            padding: 10px 12px;
            border-radius: 10px;
            font-size: 12px;
        }
        .step-body.green { background:#ecfdf5; border:1px solid #22c55e; color:#064e3b; }
        .step-body.indigo { background:#eef2ff; border:1px solid #6366f1; color:#312e81; }
        .step-body.yellow { background:#fefce8; border:1px solid #facc15; color:#78350f; }
        .step-body.teal { background:#ecfeff; border:1px solid #22c55e; color:#115e59; }
        .step-body.red { background:#fef2f2; border:1px solid #ef4444; color:#991b1b; }
        .step-body.gray { background:#f9fafb; border:1px solid #e5e7eb; color:#6b7280; }
        .step-body.dark { background:#111827; border:1px solid #111827; color:#f9fafb; }
        .table { border-collapse:collapse; width:100%; }
        .th {
            text-align:left;
            padding:4px 6px;
            font-size:9px;
            color:#6b7280;
            font-weight:500;
        }
        .cell {
            padding:4px 6px;
            font-size:9px;
            color:#111827;
            border-bottom:1px solid #e5e7eb;
        }
        .cell.strong { font-weight:600; }
        .cell.muted { color:#6b7280; }
        .cell.good { color:#16a34a; font-weight:600; }
        .cell.bad { color:#dc2626; font-weight:600; }
        .note {
            font-size:10px;
            color:#6b7280;
            margin-bottom:4px;
        }
        .code-small {
            margin:0;
            padding:6px 8px;
            border-radius:10px;
            background:#f9fafb;
            border:1px solid #d1d5db;
            font-size:10px;
            white-space:pre-wrap;
            word-break:break-word;
            color:#111827;
        }
        """,
    ) as demo:
        gr.Markdown(
            "<h1 style='text-align:center; font-size:22px; font-weight:700; margin-bottom:4px;'>"
            "HPP-BioHealth"
            "</h1>"
        )

        case_dir_state = gr.State(initial_case_dir)
        step_html_state = gr.State(initial_workflow_html)
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
                final_report_html = gr.HTML(
                    value="",
                    visible=False,
                )
                final_report_file = gr.File(
                    label="Download final report",
                    interactive=False,
                    visible=False,
                )

            with gr.Column(scale=1):
                render_html = gr.HTML(
                    value=initial_workflow_html,
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
                final_report_html,
                final_report_file,
            ],
            show_progress=False,
        )

        run_event.then(
            fn=lambda html: gr.update(value=html, visible=True),
            inputs=graph_html_state,
            outputs=graph_html,
            show_progress=False,
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
                final_report_html,
                final_report_file,
            ],
            show_progress=False,
        )

        reset_event.then(
            fn=lambda html: gr.update(value=html, visible=True),
            inputs=graph_html_state,
            outputs=graph_html,
            show_progress=False,
        )

        graph_btn.click(
            fn=render_graph,
            inputs=[step_state],
            outputs=graph_html,
            show_progress=False,
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
