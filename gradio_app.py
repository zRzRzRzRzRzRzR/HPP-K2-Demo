import json
import os
import time
from typing import Optional

import gradio as gr

from get_abnormal_node import run_abnormal_node_selection
from get_exam_node import run_examination_node_selection
from pdf_converter import convert_pdf_to_json


def resolve_case_dir(case_dir_raw: str) -> str:
    if not case_dir_raw:
        return ""
    case_dir = case_dir_raw.strip()
    if not case_dir:
        return ""
    if not os.path.isabs(case_dir):
        case_dir = os.path.abspath(case_dir)
    return case_dir


def get_diagnosis_path(pdf_path: Optional[str], case_dir_raw: Optional[str]) -> str:
    case_dir = resolve_case_dir(case_dir_raw or "")

    # 1) Prefer existing diagnosis.json in case_dir
    if case_dir:
        candidate = os.path.join(case_dir, "diagnosis.json")
        if os.path.exists(candidate):
            return candidate

    # 2) Otherwise parse uploaded PDF
    if pdf_path:
        diagnosis_path = convert_pdf_to_json(pdf_path)
        if isinstance(diagnosis_path, str) and os.path.exists(diagnosis_path):
            return diagnosis_path
        raise FileNotFoundError(
            f"convert_pdf_to_json did not produce a valid diagnosis.json (got: {diagnosis_path})"
        )

    # 3) No valid source
    raise RuntimeError(
        "No diagnosis.json found and no PDF provided. "
        "Upload a PDF or specify a case directory containing diagnosis.json."
    )


def get_exam_nodes(diagnosis_path: str):
    node_str = run_examination_node_selection(diagnosis_path=diagnosis_path)

    preview = ""
    try:
        with open(diagnosis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        preview = json.dumps(data, ensure_ascii=False, indent=2)[:800]
    except Exception:
        pass

    return node_str, preview


def simulate_workflow(pdf_file, case_dir):
    # Resolve PDF path if provided
    pdf_path = None
    if isinstance(pdf_file, str):
        pdf_path = pdf_file
    elif hasattr(pdf_file, "name"):
        pdf_path = pdf_file.name

    # First frame: show loading bubble on the right
    loading_html = (
        "<div style='margin-top:4px; padding:10px 12px; border-radius:12px; "
        "background:#eef2ff; border:1px solid #6366f1; font-size:13px; "
        "color:#111827; display:flex; align-items:center; gap:6px;'>"
        "<span>⏳</span>"
        "<span>Running Step 1 pipeline: parsing case and selecting initial examination nodes...</span>"
        "</div>"
    )
    yield loading_html
    time.sleep(0.15)

    try:
        diagnosis_path = get_diagnosis_path(pdf_path, case_dir)
        node_str, diagnosis_preview = get_exam_nodes(diagnosis_path)
    except Exception as e:
        error_html = (
            "<div style='margin-top:4px; padding:10px 12px; border-radius:12px; "
            "background:#fef2f2; border:1px solid #f87171; font-size:13px; color:#7f1d1d;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>Step 1 · Parse case & suggest initial examinations</div>"
            "<div>Pipeline execution failed.</div>"
            f"<div style='margin-top:4px; font-size:12px;'><b>Error:</b> <code>{repr(e)}</code></div>"
            "<ul style='margin:6px 0 0 18px; font-size:12px;'>"
            "<li>If a case directory is set, ensure <code>diagnosis.json</code> exists there.</li>"
            "<li>If using a PDF, verify <code>convert_pdf_to_json</code> runs successfully.</li>"
            "<li>Verify <code>run_examination_node_selection</code> works on that diagnosis.json.</li>"
            "</ul>"
            "</div>"
        )
        yield error_html
        return

    if node_str:
        result_html = (
            "<div style='margin-top:4px; padding:12px 14px; border-radius:14px; "
            "background:#ecfdf5; border:1px solid #22c55e; font-size:13px; color:#064e3b;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>"
            "Step 1 · Parse case & suggest initial examinations</div>"
            "<div style='margin-bottom:4px;'>Status: ✅ Completed</div>"
            "<div style='margin-bottom:6px; font-size:12px; color:#047857;'>"
            "Pipeline: <code>PDF / diagnosis.json → LLM (get_json) → node_id suggestions</code>"
            "</div>"
            "<div style='font-weight:500; margin-bottom:2px;'>Recommended examination node_ids (raw output):</div>"
            "<pre style='margin:0; margin-top:2px; padding:8px; border-radius:8px; "
            "background:#f9fafb; border:1px solid #d1d5db; font-size:11px; "
            "white-space:pre-wrap; word-break:break-word; color:#111827;'>"
            f"{node_str}"
            "</pre>"
            "<div style='margin-top:6px; font-size:10px; color:#6b7280;'>"
            "This is a model-generated demo output and does not constitute medical advice."
            "</div>"
            "</div>"
        )
        if diagnosis_preview:
            result_html += (
                "<details style='margin-top:6px; font-size:11px; color:#4b5563;'>"
                "<summary>Show diagnosis.json preview (debug)</summary>"
                "<pre style='margin-top:4px; padding:8px; border-radius:8px; "
                "background:#f9fafb; border:1px solid #e5e7eb; font-size:10px; "
                "white-space:pre-wrap; word-break:break-word; color:#111827;'>"
                f"{diagnosis_preview}"
                "</pre>"
                "</details>"
            )
        yield result_html
    else:
        warn_html = (
            "<div style='margin-top:4px; padding:10px 12px; border-radius:12px; "
            "background:#fffbeb; border:1px solid #fbbf24; font-size:13px; color:#92400e;'>"
            "<div style='font-weight:600; margin-bottom:4px;'>"
            "Step 1 · Parse case & suggest initial examinations</div>"
            "<div>Status: ⚠ No node_id list returned.</div>"
            "<div style='margin-top:4px; font-size:12px;'>"
            "Please run <code>get_exam_node.py</code> manually and check that "
            "<code>utils.call_large_model_llm</code> extracts the &lt;answer&gt;[...]&lt;/answer&gt; block."
            "</div>"
            "</div>"
        )
        yield warn_html


def reset_all():
    return None, (
        "<div style='font-size:13px; color:#6b7280;'>"
        "Step 1 status and results will appear here after you start the analysis."
        "</div>"
    )


def create_interface():
    with gr.Blocks(
        title="HPP-Bio Health · Causal Diagnostic Orchestrator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "<h1 style='text-align:center; margin-bottom:4px;'>HPP-Bio Health</h1>"
            "<p style='text-align:center; color:#6b7280; font-size:14px;'>"
            "Causal diagnostic orchestrator for structured intake and examination planning."
            "</p>"
        )

        gr.Markdown("## Step 1 · Parse case & suggest initial examinations")

        with gr.Row():
            # Left: controls (smaller)
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Upload PDF (optional if diagnosis.json already exists)",
                    file_types=[".pdf"],
                    type="filepath",
                )

                case_dir_input = gr.Textbox(
                    label="Case directory (optional, e.g. example/case1)",
                    value="example/case1",
                    placeholder=(
                        "If set and diagnosis.json exists here, it will be used directly "
                        "and PDF parsing will be skipped."
                    ),
                )

                with gr.Row():
                    start_btn = gr.Button("Start Analysis", variant="primary")
                    clear_btn = gr.Button("Reset", variant="secondary")

            # Right: output only (larger)
            with gr.Column(scale=2):
                gr.Markdown("### Model Rendering Panel")
                render_html = gr.HTML(
                    "<div style='font-size:13px; color:#6b7280;'>"
                    "Step 1 status and results will appear here."
                    "</div>"
                )

        start_btn.click(
            fn=simulate_workflow,
            inputs=[pdf_input, case_dir_input],
            outputs=[render_html],
        )

        clear_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[pdf_input, render_html],
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
