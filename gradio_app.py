import gradio as gr
import time
import os
import pdf_converter
HTML_PATH = os.path.join("demo.html")

def load_render_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

def run_pdf_converter(pdf_path: str):
    if pdf_converter is None:
        return None
    if hasattr(pdf_converter, "convert_pdf_to_json"):
        return pdf_converter.convert_pdf_to_json(pdf_path)
    if hasattr(pdf_converter, "main"):
        return pdf_converter.main(pdf_path)
    if hasattr(pdf_converter, "run"):
        return pdf_converter.run(pdf_path)
    return None


def simulate_workflow(pdf_file):
    if pdf_file is None:
        msg = "❌ Please upload a PDF on the left, then click 'Start Analysis'."
        yield msg, "Waiting...", "Waiting...", "Waiting...", "Waiting...", ""
        return

    run_pdf_converter(pdf_file)

    step1_chunks = [
        "Step 1: Start etiology analysis.",
        "Parsing document structure and sections.",
        "Extracting key entities: symptoms, labs, timelines.",
        "Building initial hypothesis matrix for potential causes."
    ]
    step2_chunks = [
        "Step 2: Build causal chains.",
        "Detecting exposure–outcome and temporal relationships.",
        "Identifying potential confounders and mediators.",
        "Drafting candidate causal graph for later validation."
    ]
    step3_chunks = [
        "Step 3: Simulate interventions and treatments.",
        "Evaluating impact of different interventions on key nodes.",
        "Running sensitivity checks on high-uncertainty edges.",
        "Highlighting promising treatment strategies."
    ]
    step4_chunks = [
        "Step 4: Integrated summary.",
        "Combining model confidence with domain knowledge.",
        "Flagging uncertain conclusions and open questions.",
        "Preparing structured summary for final review."
    ]
    final_chunks = [
        "Final conclusion:",
        "1. Present the most plausible causal pathway.",
        "2. Recommend prioritized interventions.",
        "3. List missing data and follow-up checks."
    ]

    s1 = ""
    s2 = ""
    s3 = ""
    s4 = ""
    sf = ""

    for c in step1_chunks:
        s1 += c + "\n"
        yield s1, "Waiting...", "Waiting...", "Waiting...", "Waiting...", ""
        time.sleep(0.25)

    for c in step2_chunks:
        s2 += c + "\n"
        yield s1, s2, "Waiting...", "Waiting...", "Waiting...", ""
        time.sleep(0.25)

    for c in step3_chunks:
        s3 += c + "\n"
        yield s1, s2, s3, "Waiting...", "Waiting...", ""
        time.sleep(0.25)

    for c in step4_chunks:
        s4 += c + "\n"
        yield s1, s2, s3, s4, "Waiting...", ""
        time.sleep(0.25)

    for c in final_chunks:
        sf += c + "\n"
        yield s1, s2, s3, s4, sf, ""
        time.sleep(0.25)

    html_content = load_render_html()
    yield s1, s2, s3, s4, sf, html_content


def reset_all():
    return None, "Waiting...", "Waiting...", "Waiting...", "Waiting...", ""


def create_interface():
    with gr.Blocks(title="Causal PDF Analysis Demo") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<div style='text-align:center; font-size:22px; font-weight:600; padding:8px 0; border:1px solid #1b5e7a; background:#0d3b4c; color:#ffffff;'>PDF Upload Area</div>"
                )
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                start_btn = gr.Button("Start Analysis")
                gr.Markdown(
                    "<div style='text-align:center; font-size:18px; font-weight:600; padding:8px 0; margin-top:16px; border:1px solid #1b5e7a; background:#0d3b4c; color:#ffffff;'>Causal Reasoning Trace</div>"
                )

                with gr.Group():
                    gr.Markdown(
                        "<div style='font-weight:600; padding:6px 8px; background:#0d3b4c; color:#ffffff;'>Stage 1: Etiology Analysis</div>"
                    )
                    step1_output = gr.Markdown("Waiting...")

                with gr.Group():
                    gr.Markdown(
                        "<div style='font-weight:600; padding:6px 8px; background:#0d3b4c; color:#ffffff; margin-top:8px;'>Stage 2: Causal Chain Construction</div>"
                    )
                    step2_output = gr.Markdown("Waiting...")

                with gr.Group():
                    gr.Markdown(
                        "<div style='font-weight:600; padding:6px 8px; background:#0d3b4c; color:#ffffff; margin-top:8px;'>Stage 3: Intervention Simulation</div>"
                    )
                    step3_output = gr.Markdown("Waiting...")

                with gr.Group():
                    gr.Markdown(
                        "<div style='font-weight:600; padding:6px 8px; background:#0d3b4c; color:#ffffff; margin-top:8px;'>Stage 4: Integrated Summary</div>"
                    )
                    step4_output = gr.Markdown("Waiting...")

                with gr.Group():
                    gr.Markdown(
                        "<div style='font-weight:600; padding:6px 8px; background:#0d3b4c; color:#ffffff; margin-top:8px;'>Final Conclusion</div>"
                    )
                    final_output = gr.Markdown("Waiting...")

                clear_btn = gr.Button("Reset")

            with gr.Column(scale=2):
                gr.Markdown(
                    "<div style='text-align:center; font-size:24px; font-weight:700; padding:10px 0; border:1px solid #1b5e7a; background:#0d3b4c; color:#ffffff;'>Model Rendering Panel</div>"
                )
                render_html = gr.HTML("")

        start_btn.click(
            fn=simulate_workflow,
            inputs=[pdf_input],
            outputs=[
                step1_output,
                step2_output,
                step3_output,
                step4_output,
                final_output,
                render_html,
            ],
        )

        clear_btn.click(
            fn=reset_all,
            outputs=[
                pdf_input,
                step1_output,
                step2_output,
                step3_output,
                step4_output,
                final_output,
                render_html,
            ],
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