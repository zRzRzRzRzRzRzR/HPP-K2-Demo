import json

from utils import call_large_model_llm

EXCLUDED_SECTIONS = {
    "cardiovascular",
    "respiratory",
    "gastrointestinal",
    "neurological",
    "musculoskeletal",
    "psychiatric",
    "other_relevant_systems",
}


SYSTEM_MESSAGE = """
You are a medical assistant who is about to issue an examination order. Based on the examination information I provide, you will recommend a list of corresponding node_ids for typical symptoms and clinical manifestations that require attention.

You will receive two inputs:

node.json

The JSON contains the following fields, with these meanings:
label: Typical symptom or clinical manifestation. This will be matched against the patient’s reported symptoms.
unit: Measurement unit for the test result, usually the numeric unit returned by the examination (for reference only).
system: Related organ/system category. There are four categories:
    + CARDIO (cardiovascular)
    + NEURO (neurological)
    + RENAL (renal)
    + METABOLIC (metabolic)
node_id: This is what you must output. Only return node_ids, do not include any other text, and strictly follow the format below:

<answer>[node_id1, node_id2, node_id3]</answer>

For example:
<answer>["CARDIO:SBP_day", "CARDIO:SBP_night", "NEURO:CognitiveDecline", "RENAL:Proteinuria_gd", "RENAL:eGFR_slope"]</answer>

means that these five items should be ordered for examination.

Let’s begin.
""".strip()


def build_json(diagnosis_path: str) -> dict:
    """Load diagnosis.json, filter review_of_systems, and return user_json dict."""
    with open(diagnosis_path, "r", encoding="utf-8") as f:
        diagnosis_data = json.load(f)

    user_json = {}
    for key, value in diagnosis_data.items():
        if key == "review_of_systems" and isinstance(value, dict):
            filtered_systems = {
                system_name: system_data
                for system_name, system_data in value.items()
                if system_name not in EXCLUDED_SECTIONS
            }
            user_json[key] = filtered_systems
        else:
            user_json[key] = value

    return user_json


def build_messages(user_json: dict, node_json: dict) -> list[dict]:
    """Build chat messages for the LLM based on filtered user_json."""
    user_json_str = json.dumps(user_json, ensure_ascii=False, indent=2)
    node_json_str = json.dumps(node_json, ensure_ascii=False, indent=2)
    user_message = f"""
This will be the patient information I provide to you, composed of JSON fields.

Patient Information:

{user_json_str}

Node Information:

{node_json_str}

Please strictly follow the instructions I provide. Identify several potentially relevant fields from the symptoms, and return the corresponding list of node_ids.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]
    return messages


def run_examination_node_selection(
    diagnosis_path: str = "example/case1/diagnosis.json",
    node_path: str = "hpp_data/node.json",
):
    """
    Orchestrate:
    1) load & filter diagnosis JSON,
    2) build messages,
    3) call large model,
    4) print user_json and model response.
    """
    user_json = build_json(diagnosis_path)
    node_json = build_json(node_path)
    messages = build_messages(user_json=user_json, node_json=node_json)
    response = call_large_model_llm(messages)
    return response


if __name__ == "__main__":
    run_examination_node_selection()
    print("LLM Output with examination:")
    print(response)