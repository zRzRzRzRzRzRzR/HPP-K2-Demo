import json

from utils import build_json, call_large_model_llm

SYSTEM_MESSAGE = """
You are a medical assistant who is about to issue an examination order. Based on the examination information I provide, you will recommend a list of corresponding node_ids for typical symptoms and clinical manifestations that require attention.

You will receive two inputs:

## Patient Information

This JSON will contain the patient’s basic information.
patient_information: Describes details such as the patient’s age and sex, which can be used as decision factors (for example, cardiovascular or heart disease is more likely in middle-aged and elderly patients).
chief_complaint: Contains the patient’s description of their current condition.
review_of_systems: Usually includes only a general_health field, representing the patient’s overall health status.

## Node Information

The JSON contains the following fields, with these meanings:
label: Typical symptom or clinical manifestation. This will be matched against the patient’s reported symptoms.
unit: Measurement unit for the test result, usually the numeric unit returned by the examination (for reference only).
system: Related organ/system category. There are four categories:
    + CARDIO (cardiovascular)
    + NEURO (neurological)
    + RENAL (renal)
    + METABOLIC (metabolic)
node_id: This is what you must output, mean what you will order for examination.

These details will help you accurately identify which Nodes are needed for diagnosis. You should avoid:
1.	Outputting irrelevant types of nodes. For example, do not order localized cardiac examinations for a patient whose condition is more suggestive of a hematologic disease.
2.	You may select 5–10 examination items; aim for precision and relevance, and avoid excessive or unnecessary items.

## OUTPUT FORMAT

Only return node_ids, do not include any other text, and strictly follow the format below:
<answer>[node_id1, node_id2, node_id3]</answer>

For example:
<answer>["CARDIO:SBP_day", "CARDIO:SBP_night", "NEURO:CognitiveDecline", "RENAL:Proteinuria_gd", "RENAL:eGFR_slope"]</answer>
""".strip()


def build_messages(user_json: dict, node_json: dict) -> list[dict]:
    """Build chat messages for the LLM based on filtered user_json."""
    user_json_str = json.dumps(user_json, ensure_ascii=False, indent=2)
    node_json_str = json.dumps(node_json, ensure_ascii=False, indent=2)
    user_message = f"""
This will be the patient information I provide to you, composed of JSON fields.

## Patient Information

{user_json_str}

## Node Information

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
    user_json = build_json(diagnosis_path)
    node_json = build_json(node_path)
    messages = build_messages(user_json=user_json, node_json=node_json)
    response = call_large_model_llm(messages)
    return response