import json

from utils import build_json, call_large_model_llm


def build_user_review_of_systems_json(diagnosis_path: str) -> dict:
    data = build_json(diagnosis_path)
    review = data.get("review_of_systems", {})
    return {
        "review_of_systems": review
    }

SYSTEM_MESSAGE = """
You are an assistant to an attending physician. You will receive a patient’s examination report and basic information wrapped in a JSON object.

## Patient Information:

You will receive one or more examination results in the following categories:

+ Cardiovascular
+ Respiratory
+ Gastrointestinal
+ Neurological
+ Musculoskeletal
+ Psychiatric

Next, you will receive a Node mapping table for the relevant clinical indicators.

## Node Information:

The JSON contains the following fields, with these meanings:
- label: Typical symptom or clinical manifestation. This will be matched against the patient’s reported symptoms.
- unit: Measurement unit for the test result, usually the numeric unit returned by the examination (for reference only).
- system: Related organ/system category. There are four categories:
    + CARDIO (cardiovascular)
    + NEURO (neurological)
    + RENAL (renal)
    + METABOLIC (metabolic)
- node_id: This is the identifier you must output for abnormal findings, along with the corresponding measured value.

## Task

Using your medical knowledge, first identify clearly abnormal values from the examination data. Then, based on the patient’s symptoms and findings, select the smallest necessary set of the most relevant node_id entries to represent these abnormalities.

## OUTPUT FORMAT

You must return a JSON object containing the abnormal node_ids and their corresponding abnormal values. Units must follow the standardized form associated with each node_id. 
Use the following format:
{
  "abnormal_points": [
    {"node_id": "NODE NAME", "value": "EXAM VALUE with ABNORMAL", "unit": "NODE UNIT"},
  ]
}

here is an example output:

{
  "abnormal_points": [
    {"node_id": "CARDIO:SBP_day", "value": 152, "unit": "mm[Hg]"},
    {"node_id": "CARDIO:SBP_night", "value": 138, "unit": "mm[Hg]"}
  ]
}
"""

USER_MESSAGE = """

"""


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


def run_abnormal_node_selection(
    diagnosis_path: str = "example/case1/diagnosis_exam.json",
    node_path: str = "hpp_data/node.json",
):
    user_json = build_user_review_of_systems_json(diagnosis_path)
    node_json = build_json(node_path)
    messages = build_messages(user_json=user_json, node_json=node_json)
    response = call_large_model_llm(messages)
    return response


if __name__ == "__main__":
    response = run_abnormal_node_selection()
    print("LLM Output with abnormal Node:")
    print(response)
