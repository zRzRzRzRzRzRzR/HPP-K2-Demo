import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path

load_dotenv()

SYSTEM_PROMPT = """ 
You are a precise medical information extraction model.

- Output only valid JSON that matches the provided json_schema.
- Do not include any field whose value is unknown, missing, or not explicitly supported by the images.
- Do not hallucinate, infer, or guess any values.
- Do not add new fields, and do not rename any existing key.
- Do not wrap the JSON in markdown, code fences, or explanations.
"""

USER_PROMPT = """
You are a clinical data extraction assistant.
You will be given one or more images of a patient intake/consultation form and clinical notes.
Read all images carefully and extract information into the specified JSON schema.
Only use information that is explicitly present in the images.
1. If a field has no explicit information, omit this field from the JSON output.
2. Keep a parent object only if at least one of its child fields is present.
3. Do not add any extra fields beyond the schema keys.
4. Do not change key names.
5. Copy dates exactly as shown.
6. If a severity score from 1 to 10 exists, output it as a string (e.g. \"7\"); otherwise omit that field.
"""


def empty_schema() -> Dict[str, Any]:
    return {
        "patient_information": {
            "name": "",
            "patient_id": "",
            "date_of_birth": "",
            "date_of_evaluation": "",
            "referring_physician": "",
            "insurance_information": "",
        },
        "chief_complaint": {
            "primary_complaint": "",
            "duration": "",
            "severity_1_to_10": "",
        },
        "medical_history": {
            "past_medical_history": "",
            "family_medical_history": "",
            "current_medications": "",
            "allergies": "",
            "previous_hospitalizations_surgeries": "",
        },
        "review_of_systems": {
            "general_health": "",
            "cardiovascular": "",
            "respiratory": "",
            "gastrointestinal": "",
            "neurological": "",
            "musculoskeletal": "",
            "psychiatric": "",
            "other_relevant_systems": "",
        },
    }


def pdf_to_images_base64(pdf_path: str, dpi: int = 200) -> List[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    images_b64: List[str] = []
    for page in pages:
        buf = BytesIO()
        page.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images_b64.append(b64)
    return images_b64


def get_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patient_information": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "patient_id": {"type": "string"},
                    "date_of_birth": {"type": "string"},
                    "date_of_evaluation": {"type": "string"},
                    "referring_physician": {"type": "string"},
                    "insurance_information": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "chief_complaint": {
                "type": "object",
                "properties": {
                    "primary_complaint": {"type": "string"},
                    "duration": {"type": "string"},
                    "severity_1_to_10": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "medical_history": {
                "type": "object",
                "properties": {
                    "past_medical_history": {"type": "string"},
                    "family_medical_history": {"type": "string"},
                    "current_medications": {"type": "string"},
                    "allergies": {"type": "string"},
                    "previous_hospitalizations_surgeries": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "review_of_systems": {
                "type": "object",
                "properties": {
                    "general_health": {"type": "string"},
                    "cardiovascular": {"type": "string"},
                    "respiratory": {"type": "string"},
                    "gastrointestinal": {"type": "string"},
                    "neurological": {"type": "string"},
                    "musculoskeletal": {"type": "string"},
                    "psychiatric": {"type": "string"},
                    "other_relevant_systems": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def extract_structured_data(images_b64: List[str]) -> Dict[str, Any]:
    client = OpenAI()
    schema = get_json_schema()
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": USER_PROMPT,
        }
    ]
    for b64 in images_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient_intake_extraction",
                "schema": schema,
            },
        },
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def pdf_to_json(pdf_path: str, output_json_path: str) -> Dict[str, Any]:
    images_b64 = pdf_to_images_base64(pdf_path)
    result = extract_structured_data(images_b64)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def convert_pdf_to_json(pdf_path: str) -> str:
    output_json_path = os.path.splitext(pdf_path)[0] + ".json"
    pdf_to_json(pdf_path, output_json_path)
    return output_json_path
