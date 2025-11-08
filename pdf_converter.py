# pdf_converter.py

import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path

load_dotenv()


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
    """
    将 PDF 每一页转换为 PNG，并返回 base64 字符串列表（不带 data:image 前缀）。
    """
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
    """
    与 empty_schema 对应的 JSON Schema，用于强约束输出。
    """
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
                "required": [
                    "name",
                    "patient_id",
                    "date_of_birth",
                    "date_of_evaluation",
                    "referring_physician",
                    "insurance_information",
                ],
                "additionalProperties": False,
            },
            "chief_complaint": {
                "type": "object",
                "properties": {
                    "primary_complaint": {"type": "string"},
                    "duration": {"type": "string"},
                    "severity_1_to_10": {"type": "string"},
                },
                "required": [
                    "primary_complaint",
                    "duration",
                    "severity_1_to_10",
                ],
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
                "required": [
                    "past_medical_history",
                    "family_medical_history",
                    "current_medications",
                    "allergies",
                    "previous_hospitalizations_surgeries",
                ],
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
                "required": [
                    "general_health",
                    "cardiovascular",
                    "respiratory",
                    "gastrointestinal",
                    "neurological",
                    "musculoskeletal",
                    "psychiatric",
                    "other_relevant_systems",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "patient_information",
            "chief_complaint",
            "medical_history",
            "review_of_systems",
        ],
        "additionalProperties": False,
    }


def extract_structured_data_with_gpt5(images_b64: List[str]) -> Dict[str, Any]:
    if not images_b64:
        return empty_schema()

    client = OpenAI()
    schema = get_json_schema()

    user_content = [
        {
            "type": "text",
            "text": (
                "You are a clinical data extraction assistant. "
                "You will be given one or more images of a patient intake/consultation form "
                "and clinical notes. Read all images carefully and extract information into "
                "the specified JSON schema.\n"
                "Rules:\n"
                "- Only use information that is explicitly present in the images.\n"
                "- If a field is unknown, missing, or not clearly stated, use an empty string.\n"
                "- Do NOT add new fields.\n"
                "- Do NOT change key names.\n"
                "- Dates should be copied exactly as shown.\n"
                "- Severity_1_to_10 should be the numeric string if available, else empty.\n"
            ),
        }
    ]

    for b64 in images_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a precise medical information extraction model. "
                        "Output only valid JSON that matches the provided json_schema. "
                        "No explanations."
                    ),
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
                "strict": True,
            },
        },
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def pdf_to_json_via_gpt(pdf_path: str, output_json_path: str) -> Dict[str, Any]:
    print(f"Processing PDF (via screenshots + GPT-5): {pdf_path}")
    print("=" * 60)

    images_b64 = pdf_to_images_base64(pdf_path)
    print(f"Total pages converted to images: {len(images_b64)}")

    result = extract_structured_data_with_gpt5(images_b64)

    print("\nExtracted JSON Data:")
    print("-" * 30)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"JSON data saved to: {output_json_path}")

    return result


def convert_pdf_to_json(pdf_path: str) -> str:
    """
    对外统一接口：
    - 输入 PDF 路径
    - 在同目录生成 diagnosis.json
    - 返回 diagnosis.json 的路径（给后续检查节点选择用）
    """
    base_dir = os.path.dirname(pdf_path)
    output_json_path = os.path.join(base_dir, "diagnosis.json")
    pdf_to_json_via_gpt(pdf_path, output_json_path)
    return output_json_path


if __name__ == "__main__":
    pdf_path = "example/case1/diagnosis case1.pdf"
    convert_pdf_to_json(pdf_path)
