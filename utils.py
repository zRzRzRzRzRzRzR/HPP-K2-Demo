import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def call_embedding(texts, model="embedding-3"):
    if isinstance(texts, list):
        texts = [" " if text == "" else text for text in texts]

    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    embeddings = []
    for item in response.data:
        emb = np.array(item.embedding, dtype="float32")
        embeddings.append(emb)
    return embeddings


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def call_embeddings_batch(texts, model="embedding-3", batch_size=10):
    client = OpenAI()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            emb = np.array(item.embedding, dtype="float32")
            embeddings.append(emb)
    return embeddings


def build_json(diagnosis_path: str) -> dict:
    with open(diagnosis_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_response(content: str) -> str:
    if not content:
        return ""

    content = content.strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def call_large_model_llm(messages, api_key=None, base_url=None, model=None):
    model = model or os.getenv("K2_THINK_MODEL", "MBZUAI-IFM/K2-Think")
    base_url = base_url or os.getenv(
        "K2_THINK_API_URL", "https://llm-api.k2think.ai/v1/"
    )
    api_key = api_key or os.getenv("K2_THINK_API_KEY", "EMPTY")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=8192,
        stream=False,
    )

    raw_content = (response.choices[0].message.content or "").strip()
    return format_response(raw_content)
