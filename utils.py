import json
import re
import numpy as np
from openai import OpenAI
import os

K2_THINK_MODEL = os.getenv("K2_THINK_MODE", "MBZUAI-IFM/K2-Think")
K2_THINK_API_URL = os.getenv("K2_THINK_API_URL", "https://llm-api.k2think.ai/v1/")
K2_THINK_API_KEY = os.getenv("K2_THINK_API_KEY", "EMPTY")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def call_large_model_llm(messages, api_key=None, base_url=None, model=None):
    api_key = api_key or K2_THINK_API_KEY
    base_url = base_url or K2_THINK_API_URL
    model = model or K2_THINK_MODEL

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            stream=False,
        )
        if response.choices[0].message.content.strip():
            content = response.choices[0].message.content.strip()
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return content

    except Exception as e:
        print(f"Error in call_large_model as {e}")
        return {}


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


def clean_response(response_str):
    response_str = response_str.strip()
    if response_str.startswith("```") and response_str.endswith("```"):
        lines = response_str.split("\n")
        lines = lines[1:-1]
        response_str = "\n".join(lines).strip()
    return response_str


def call_embeddings_batch(texts, model="embedding-3", batch_size=10):
    client = OpenAI()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            emb = np.array(item.embedding, dtype="float32")
            embeddings.append(emb)
    return embeddings
