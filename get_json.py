import json
from utils import call_large_model_llm
with open('example/case1/diagnosis.json', 'r', encoding='utf-8') as f:
    diagnosis_data = json.load(f)

excluded_sections = {
    'cardiovascular',
    'respiratory',
    'gastrointestinal',
    'neurological',
    'musculoskeletal',
    'psychiatric',
    'other_relevant_systems'
}

user_json = {}
for key, value in diagnosis_data.items():
    if key == 'review_of_systems':
        filtered_systems = {}
        for system_name, system_data in value.items():
            if system_name not in excluded_sections:
                filtered_systems[system_name] = system_data
        user_json[key] = filtered_systems
    else:
        user_json[key] = value

user_json_str = json.dumps(user_json, ensure_ascii=False, indent=2)

system_message = """
你将会读取到一个JSON文件, 代表了一个, 你将要根据病人症状的描述来判定他可能会使用到哪些节点的问题

本JSON的信息如下, 其中每个字段的含义是
label:  典型症状和表现, 这将会和我输入的用户症状匹配
node_id: 这是你要输出的内容, 仅仅返回 node_id, 不要返回任何其他文字, 并按照如下格式返回

<answer>[node_id1, node_id2, node_id3]</answer>

比如:
<answer>["CARDIO:PWV", "NEURO:WMH_vol", "NEURO:CognitiveDecline", "RENAL:Proteinuria_gd", "RENAL:eGFR_slope"]</answer>

开始吧
"""
print(user_json_str)
breakpoint()
user_message = f"""
This will be the patient information I provide to you, composed of JSON fields.

{user_json_str}

Please strictly follow the instructions I provide. Identify several potentially relevant fields from the symptoms, and return the corresponding list of node_ids.
"""

message = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]
response = call_large_model_llm(message)
print(response)
