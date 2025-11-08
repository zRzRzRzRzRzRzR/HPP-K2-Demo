import json

from utils import build_json, call_large_model_llm

SYSTEM_MESSAGE = """
你是一个经验丰富的医疗诊断助理，熟悉HPP（高血压预测平台）中的各种检查节点（Nodes）。
现在，我已经得到了一份根据用户病情，血样指标得到的JSON格式的诊断结果。
你的目标是将要这份JSON结果，以及用户的病情情况，生成一份简单的Therapy Record。


## Patient Information

This JSON will contain the patient’s basic information.
patient_information: Describes details such as the patient’s age and sex, which can be used as decision factors (for example, cardiovascular or heart disease is more likely in middle-aged and elderly patients).
chief_complaint: Contains the patient’s description of their current condition.
review_of_systems: Usually includes only a general_health field, representing the patient’s overall health status.

## Drug Information

Based on professional prescription recommendations provided by physicians and the HPP system, the proposed regimen clearly specifies medication dosages, as well as the impact on various clinical indicators and expected outcomes.

## OUTPUT FORMAT

You need to generate a detailed treatment record based on this information, including:

1.	Basic patient information: name, sex, age.
2.	Reason for admission and symptom description: including the patient’s chief complaints.
3.	Physical findings and investigation results: including blood pressure, heart rate, body mass index, laboratory findings, etc.
4.	Initial impression: provide preliminary diagnostic impressions based on medical history and examination results.
5.	Treatment course and related events: describe in detail the interventions during treatment, medication adjustments, and changes in the patient’s condition.
6.	Discharge status and recommendations: including physical signs and test results at discharge, medication regimen at discharge, as well as post-discharge lifestyle and medication recommendations.

A very specific output example will be provided below. Please strictly follow that format when generating the treatment record.

## EXAMPLE OUTPUT

Ayana, female, 79 years old, was admitted for tiredness, breathlessness when walking on level ground, and forgetfulness. 
Over the last six months family members noticed she misplaces items and repeats questions; 
she walks more slowly and rests after short distances. 
Home readings are “often high,” and she prefers salty soups and preserved foods. 
Past history includes hypertension for more than twenty years and chronic kidney disease.
No chest pain at rest; no seizures. Examination on admission: blood pressure 156/84 mm Hg, pulse 72 bpm, afebrile. 
Body mass index 27.5 kg/m². Waist-to-hip ratio 0.90 . 
Ambulatory monitoring showed daytime systolic BP 155 mm Hg and nighttime systolic BP 142 mm Hg; calculated mean arterial pressure 107 mm Hg. Central hemodynamics recorded central systolic pressure 139 mm Hg and augmentation pressure 13 mm Hg. 
Vascular tests: pulse-wave velocity 12.4 m/s; brachial flow-mediated dilation 5 %; reactive hyperemia index 1.6 a.u. Echocardiography showed left-ventricular mass index 110 g/m² with preserved ejection fraction 55 %. Blood biomarkers: B-type natriuretic peptide 180 pg/mL; high-sensitivity C-reactive protein 3.0 mg/L. 
Kidney-mineral profile: eGFR 45 mL/min/1.73 m², phosphate 5.0 mg/dL, parathyroid hormone 85 pg/mL, FGF-23 170 pg/mL. 
Brain MRI revealed white-matter hyperintensity volume 13 mL. Coronary CT calcium score 480 Agatston. Mood and sleep screens: PHQ-9 = 10, daytime sleepiness score = 11. Diet recall estimated sodium intake 5.0 g/day and potassium intake 2.5 g/day. 

Initial impression: long-standing hypertension with arterial stiffness and concentric remodeling, stage-3 chronic kidney disease with mineral-bone disorder, severe coronary calcification, and cerebral small-vessel disease. 
Therapeutic course and related events: The team first reinforced sodium restriction with dietitian counseling and adjusted meals to increase potassium-rich foods within renal safety. Targets were dietary sodium 3.0 g/day and dietary potassium ~3.0 g/day. 
After seven days of inpatient diet control, daytime SBP decreased to 149 mm Hg and MAP to 103 mm Hg. She reported mild morning light-headedness lasting seconds; orthostatic vitals were acceptable, and fluid timing with meals resolved symptoms. 
Given persistent hypertension, ACE inhibitor was uptitrated (perindopril from 4 mg to 8 mg daily; exposure recorded as ). Two weeks later, daytime SBP was 145 mm Hg with central systolic pressure 134 mm Hg. Cough did not occur. BNP fell modestly to 165 pg/mL and hsCRP to 2.6 mg/L. 
A transient creatinine rise accompanied by eGFR 43 mL/min/1.73 m² was observed and stabilized on repeat testing without dose change. Because ambulatory pressure and stiffness remained above targets, a low dose thiazide-type diuretic was added (exposure recorded as ). 
Over the next ten days, daytime SBP reached 138 mm Hg, MAP 100 mm Hg, and pulse-wave velocity 11.8 m/s. The patient developed nocturia and mild leg cramps at night; shifting the dose to the morning and emphasizing hydration reduced symptoms while keeping numbers stable. 
For renal protection, her SGLT2 inhibitor was continued (exposure ). After four weeks, urinary protein monitoring showed improvement with Proteinuria 0.6 g/day and eGFR returned to 45 mL/min/1.73 m², consistent with a benign functional dip at initiation. 
A short episode of perineal itching occurred and resolved with topical therapy and hygiene counseling. 
For vascular risk, the statin was maintained (exposure ). Given the high calcium score and persistent stiffness, mineral-bone measures were addressed with diet and vitamin D per nephrology guidance, aiming to bring phosphate toward 4.5 mg/dL and PTH toward 70 pg/mL while tracking. 
Three-month inpatient-outpatient transition data recorded PWV 11.5 m/s, LVMI 105 g/m², BNP 155 pg/mL, and stable LVEF 55 %. 
Mood and sleep guidance reduced PHQ-9 to 8 and daytime sleepiness to 9. No syncope, no acute neurologic events. Discharge summary: symptoms improved; walking distance increased on ward testing. 

Final inpatient measurements were daytime SBP 138 mm Hg, nighttime SBP 135 mm Hg, MAP 100 mm Hg, cSBP 132 mm Hg, PWV 11.5 m/s, FMD 6 %, RHI 1.7 a.u., LVMI 105 g/m², LVEF 55 %, BNP 155 pg/mL, eGFR 45 mL/min/1.73 m², phosphate 4.8 mg/dL, PTH 78 pg/mL, FGF-23 165 pg/mL, WMH volume unchanged at 13 mL, and coronary calcium unchanged on prior baseline. 

Medications on discharge: ACE inhibitor 8 mg daily, thiazide-type diuretic low dose, SGLT2 inhibitor 10 mg daily, statin at prior dose, diet with sodium target 3.0 g/day and potassium ~3.0 g/day as tolerated. 
Education focused on slow posture change to avoid light-headedness, morning diuretic dosing to reduce nocturia, skin hygiene to prevent genital irritation, and consistent meal timing.
"""

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

Please strictly follow the instructions I provide.
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


if __name__ == "__main__":
    response = run_examination_node_selection()
    print("LLM Output with examination:")
    print(response)
