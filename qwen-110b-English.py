import pandas as pd
import json
import csv
import os
import argparse
from openai import OpenAI
import re
client = OpenAI(
    base_url='http://localhost:11434/v1', 
    api_key='ollama',
)
def model_predict(report):
    # Define the tags that need to be detected and their corresponding prompt templates
    prompt_templates = {
        "Atelectasis": """Your task is to identify and label the presence of Atelectasis in the given text reports.
Lung Fields:
Look for phrases in the report that describe the lung fields. If the report does not mention terms like "homogeneous opacity" or "increased density" in a specific lobe or segment, it may suggest that Atelectasis is not present. Note the absence of such terms and any other signs of Atelectasis in the lung fields.
Air-Bronchograms:
Check if the report mentions the presence or absence of air-bronchograms within the affected area. If air-bronchograms are not mentioned, it may indicate that Atelectasis is not present or that the atelectasis is complete. The absence of air-bronchograms can be a key indicator.
Lung Markings:
Evaluate the description of pulmonary vessels and bronchial markings in the report. If the report does not mention that the vessels appear crowded or that the bronchial markings are displaced or distorted, it may suggest that Atelectasis is not present. Note the absence of these changes in lung markings.
Diaphragm and Costophrenic Angles:
Observe the report's description of the diaphragm contour and costophrenic angles. If the report does not mention any abnormalities such as an elevated diaphragm on the affected side or blunting of the costophrenic angle, it may indicate that Atelectasis is not present. Note the absence of these abnormalities.
Heart and Mediastinum:
Note the report's comments on the heart size and shape, as well as any shifts in the mediastinum. If the report does not mention any mediastinal shift or other abnormalities related to the heart and mediastinum that are consistent with Atelectasis, it may suggest that Atelectasis is not present. Note the absence of these abnormalities.
Example:
Lung Fields: The report does not mention "homogeneous opacity" or "increased density" in any specific lobe or segment.
Air-Bronchograms: The report does not mention the presence or absence of air-bronchograms.
Lung Markings: The report does not mention crowded vessels or displaced/distorted bronchial markings.
Diaphragm and Costophrenic Angles: The report does not mention an elevated diaphragm or blunting of the costophrenic angle.
Heart and Mediastinum: The report does not mention any mediastinal shift or other abnormalities related to the heart and mediastinum.
""",
        "Cardiomegaly":"""Your task is to identify and label the presence of Cardiomegaly in the given text reports.
Look for phrases in the report that describe the heart size. Terms like "enlarged heart," "cardiomegaly," or "increased heart size" may suggest cardiomegaly, but it's important to note that an enlarged heart alone does not necessarily confirm cardiomegaly. Pay attention to the specific measurements or descriptions provided. Evaluate the heart shape. Descriptions such as "globular heart," "left ventricular enlargement," or "right ventricular enlargement" can offer clues about the type and location of potential cardiomegaly, yet they also need to be considered in the context of other findings.
Example:
Cardiac Contours:
Examine the report for descriptions of cardiac contours. Phrases like "bulging cardiac contours" or "prominent cardiac silhouette" may suggest cardiomegaly. Note any asymmetry or abnormal contours mentioned.
Heart-to-Thorax Ratio:
Check if the report mentions the heart-to-thorax ratio. A ratio greater than 0.5 (heart width to chest width) is often considered indicative of cardiomegaly. Note the specific ratio if provided.
Pulmonary Vascular Markings:
Evaluate the description of pulmonary vascular markings. In cardiomegaly, the report may mention "increased pulmonary vascular markings" or "pulmonary congestion," which can be associated with heart failure. Note the presence or absence of these findings.
Heart Size and Shape:
The report only mention' Cardiac size is somewhat enlarged' not indicate Cardiomegaly  
        """ ,
        "Consolidation": """
​​Criteria for Consolidation​​
​​Mandatory Terminology​​:
The report ​​must explicitly state "consolidation"​​ (e.g., "right lower lobe consolidation").
​​OR​​ describe ​​air-bronchograms​​ (e.g., "air-filled bronchi within the opacity").
​​OR​​ mention ​​obscured pulmonary vessels/distorted bronchial markings​​ within the opacity.
​​Exclusion of Indirect Terms​​:
Terms like "opacity," "pneumonia," "collapse," or "infiltrate" ​​do not imply consolidation​​ unless paired with the above criteria.
​​Final Report Priority​​:
Disregard preliminary notes (e.g., "WET READ") if the Final Report contradicts them.
​​Analysis Steps​​
​​Search for Direct Terminology​​:
Scan the entire report for words/phrases: "consolidation," "air-bronchogram," "obscured vessels," or "distorted bronchial markings."
​​Location and Air-Bronchograms​​:
If consolidation is mentioned, check if it specifies a ​​lobar/segmental location​​ (e.g., "left upper lobe consolidation").
Confirm if ​​air-bronchograms​​ are described within the opacity.
​​Vessel/Bronchial Markings​​:
Check if the report states that pulmonary vessels are ​​obscured​​ or bronchial structures are ​​distorted​​ in the affected area.
​​Final Report vs. Preliminary Notes​​:
Prioritize findings in the ​​Final Report​​ section. Ignore conflicting descriptions elsewhere.

​​Examples of Consolidation vs. Non-Consolidation​​
​​​Case 1:​Consolidation Present​​:
"PA view shows right middle lobe consolidation with air-bronchograms."
​​Keywords​​: "consolidation," "air-bronchograms."
​​​Case 2:​Consolidation Absent​​:
"Multifocal opacities consistent with pneumonia; no air-bronchograms seen."
​​Reason​​: "Pneumonia" ≠ consolidation; lacks required terminology.
​​Case 3: Consolidation Absent​​
WET READ: "Suspected consolidation in left lower lobe."  
FINAL REPORT: "Left lower lobe opacity likely due to atelectasis; no air-bronchograms."  
**Consolidation**: Absent  
**Rationale**: Final Report attributes opacity to atelectasis, no explicit consolidation criteria.  
​​Case 4: Consolidation Absent (Pneumonia ≠ Consolidation)​​
FINAL REPORT: "Multifocal pneumonia; no consolidation or air-bronchograms identified."  
**Consolidation**: Absent  
**Rationale**: "Pneumonia" mentioned, but lacks consolidation terminology.  
​​Case 5: Consolidation Absent (Exclusion by CT)​​
FINAL REPORT: "Opacity in right mid-lung raises concern for consolidation; subsequent CT showed no consolidation."  
**Consolidation**: Absent  
**Rationale**: CT explicitly excluded consolidation.  
​​Case 6: Consolidation Absent (WET READ vs. Final Report)​​
WET READ: "Consolidation noted on lateral view."  
FINAL REPORT: "Diffuse opacities consistent with pulmonary edema; no consolidation."  
**Consolidation**: Absent  
**Rationale**: Final Report overrides WET READ and confirms absence.  
​​Case 7: Consolidation Absent (Aspiration ≠ Consolidation)​​
FINAL REPORT: "New right lower lobe opacities worrisome for aspiration; no air-bronchograms."  
**Consolidation**: Absent  
**Rationale**: "Aspiration" mentioned, but no consolidation criteria met.  

Critical Notes​​
​​No Inferences​​: Do not assume consolidation from clinical context (e.g., "pneumonia" or "fever").
​​Strict Adherence​​: Absence of explicit terms = ​​"Absent"​​, even if pathology is suspected.
 """,
        "Edema": """Analyze the following radiology report to check if the condition 'Edema' is present.Your task is to identify and label the absence of Edema in the given text reports.Look for phrases in the report that describe the lung fields.Note the distribution and characteristics of the suspected edematous areas.Prominent pulmonary vessels may suggest increased pulmonary blood flow due to heart failure, a common cause of Edema. Thickened or blurred bronchial markings could be due to fluid in the bronchial walls or surrounding tissues.Observe the report's description of the diaphragm contour and costophrenic angles. Edema may be associated with a flattened diaphragm or blunted costophrenic angles. Blunted costophrenic angles may be due to the presence of pleural effusion, which can be a complication of Edema.
Example :
Lung Fields:
the report may mention terms like "decreased lung transparency" or "diffuse pulmonary opacities" "ground-glass appearance"
Lung Markings:
the report mentions "prominent pulmonary vessels", "thickened or blurred bronchial markings"
Diaphragm and Costophrenic Angles :
the report describes "flattened diaphragm", "blunted costophrenic angles"
Critical Notes:
No Assumptions :
Do not infer edema from indirect terms (e.g., "heart failure" ≠ edema).
Explicit Mention Required:
Only label edema if direct terminology (e.g., "edema", "fluid overload", "pulmonary edema") is used.
Prioritize Final Report:
Noted that if there is a discrepancy between the WET READ findings and the Final Report, the Final Report should be prioritized.

""",
        "Enlarged Cardiomediastinum": """
Evaluate if the report mentions any changes in the size or shape of the heart that may suggest enlargement.Check for descriptions of the mediastinum that may indicate widening or other abnormalities. If none of the Enlarged Cardiomediastinum features below are present, Enlarged Cardiomediastinum is Absent.
Example :
Heart Size and Shape:
the report mentions "cardiomegaly"、"enlarged heart".the report only mentions chronic or stable heart size, the label should be labeled as 0.
Mediastinum :
the report mentions "widened mediastinum .if the report only mentions a stable mediastinum, such as "stable mediastinal contours," Enlarged Cardiomediastinum should be labeled as 0.
Critical Notes :
No Assumptions:
Do not infer Enlarged Cardiomediastinum from indirect terms.
Explicit Mention Required:
Only label Enlarged Cardiomediastinum if direct terminology is used.
Prioritize Final Report:
If there is a discrepancy between the WET READ findings and the Final Report, the Final Report should be prioritized.
Exclude Chronic/Stable Findings:
If the report states that the heart size or mediastinum is chronic or stable, such as "chronic cardiomegaly" or "stable mediastinal contours," the report should be labeled as 0。 

""",
        "Fracture": """Analyze the following radiology report to check if the condition 'Fracture' is present.Your task is to identify and label the absence of fracture in the given text reports.
Look for descriptions of interrupted bone continuity.Check for post-traumatic bone reaction patterns .Only activate the analysis if the text involves the thoracic skeletal structures.
Critical Notes:
No Assumptions:
Do not infer a fracture from indirect terms.
Explicit Mention Required:
Only label a fracture if direct terminology is used.
Prioritize Final Report:
If there is a discrepancy between the WET READ findings and the Final Report, the Final Report should be prioritized.
        """,
        "Lung Lesion":  """Analyze the following radiology report to check if the condition 'Lung Lesion' is present.
All provided cases are classified as 0 (No Lung Lesion) due to descriptions of atelectasis, effusion, cardiac/catheter-related changes, chronic findings, or indeterminate terminology.


Lung Lesion = 1 if the report includes following terms :
Cavitation (e.g., "cavity," "cavitation," "possible cavitation," "suggestion of cavitation").
Nodule/Mass (e.g., "nodule," "nodular opacity," "mass," "nodular density").
Fibrosis/Bronchiectasis (e.g., "bronchiectasis," "fibrosis," "honeycombing").


Lung Lesion = 0 for ALL other findings, including:
Non-Structural Infections (e.g., "pneumonia," "consolidation," "opacities" without cavitation/nodule).
Mechanical/Functional Changes (e.g., atelectasis, pleural effusion, pulmonary edema).
Indeterminate Language (e.g., "possible pneumonia," "cannot exclude") without structural terms.
Chronic/stable pathologies (e.g., COPD, old nodules).
Non-specific findings requiring further evaluation (e.g., undiagnosed infiltrates).


""",
        "No Finding": None,  # The prompt for 'No Finding' can be handled separately.
        "Pleural Effusion": """Analyze the following radiology report to check if the condition 'Pleural Effusion' is present.Your task is to identify and label the absence of Pleural Effusion in the given text reports. Consider the following aspects.Check if the report mentions the location of the effusion.Look for descriptions of the size of the effusion.Identify any specific characteristics of the effusion and consider any associated findings mentioned in the report.
Example:
Location :
"Right pleural effusion," "Left pleural effusion," "Bilateral effusion"
Size :
"Small effusion," "Minimal effusion," "Large effusion"
Characteristics:
"Loculated pleural effusion," "Layering effusion," "Blunting of costophrenic angle"
Associated Findings:
"Respiratory distress," "Chest pain," "Pneumonia," "Atelectasis"
Critical Notes:
No Assumptions:
Do not infer pleural effusion from indirect terms.
Explicit Mention Required:
Only mark Pleural Effusion as 1 if the report explicitly mentions pleural effusion and its characteristics.
Prioritize Final Repor :
If there is a discrepancy between the WET READ findings and the Final Report, the Final Report should be prioritized.

""",
        "Pleural Other": """"Your task is to identify and label the absence of Pleural Other in the given text reports.A
Explicit terms only​​ (e.g., pleural thickening, pneumothorax, pleural mass, loculation, calcification, fibrosis, empyema).
​​Exclude​​ if only pleural effusion is mentioned without additional pleural pathology.
Example:
Abnormalities:
"Pleural thickening," "Loculations," "Pneumothorax," "Atelectasis"
Location :
"Right pleural thickening," "Left pleural," "Bilateral pleural""bilateral loculated effusions."
Characteristics:
"Size," "Loculated," "Blunting of costophrenic angle"
Changes:
"increased thickening," "decreased effusion."
Clinical Context:
"Respiratory distress," "Chest pain," "History of pneumonia"
Associated Findings:
"Cardiomegaly," "Interstitial edema," "Pulmonary nodules"
Critical Notes:
Exclude pleural effusion​:
​​Pleural Other​​: Defined as ​​any pleural abnormality other than pleural effusion​​. pleural effusion is labeled 0
​​Exclude​​ if only pleural effusion is mentioned without additional pleural pathology.
No Assumptions:
Do not infer pleural abnormalities from indirect terms.
Explicit Mention Required:
Mark ‘Pleural Other’ as 1 only if the report explicitly mentions pleural abnormalities and their characteristics.
Prioritize Final Report:
If there is a discrepancy between the WET READ findings and the Final Report, prioritize the Final Report.
Case-by-Case Judgment​​:

​​Case 1 & 2​​:
​​Explicit Terms​​: "Small newly appeared left pleural effusion."
​​Pleural Other​​: False 
​​Case 3​​:
​​Explicit Terms​​: "Moderate bilateral pleural effusions."
​​Pleural Other​​: False 
​​Case 4​​:
​​Explicit Terms​​: "Previous left pleural effusion or pleural thickening has resolved."
​​Pleural Other​​: False 
​​Case 5​​:
​​Explicit Terms​​: "Moderate left pleural effusion has increased."
​​Pleural Other​​: False 
​​Case 6​​:
​​Explicit Terms​​: "Small bilateral pleural effusions are persistent."
​​Pleural Other​​: False 
​​Case 9 & 10​​:
​​Explicit Terms​​: "Increase in bilateral pleural effusions."
​​Pleural Other​​: False 
​​Case 11​​:
​​Explicit Terms​​: "Small right pleural effusion has decreased."
​​Pleural Other​​: False 
​​Case 13​​:
​​Explicit Terms​​: "Small right pleural effusion is new."
​​Pleural Other​​: False 
​​Case 14​​:
​​Explicit Terms​​: "Small pleural effusions bilaterally."
​​Pleural Other​​: False 
​​Case 15​​:
​​Explicit Terms​​: "Trace pleural effusion suspected on the right."
​​Pleural Other​​: False 
​​Case 16​​:
​​Explicit Terms​​: "Blunting of the left costophrenic angle" (effusion only).
​​Pleural Other​​: False 
​​Case 17 & 18​​:
​​Explicit Terms​​: "Small left pleural effusion has increased."
​​Pleural Other​​: False 
​​Case 19​​:
​​Explicit Terms​​: "Large right-sided pleural effusion."
​​Pleural Other​​: False 
​Case 20​:
​IMPRESSION​​:
"The pigtail catheter is again noted in the right base. ​​There is no pneumothorax​​... Persistent patchy density in the right upper lobe and in both lung bases."
​​Analysis​​:
​​Explicit Terms​​:
​​No pneumothorax​​ (excluded).
​​Pigtail catheter​​ (medical device, not pathology).
​​Patchy density​​ (parenchymal, not pleural).
​​Pleural Other​​:  ​​0​​
​​Case 21:​​
​​Report​​:
​​FINDINGS​​:
"Opacity at the left costophrenic angle... ​​may be due to overlying soft tissue​​... ​​No large pleural effusion or pneumothorax​​."
​​IMPRESSION​​:
"Posterior, inferior opacity... ​​may relate to overlying soft tissue​​."
​​Analysis​​:
​​Explicit Terms​​:
​​No large pleural effusion or pneumothorax​​ (excluded).
​​Opacity attributed to soft tissue​​ (not pleural).
​​Pleural Other​​:  ​​0​​


"""

,
        "Lung Opacity": """Analyze the following radiology report to check if the condition 'Lung Opacity' is present.please must follow these steps:

1. Explicit Mention Check: Look for explicit mentions of 'Lung Opacity' or related terms such as 'opacity', 'opacification', 'patchy opacities', 'confluent opacities', 'ground-glass opacities', etc., in the report. Only consider terms that directly describe lung opacity.
​​Label as 0​​：
Chronic/scarring ("chronic consolidation," "scarring"), stable findings ("no change"), or non-consolidative pathologies (atelectasis without consolidation, edema).
Chronic/stable findings ("scarring," "interstitial disease").
Isolated atelectasis ("atelectasis" without consolidation).
Space-occupying lesions ("mass," "nodule," "metastasis").
Technical artifacts ("underpenetration," "limited evaluation").
Pulmonary edema or pleural effusion (unless directly causing opacity).

2. Location and Characteristics: Note the location (right, left, bilateral) and characteristics (patchy, confluent, ground-glass) of any mentioned lung opacity.
3. Associated Findings: Consider associated findings such as cardiomegaly, interstitial edema, and pleural effusion, but do not infer lung opacity from these findings alone.
4. Clinical Context: Review the clinical context provided in the report to understand the patient's condition, but do not assume lung opacity based on clinical context alone.
5. Differentiation: Differentiate between lung opacity due to effusion and other potential causes. If the report explicitly states that opacification is due to a cause other than effusion (e.g., atelectasis, pneumonia), consider this in your judgment.
6. Impression Sections: Prioritize the impression section for conclusive statements about lung conditions. If the impression section explicitly mentions lung opacity, use this as the primary basis for your judgment.
7. No Assumptions: Do not make assumptions or inferences about lung opacity from indirect terms or lack of information. Only mark Lung Opacity as 1 if the report explicitly mentions it and its characteristics.
8. Final Judgment: Based on the above steps, determine if 'Lung Opacity' is present. Output 1 if present, 0 if not present.



""",
        "Pneumonia": """Label "Pneumonia" as '1' ONLY IF ALL criteria are definitively met.  
**Otherwise, label "Pneumonia=0"**. Follow this decision tree:  

1. **Imaging Evidence** (ALL Required):  
   - Consolidation: Must show "air bronchograms" or "lobar distribution".  
   - Pleural Effusion: Exudative (e.g., "loculated", "parapneumonic") AND adjacent to consolidation.  

2. *Clinical/Lab Evidence (ALL Required):  
   - Fever >38.5°C + WBC >15×10⁹/L + CRP >100 mg/L.  
   - **Microbiological Proof**: Positive culture (sputum/blood) or antigen (e.g., Legionella).  

3. Exclusion of Alternatives (ALL Required):  
   - No signs of atelectasis (e.g., "linear opacities", "volume loss").  
   - No heart failure (BNP <100 pg/mL) or fluid overload.  
   - No aspiration risk (e.g., alcoholism, dysphagia).  

If ANY of the following are true, label "0" immediately:  
- Terms like "consider", "cannot exclude", "possible", or "worrisome for".  
- Effusion is "transudative", "small", or unrelated to consolidation.  
- Clinical data (fever/WBC/CRP) missing or inconclusive.  
- Alternative explanations (atelectasis/edema) are mentioned.  

""",
        "Pneumothorax": """Analyze the following radiology report to check if the condition 'Pneumothorax' is present.Your task is to identify and label the absence of 'Pneumothorax in the given text reports. 
Consider all relevant information provided in the text such as imaging findings, descriptions of lung fields, pleural spaces, and any specific mention of air in the pleural cavity or related abnormalities that could indicate pneumothorax.
Example:
Direct Mentions:
"Pneumothorax""Suspected pneumothorax""Possible pneumothorax"
Radiological Descriptions:
"Air in the pleural cavity""Hyperlucent area""Lack of vascular markings" (in the affected area)
Size Descriptors:
"Small pneumothorax""Moderate pneumothorax""Large pneumothorax"
Associated Findings:
"Collapsed lung""Partial lung collapse""Atelectasis"

""",
        "Support Devices":  """Analyze the following radiology report to check if the condition 'Support Devices' is present.Your task is to identify and label the absence of Support Devices in the given text reports.
Scan the text for mentions of devices. Note the specific type of device and its location.If a device is mentioned but not clearly identified, note it as "unspecified support device."
Example:
mentions of devices:
“catheters”,” pacemakers”,”feeding tubes”, “drains”, “Cholecystectomy clips”, “Surgical clips Vertebral fixation hardware”  ”Anchor screws” “Cervical spinal hardware” “Median sternotomy wires” “Mitral valve replacement” “Endotracheal tube” •  "Cholecystectomy clips are noted in the right upper quadrant.""Dobhoff tube tip is in the stomach.""Right internal jugular line terminates at the level of superior SVC.""A dual lead ICD/pacemaker is in stable position."“Stent in the left brachiocephalic vein”or any other medical equipment.
location:
"right internal jugular line," "dual lead ICD/pacemaker"

 """,
        "emphysema": """Analyze the following radiology report to check if the condition 'emphysema' is present.Your task is to identify and label the absence of emphysema in the given text reports.
These reports typically describe findings consistent with emphysema, which may include direct mentions of emphysema or descriptions of radiological features suggestive of the condition. 
Example:
“direct mention of" emphysema”,“chronic obstructive pulmonary disease (COPD)”,or ”hyperinflated lungs”,”descriptions of increased lung volumes”,”flattened diaphragms”,”reduced vascular markings”,”air trapping", or "bullae"

""",
        "interstitial lung disease": """Analyze the following radiology report to check if the condition 'interstitial_lung_disease' is present.Your task is to identify and label the absence of interstitial_lung_disease in the given text reports.Pay close attention to the following key phrases and findings that may indicate interstitial lung disease,
Example:
'Interstitial edema''Interstitial abnormality”,”Interstitial pulmonary edema”,”Chronic interstitial lung disease”,”Increased interstitial markings”,”Focal opacities in the lung bases” (which could be related to atelectasis or infection in the context of interstitial lung disease)

""",
        "calcification(lung and mediastinal)": """Your task is to identify and label the absence of calcification(lung and mediastinal)
 in the given text reports.For each report, check for mentions of calcification in the lungs or mediastinum If calcification (lung or mediastinal) is present in the report, assign a label of 1. If no calcification is mentioned, assign a label of 0. If the meaning of the Key textual features is similar, it is also labeled to be 1
Example:
Lung calcification:
“calcified nodules”, “calcified granulomas”,“calcified plaques in the lungs”
Mediastinal calcification:
“aortic knuckle calcified aorta”,“calcified coronary vessels”,“calcified aortic knob”,“calcified plaques in the mediastinum or diaphragm”

""",
        "Trachea and bronchus": """​
Carefully review the following chest X-ray reports and determine if there are any findings related to tracheobronchial lesions. Specifically, look for mentions of bronchiectasis, cavitation, or any other abnormalities involving the trachea or bronchus. Exclude findings related to endotracheal tube placement or other non-lesional findings. 
Label as "0" if ​​any​​ of the following apply:
​​No Direct Airway Involvement​​:
Example: "Cavitation ​​in the lung mass​​" → Lesion is parenchymal, not tracheobronchial.
​​Iatrogenic/External Causes Dominant​​:
Example: "Right main stem bronchus intubation" → Device malposition, not a structural lesion.
​​Compressive/Reversible Changes​​:
Example: "Large pleural effusion with compressive atelectasis" → External compression, no airway damage.

"Nodule at right apex. Suspected emphysema. No acute disease."
​​Label​​: ​​0​​ (parenchymal nodule/emphysema; no airway involvement).
"Mild rightward tracheal shift due to enlarged thyroid. No consolidation."
​​Label​​: ​​0​​ (tracheal displacement from external compression; no airway lesion).
Report 1​​
​​Key Excerpt​​:
"No areas of airspace consolidation. No suspicious pulmonary nodules. No pleural effusions."
​​Analysis​​:
No mention of tracheobronchial structural abnormalities.
​​Label​​: 0
Report 2​​
​​Key Excerpt​​:
"New right lower lobe opacities worrisome for aspiration. Left lower lobe opacities due to atelectasis/aspiration."
​​Analysis​​:
Aspiration/atelectasis are ​​non-structural​​ findings.
​​Label​​: 0
Report 3 (Trauma Patient)​​
​​Key Excerpt​​:
"Right main stem bronchus intubation. Subtle pneumomediastinum."
​​Analysis​​:
ET tube malposition and pneumomediastinum are ​​iatrogenic/non-lesional​​.
​​Label​​: 0
Report 4 (Pleural Effusion)​​
​​Key Excerpt​​:
"Large right pleural effusion. Left lung clear. No pulmonary edema."
​​Analysis​​:
Effusion causes compressive atelectasis but ​​no airway lesions​​.
​​Label​​: 0
Report 5 (Respiratory Failure)​​
​​Key Excerpt​​:
"Progression of right perihilar opacities (pulmonary edema/infection). ET tube terminates 4.5 cm above carina."
​​Analysis​​:
Edema/infection are ​​non-structural​​. ET tube is properly positioned.
​​Label​​: 0
Report 6 (Emphysema)​​
​​Key Excerpt​​:
"Nodular focus at right apex (7 mm). Emphysema and apical scarring."
​​Analysis​​:
Nodule and emphysema are ​​non-airway lesions​​.
​​Label​​: 0
​Report 7 (Head Trauma)​​
​​Key Excerpt​​:
"Mild pneumopericardium/pneumomediastinum. Hypoinflated lungs."
​​Analysis​​:
Air leaks and hypoinflation are ​​non-structural​​.
​​Label​​: 0
Report 8 (Lung Cancer with Pneumonia)​​
​​Key Excerpt​​:
"Left pleural effusion and consolidation. Mild pulmonary edema."
​​Analysis​​:
Consolidation and edema are ​​non-airway pathologies​​.
​​Label​​: 0
""",
        "cavity and cyst": """
Rules:  
1. Focus on keywords: "cavity", "cavitation", "cyst".  
2. "Suggestion of cavitation" counts as `1` (probable).  
3. Ignore unrelated terms (e.g., "bronchiectasis", "effusion").  

Examples:  
"suggestion of cavitation" → `1`  
"air-fluid level in cavitary lesion" → `1`  
"cystic changes" → `1`  
""",
        "mediastinal other": """Analyze the provided chest radiograph reports ​​strictly​​ to determine if "mediastinal other" abnormalities are present. Output ​​1​​ only if the report explicitly mentions ​​ANY​​ of the following:

​​Widened mediastinum​​ (stable, unchanged, or new)
​​Pneumomediastinum​​ (subtle or definite)
​​Postsurgical mediastinal prominence​​
​​Mediastinal shift​​ (e.g., due to scarring, lobectomy)
​​Mediastinal mass, vascular congestion, or thyroid-related mediastinal widening​​

​​Exclude​​ cases where:

- Cardiomegaly, pleural effusion, pneumothorax, or atelectasis **without direct mediastinal involvement**.  
- Anatomic variants (e.g., "mediastinal fat", "tortuous aorta").  
- Adjacent organ changes (e.g., hiatal hernia, tracheal deviation from thyroid).  
- Ambiguous terms (e.g., "mediastinal contours unremarkable").  

​​Instructions:​​
Parse each report line by line.
Flag ​​1​​ ​​only​​ if the criteria above are met.
Flag ​​0​​ for ​​all other cases​​, even if uncertain.
​​Never assume abnormalities not explicitly stated.​​
​​Example Output:​​

​​Label 1 (widened mediastinum)
​​Label 1 (new pneumomediastinum)
​​Label 0 (no mediastinal terms)
​​Label 1​​ – "Hiatal hernia" altering mediastinal contour.
​​Label 1​​ – "Indentation from enlarged thyroid" → Mediastinal structural abnormality.
​​Label 1​​ – "Left mediastinal contour thickening" → Direct mediastinal involvement.

""",
        "pulmonary vascular abnormal": """Strictly evaluate chest X-ray reports for **pulmonary vascular abnormalities** and output **1** or **0** based on the rules below:  

Criteria for Output 1  
The report must include **at least one** of these **exact terms or descriptions**:  
1. Direct vascular findings:  
   - "pulmonary vascular engorgement"  
   - "enlarged/prominent/dilated pulmonary arteries"  
   - "pulmonary vascular congestion"  
   - "elevated pulmonary venous pressure"  
   - "vascular redistribution"  
   - "central pulmonary vascularity"  
   - "prominent azygos vein"  

2. Indirect vascular findings:  
   - Only these phrases** for cardiac/volume overload:  
     - "volume overload"  
     - "cardiac decompensation"  
     - "biventricular decompensation"  
     - "cardiogenic pulmonary edema"  
   - Active vascular changes (e.g., "vascular congestion has worsened", **not** "resolved" or "improved").  

Criteria for Output 0
Output **0** if:  
1. Uses **synonyms or non-listed terms** (e.g., "fluid overload", "vascular crowding", "perihilar prominence").  
2. Mentions "pulmonary edema" **without** "cardiogenic" or explicit cardiac link.  
3. Contains resolved/improved abnormalities (e.g., "congestion has resolved").  
4. Focuses on non-vascular pathologies (e.g., "atelectasis", "pleural effusion").  
"""

    }


    data_dict = {key: 0 for key in prompt_templates.keys()}  # Initialize all labels to 0.
    content = {}  # Used to store the response content for each labels

    for key in prompt_templates:
        if key != "No Finding": 
            prompt_template = prompt_templates[key]
            prompt= f"""Medical Imaging Report Analysis Task:
[Target Condition] {key}
[Report Content] {report}
{prompt_template.format(report=report)}
Please perform the following:
1. Analyze whether {key} is explicitly mentioned
2. Consider all relevant observations and terminology
3. Return 1 if confirmed or highly suspected (≥80% probability)
4. Return 0 if excluded, not mentioned, or low probability 

Response Format:
Single numeric character (0/1) with no punctuation or explanations."""

            response = client.chat.completions.create(
                model="qwen:110b",
                messages=[{'role': 'system', 'content': 'You are a radiologist analyzing medical reports.'},
                          {'role': 'user', 'content': prompt}]
            )


            # Extract the output content from the model.
            detection_result = response.choices[0].message.content.strip()
            content[key] = detection_result  # Save the response content for each label.

            match = re.search(r'(1|0)', detection_result)
            if match:
                data_dict[key] = int(match.group(1))
            else:
                data_dict[key] = 0
#check the value of 'No Finding'.
    if all(value == 0 for key, value in data_dict.items() if key != "No Finding"):
        data_dict["No Finding"] = 1
    else:
        data_dict["No Finding"] = 0
    return data_dict, content

def main(start_index=0):
    csv_file = './data/tb_mimic_cxr_metadata.csv'
    df = pd.read_csv(csv_file)

    fields = ['study_id', 'id', 'content', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'No Finding',
              'Pleural Effusion', 'Pleural Other', 'Lung Opacity', 'Pneumonia', 'Pneumothorax',
              'Support Devices', 'emphysema', 'interstitial lung disease', 'calcification(lung and mediastinal)',
              'Trachea and bronchus', 'cavity and cyst', 'mediastinal other',
              'pulmonary vascular abnormal']

    output_file = 'qwen.csv'
    status_file = 'last_processed_index.txt'

    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0

    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        try:
            study_id = row['study_id']
            report = row['report']
            id = row['id']

            prediction, content = model_predict(report)

            with open(output_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                row_data = {
                    'study_id': study_id,
                    'id': id,
                    'content': json.dumps(content),
                }
                for field in fields[3:]:
                    row_data[field] = prediction.get(field, 0)
                writer.writerow(row_data)

            # Update the processing progress.
            with open(status_file, 'w') as f:
                f.write(str(i))

            print(f"Processed row {i}/{len(df)-1} - Study ID: {study_id}")

        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")
            with open('errors.log', 'a') as f:
                f.write(f"Row {i} error: {str(e)}\n")

if __name__ == '__main__':
    # Set the command-line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0,
                       help='Start processing from this index (0-based)')
    args = parser.parse_args()

    # Attempt to resume from the last progress
    status_file = 'last_processed_index.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            try:
                last_index = int(f.read().strip())
                resume_index = last_index + 1
                print(f"Resuming from previous progress at index {resume_index}")
                args.start = resume_index
            except:
                pass

    # Run the  program.
    main(start_index=args.start)
