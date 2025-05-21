#  HMC-EvalRRG
__《Boosting Clinical Value in Radiology Report Generation: An new Evaluation Framework by Human-Machine cooperation》__  
# Introduction： 
We hypothesize that a Human-Machine Collaborative Framework integrating hierarchical clinical validity assessment and dynamic radiologist reasoning will achieve ≥95% diagnostic accuracy in AI-generated reports, surpassing conventional NLP benchmarks.To address these gaps, we propose a Human-Machine framework with two key innovations: (1) A structured evaluation framework that quantifies clinical validity through hierarchical assessment of entity matching, negation recognition, and diagnostic logic consistency, building on knowledge-enhanced architectures. (2) An iterative Human-Machin cooperation optimization process inspired by dynamic prompting techniques, integrating radiologists’ reasoning into prompt engineering, enabling systematic refinement of AI models to achieve ≥95% diagnostic accuracy (validated against expert annotations), this approach addresses terminology inconsistencies and data scarcity through knowledge-aware training. This study aims to validate the superiority of hierarchical clinical validity metrics over conventional NLP benchmarks and demonstrate non-inferiority of AI-generated reports to board-certified radiologists through radiologist-guided optimization. 
## Feature:
1.Supports prediction of over 21 types of chest disease tags, such as atelectasis, cardiomegaly, pneumonia, etc.  
2.Deployed using the Ollama local large language model.  
3.Includes features for resuming from breakpoints and automatic error logging.  
4.Outputs results in CSV format 
  
![image](https://github.com/user-attachments/assets/f5dd976f-f4dc-47d5-9211-b51c3d44ce8a)

#  Getting Started:  
## Installation
### 1. Prepare the code and the environment
    cd HMC-EvalRRG
    pip install -r requirements.txt
### 2. Prepare the training dataset
We use the [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr/2.1.0/).To handle Chinese reports, add encoding='gbk' when specifying the input report path.After downloading the data, place it in the ./data folder.
You can prepare medical report data in CSV format.  
The file should include the following fields:  
  study_id: Study identifier  
  id: Case ID  
  report: Imaging report text  
## Example Structure:  
```csv
study_id,id,report
1001,1,"PA view shows right middle lobe consolidation..."
```

# Configuration Instructions:  
Modify the parameters of the model_predict() function:  
```python
client = OpenAI(
    base_url='http://localhost:11434/v1',  
    api_key='ollama',                      
    timeout=30                             
)
```

# Output Results：  
The output file qwen.csv includes:  
  Original report identifier  
  Model response in JSON format  
  Prediction results for each pathological label (0/1)  
## Example:  
```csv
study_id,id,content,Atelectasis,Cardiomegaly,...
1001,1,"{'Atelectasis': '0',...}",0,1,...,0
```

# Acknowledgements  
  Ollama for providing local model services  
  MIMIC-CXR dataset   
  OpenAI Python client library  

# Frequently Asked Questions  
Q: Model loading failed?  
A: Ensure that the Ollama service is running and execute ollama  qwen:110b.  
Q: Slow processing speed?  
A: Try the following methods:  
Use a smaller model.  
Increase the timeout parameter.  
Process data in batches.  
Q: How to customize tags?  
A: Modify the detection rules in the prompt_templates dictionary.  
