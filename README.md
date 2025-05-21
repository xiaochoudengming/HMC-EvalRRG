# HMC-EvalRRG
Boosting Clinical Value in Radiology Report Generation: An new Evaluation Framework by Human-Machine cooperation《Boosting Clinical Value in Radiology Report Generation: An new Evaluation Framework by Human-Machine cooperation》  
A medical imaging report analysis system based on Ollama's local large language model, automatically detecting 21 chest X-ray pathologies.  
Features：  
1.Supports prediction of over 21 types of chest disease tags, such as atelectasis, cardiomegaly, pneumonia, etc.  
2.Deployed using the Ollama local large language model.  
3.Includes features for resuming from breakpoints and automatic error logging.  
4.Outputs results in CSV format  
![image](https://github.com/user-attachments/assets/f5dd976f-f4dc-47d5-9211-b51c3d44ce8a)

Prerequisites：  
Python 3.8+  
Ollama service (running locally)  
Install required libraries:  

```pip install -r requirements.txt```

Data Preparation  
We use the MIMIC-CXR dataset.To handle Chinese reports, add encoding='gbk' when specifying the input report path.  
You can prepare medical report data in CSV format.  
The file should include the following fields:  
  study_id: Study identifier  
  id: Case ID  
  report: Imaging report text  
Example Structure:  
```csv
study_id,id,report
1001,1,"PA view shows right middle lobe consolidation..."
```

Configuration Instructions:  
Modify the parameters of the model_predict() function:  
```python
client = OpenAI(
    base_url='http://localhost:11434/v1',  
    api_key='ollama',                      
    timeout=30                             
)
```

Output Results  
The output file qwen.csv includes:  
  Original report identifier  
  Model response in JSON format  
  Prediction results for each pathological label (0/1)  
Example:  
```csv
study_id,id,content,Atelectasis,Cardiomegaly,...
1001,1,"{'Atelectasis': '0',...}",0,1,...,0
```

Acknowledgements  
Ollama for providing local model services  
MIMIC-CXR dataset   
OpenAI Python client library  

Frequently Asked Questions  
Q: Model loading failed?  
A: Ensure that the Ollama service is running and execute ollama  qwen:110b.  
Q: Slow processing speed?  
A: Try the following methods:  
Use a smaller model.  
Increase the timeout parameter.  
Process data in batches.  
Q: How to customize tags?  
A: Modify the detection rules in the prompt_templates dictionary.  
