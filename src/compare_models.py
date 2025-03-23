#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install transformers')

from transformers import pipeline


# In[3]:


mpnet_model = "tiedaar/short-answer-classification"
bleurt_model = "vaiibhavgupta/finetuned-bleurt-large"
modernbert_model = "answerdotai/ModernBERT-base"
bleurt_threshold = 0.7


# In[4]:


class Bleurt():
    model_name = "vaiibhavgupta/finetuned-bleurt-large"
    threshold = 0.7

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            device="cuda",
        )

    def __call__(self, candidate: str, reference: str) -> int:
        sequence = f"{candidate}[SEP]{reference}"

        result = self.classifier(sequence)
        score = result[0]["score"]

        return 1 if score > self.threshold else 0


# In[5]:


class Mpnet():
    model_name = "tiedaar/short-answer-classification"
    revision = "77b846ec4606bfcfdf913888d7f0ab51f977a579"

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            revision=self.revision,
            device="cuda",
            )

    def __call__(self, candidate: str, reference: str) -> int:
        sequence = f"{candidate}</s>{reference}"

        result = self.classifier(sequence)
        label = result[0]["label"]

        return 1 if label == "correct_answer" else 0


# In[16]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ModernBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base") ## change model_path??
        self.model = AutoModelForSequenceClassification.from_pretrained("../results/modernbert_multirc")
        self.model.eval()  # Set model to evaluation mode
    
    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs to get predictions
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return  1 if predicted_class == 1 else 0

