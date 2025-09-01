#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from transformers import pipeline, AutoTokenizer


# In[2]:


bleurt_model = "vaiibhavgupta/finetuned-bleurt-large"
mpnet_model = "lear-lab/short-answer-classification"
modernbert_model = "/home/jovyan/active-projects/itell-question-generation/results/modernbert_multirc"
bleurt_threshold = 0.7


# In[3]:


class Bleurt():
    model_name = bleurt_model
    threshold = bleurt_threshold

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


# In[4]:


class Mpnet():
    model_name = mpnet_model
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


# In[8]:


from transformers import AutoTokenizer

class ModernBERT():
    model_name = modernbert_model
    tokenizer_name = "answerdotai/ModernBERT-base"


    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name),
            device="cuda",
            )

    def __call__(self, candidate: str, reference: str) -> int:
        seq = f"{candidate}</s>{reference}"
        out = self.classifier(seq)[0]
        label = out["label"]

        main = label.split("_")[0].lower()
        return 1 if main == "correct" else 0



if __name__ == "__main__":
    
    mpnet = Mpnet()
    bleurt = Bleurt()
    modernBERT = ModernBERT()
    
    
    # In[13]:
    
    
    pipes = {
        "Mpnet_pipe": Mpnet(),
        "Bleurt_pipe": Bleurt(),
        "ModernBERT_pipe": ModernBERT()
    }
    
def evaluate_and_score(dataset, pipes, label_key="labels"):
  
    texts, true_labels = [], []
    all_preds = { name: [] for name in pipes }

    for ex in dataset["test"]:
        text  = ex["text"]
        label = ex[label_key]

        texts.append(text)
        true_labels.append(label)

        for name, pipe in pipes.items():
            # call each adapter exactly once
            pred = pipe(text, ex.get("reference", None))  
            all_preds[name].append(pred)

    df = pd.DataFrame({
        "input_text":  texts,
        "true_label":  true_labels,
        **{ f"{n}_pred": preds for n, preds in all_preds.items() }
    })

    def score(col):
        return {
            "accuracy": accuracy_score(df["true_label"], df[col]),
            "f1_macro":  f1_score(df["true_label"], df[col], average="macro")
        }

    metrics = {
        name: score(f"{name}_pred")
        for name in pipes
    }

    return df, metrics
    
    # df_predictions = evaluate_all_models(ds, models)
    
    
    # In[14]:
    
    
    mpnet("This a strong answer to the question", "This is a reference answer to the question")  # Scored incorrect (0)
    
    
    # In[15]:
    
    
    bleurt("This a strong answer to the question", "This is a reference answer to the question")  # Scored correct (1)

