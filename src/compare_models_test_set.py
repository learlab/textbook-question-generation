#!/usr/bin/env python
# coding: utf-8

# In[27]:


import re

from tqdm.auto import tqdm
import pandas as pd
import torch
import datasets
from transformers import pipeline, AutoTokenizer

torch.set_float32_matmul_precision('high')


# In[16]:


dataset_path = '../bin/multirc_dataset.hf'
ds = datasets.DatasetDict.load_from_disk(dataset_path)["test"]
ds


# In[40]:


def split_samples(example):
    """Split strings into reference/answer components,
    so models can join them together differently."""
    reference, candidate = example["text"].split("</s>")
    candidate = candidate.strip().removeprefix("Answer:").strip()
    example["answer"] = reference
    example["response"] = candidate
    return example

ds = ds.map(split_samples, remove_columns=["index", "text"])


# In[21]:


class Bleurt():
    model_name = "vaiibhavgupta/finetuned-bleurt-large"
    threshold = 0.7

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            device="cuda",
        )

    def __call__(self, input_dict) -> int:
        reference = input_dict.get("answer", "")
        candidate = input_dict.get("response", "")
        
        sequence = f"{candidate}[SEP]{reference}"

        result = self.classifier(sequence)
        score = result[0]["score"]

        return 1 if score > self.threshold else 0


# In[22]:


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

    def __call__(self, input_dict) -> int:
        reference = input_dict.get("answer", "")
        candidate = input_dict.get("response", "")
        
        sequence = f"{candidate}</s>{reference}"

        result = self.classifier(sequence)
        label = result[0]["label"]

        return 1 if label == "correct_answer" else 0


# In[70]:


class MpnetLocal():
    """The model weights stored locally are the same as those in the hub.
    We won't use this because results are identifical to the Mpnet class above"""

    model_name = "../bin/mpnet_multimc_classifier"
    tokenizer_name = "microsoft/mpnet-base"

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name),
            device="cuda",
            )

    def __call__(self, input_dict) -> int:
        reference = input_dict.get("answer", "")
        candidate = input_dict.get("response", "")
        
        sequence = f"{candidate}</s>{reference}"

        result = self.classifier(sequence)
        label = result[0]["label"]

        return 1 if label == "correct_answer" else 0


# In[72]:


class ModernBERT():
    model_name = "../results/modernbert_multirc/"
    tokenizer_name = "answerdotai/ModernBERT-base"

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name),
            device="cuda",
            )

    def __call__(self, input_dict) -> int:
        reference = input_dict.get("answer", "")
        candidate = input_dict.get("response", "")
        sequence = f"{candidate}</s>{reference}"

        result = self.classifier(sequence)
        label = result[0]["label"]

        return 1 if label == "correct" else 0


# In[73]:


# NOTE: these classes are not designed to take advantage of Pipeline's batching optimizations.

pipe_dict = {
    "Mpnet": Mpnet(),
    # "MpnetLocal": MpnetLocal(),
    "Bleurt": Bleurt(),
    "ModernBERT": ModernBERT(),
}


# In[74]:


def evaluate_all_models(dataset, pipe_dict):
    pred_dict = {name: [] for name in pipe_dict.keys()}

    for name, pipe in tqdm(pipe_dict.items(), total=len(pipe_dict)):
        for example in tqdm(dataset, total=len(dataset)):
            pred_dict[name].append(pipe(example))

    return pd.DataFrame(pred_dict)

df = evaluate_all_models(ds, pipe_dict)
df["labels"] = ds["labels"]


# In[75]:


df.sample(5)


# In[90]:


df.to_csv("../data/multirc-dataset-preds.csv", index=False)


# In[77]:


df["ensemble"] = (
    (df["Mpnet"] == 1)
    | (df["Bleurt"] == 1)
).astype(int)

df[[
    "labels",
    "ensemble",
    "Mpnet",
    "Bleurt",
    "ModernBERT",
]].corr(method="spearman")


# In[93]:


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

def get_metrics(df, model_names):
    metrics = []
    
    for model_name in model_names:
        preds = df[model_name]

        acc = accuracy_score(df["labels"], preds)
        p, r, f1, _ = precision_recall_fscore_support(
            df['labels'],
            preds,
            average="binary",
            pos_label=0
        )
        
        metrics.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': p,
            'Recall': r,
            'F1-Score': f1
        })

        print(classification_report(df["labels"], preds))
    return pd.DataFrame(metrics)

model_names = ["ensemble", "Mpnet", "MpnetLocal", "Bleurt", "ModernBERT"]
get_metrics(df, model_names).round(3)

