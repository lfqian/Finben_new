import evaluate
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import datetime
from collections import defaultdict
from FactScoreLite.factscore import FactScore

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


# bertscore
def bertscore(items):
    """
    Calculate BERTScore for a list of (reference, candidate) pairs.
    passthrough for efficiency
    """
   
    return items

def bertscore_agg(items):
    """
    Aggregate BERTScore F1 scores for a list of items.
    Higher is better.
    """

    refs = [normalize_bertscore_text(item[0]) for item in items]
    preds = [normalize_bertscore_text(item[1]) for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("evaluate-metric/bertscore",device=DEVICE)

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang='en')
    
    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0

def normalize_bertscore_text(text):
    
    """
    You can achieve the specific functions
    """
    
    ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

    text = text.lower()

    # Remove stock tickers
    text = ticker_pattern.sub('', text)

    # Remove hyphens/dashes
    text = re.sub(r'[-–—]', ' ', text)

    # Remove punctuation
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\_`~()"]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# rouge1
def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]



## FActScore
def FActScore(items):

    return items

def FActScore_agg(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    ## change this path to your Finben directory
    path1 = "~/FinBen/decisions.json"
    path2 = "~/FinBen/facts.json"
    if os.path.exists(path1):
        os.remove(path1)

    if os.path.exists(path2):
        os.remove(path2)
        
    fact_scorer = FactScore()
    scores, _ = fact_scorer.get_factscore(generations=preds, knowledge_sources=refs)

    return scores


## Accuracy
def math_acc(items):
    return items


def math_acc_agg(items):
    true_answer = [extract_first_number(item[0]) for item in items]
    pred_answer = [extract_first_number(item[1]) for item in items]

    # Define tolerance percentage (5% allowed deviation)
    tolerance = 0.05  # 5%

    correct = 0
    for true_number, pred_number in zip(true_answer, pred_answer):
        if true_number is not None and pred_number is not None:
            difference = abs(true_number-pred_number)
            allowed_difference = true_number*tolerance

            if difference <= allowed_difference:
                correct += 1
        elif true_number is None and pred_number is None:
            correct += 1
        else:
            continue

    accuracy = correct/len(true_answer)
    return accuracy

def extract_first_number(value):
    """
    Extracts the first numeric value from a given string.
    Ignores any explanations or additional text after the number.
    Returns the number as a float or None if not found.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    match = re.search(r"-?\d+(\.\d+)?", value)
    return float(match.group(0)) if match else None 



    









