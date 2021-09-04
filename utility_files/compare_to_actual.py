# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:44:27 2021

@author: Samarth Gupta (samarthgupta24@gmail.com)
"""

import numpy as np
import re
from datasets import load_dataset, load_metric
wer_metric = load_metric("wer")

def wer_from_actual(pred_text,actual_text):
    pred_text = process_text(pred_text)
    actual_text = process_text(actual_text)

    wer = wer_metric.compute(predictions=[pred_text], references=[actual_text])

    return {"wer": wer}

def process_text(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\\n"]'
    text = re.sub(chars_to_ignore_regex, '', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    return text


