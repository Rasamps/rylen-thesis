# Data stuff
import pandas as pd
import numpy as np
# Text stuff
import re
from unidecode import unidecode

# Make a tokenization function for the input
def webnlg_input_tokenizer(text):
    tokenization = text.split(" [SEP] ")
    tokenization = [token for triple in tokenization for token in triple.split(" | ")]
    tokenization = [subtoken for token in tokenization for subtoken in token.split("_")]
    tokenization = [subtoken for token in tokenization for subtoken in re.sub(r'(\w)([A-Z])', r'\1 \2', token).split()]
    tokenization = [unidecode(token).lower() for token in tokenization]
    tokenization = [subtoken for token in tokenization for subtoken in re.sub(r'[^A-Za-z0-9\.]', ' ', token).split()]

    return tokenization

def e2e_input_tokenizer(text):
    reg = r'\[.*?\]'
    attributes = re.sub(reg, "", text).split(", ")
    values = re.findall(reg, text)
    tokenization = [attributes[l//2] if (l%2 == 0) else values[l//2][1:-1] for l in range(len(attributes) + len(values))]
    tokenization = [subtoken for token in tokenization for subtoken in re.sub(r'(\w)([A-Z])', r'\1 \2', token).split()]
    tokenization = [unidecode(token).lower() for token in tokenization]
    return tokenization


# Make a tokenization function for the output
def output_tokenizer(text):
    tokenization = text.replace("(",  "").replace(")", "")
    tokenization = tokenization.replace(",", "")
    tokenization = re.sub("\.$", "", tokenization)
    tokenization = tokenization.replace("'s", " ")
    tokenization = tokenization.replace("/", " ")
    tokenization = tokenization.replace("-", " ")
    tokenization = tokenization.split()
    tokenization = [unidecode(token).lower() for token in tokenization]

    return tokenization

# Make a vocabulary from the tokenized input
def create_input_vocabulary(texts):
    vocabulary = {"<s>": 1, "<pad>": 0, "</s>": 2, "<unk>": 3}
    index = 4

    for text in texts:
        for token in text:
            if (token not in vocabulary):
                vocabulary[token] = index
                index += 1

    return vocabulary

# Make a vocabulary from the tokenized output
def create_output_vocabulary(texts):
    vocabulary = {"<s>": 1, "<pad>": 0, "</s>": 2, "<unk>": 3}
    index = 4

    for text in texts:
        for token in text:
            if (token not in vocabulary):
                vocabulary[token] = index
                index += 1
    return vocabulary