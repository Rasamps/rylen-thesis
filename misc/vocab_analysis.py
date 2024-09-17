import json
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

with open("(insert_a_file_containing_text_predictions)", "r") as file:
    pred_ref = json.load(file)

index = 0
vocab = {}
references = pred_ref["predictions"]

for ref in references:
    tokens = nlp(ref)
    for token in tokens:
        if (token not in vocab):
            vocab[token] = index
            index += 1

print(len(vocab))