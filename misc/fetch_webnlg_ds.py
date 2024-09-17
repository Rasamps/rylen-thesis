import numpy as np
import pandas as pd
from datasets import load_dataset

def load_data():
    dataset = load_dataset("web_nlg", "release_v3.0_en")
    train = dataset["train"]
    dev = dataset["dev"]
    test = dataset["test"]
    return train, dev, test
    
def process_train(train):
    to_return = {"id": [], "split": [], "category": [], "size": [], "input": [], "output": []}
    for i, row in enumerate(train):
        for output in row["lex"]["text"]:
            to_return["id"].append(i)
            to_return["split"].append("train")
            to_return["category"].append(row["category"])
            to_return["size"].append(row["size"])
            input = " [SEP] ".join(row["modified_triple_sets"]["mtriple_set"][0])
            to_return["input"].append(input)
            # output = " <N> ".join(row["lex"]["text"])
            to_return["output"].append(output)
    return to_return

def process_dev(dev):
    to_return = {"id": [], "split": [], "category": [], "size": [], "input": [], "output": []}
    for i, row in enumerate(dev):
        for output in row["lex"]["text"]:
            to_return["id"].append(i)
            to_return["split"].append("dev")
            to_return["category"].append(row["category"])
            to_return["size"].append(row["size"])
            input = " [SEP] ".join(row["modified_triple_sets"]["mtriple_set"][0])
            to_return["input"].append(input)
            # output = " <N> ".join(row["lex"]["text"])
            to_return["output"].append(output)
    return to_return

def process_test(test):
    to_return = {"id": [], "split": [], "category": [], "seen_category": [], "no_ref": [], "size": [], "input": [], "output": []}
    for i, row in enumerate(test):
        if (row["test_category"] == "rdf-to-text-generation-test-data-without-refs-en"):
            to_return["id"].append(i)
            to_return["split"].append("test")
            to_return["category"].append(row["category"])
            if (row["category"] in ["MusicalWork","Scientist","Film"]):
                to_return["seen_category"].append(False)
            else:
                to_return["seen_category"].append(True)
            to_return["no_ref"].append(True)
            to_return["size"].append(row["size"])
            input = " [SEP] ".join(row["modified_triple_sets"]["mtriple_set"][0])
            to_return["input"].append(input)           
            to_return["output"].append("Empty")
        else:
            for output in row["lex"]["text"]:
                to_return["id"].append(i)
                to_return["split"].append("test")
                to_return["category"].append(row["category"])
                if (row["category"] in ["MusicalWork","Scientist","Film"]):
                    to_return["seen_category"].append(False)
                else:
                    to_return["seen_category"].append(True)
                to_return["no_ref"].append(False)
                to_return["size"].append(row["size"])
                input = " [SEP] ".join(row["modified_triple_sets"]["mtriple_set"][0])
                to_return["input"].append(input)
                # output = " <N> ".join(row["lex"]["text"])
                to_return["output"].append(output)
    return to_return

def __main__():
    train, dev, test = load_data()
    train = pd.DataFrame.from_dict(process_train(train))
    dev = pd.DataFrame.from_dict(process_dev(dev))
    test = pd.DataFrame.from_dict(process_test(test))

    train.to_csv("(path_to_data)/webnlg_train.csv", header = True, index = False)
    dev.to_csv("(path_to_data)/webnlg_dev.csv", header = True, index = False)
    test.to_csv("(path_to_data)/webnlg_test.csv", header = True, index = False)

__main__()