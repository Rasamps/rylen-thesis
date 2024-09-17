# General libraries
import sys
import numpy as np
import pandas as pd

# Modelling libraries
import torch
from models.m_language_models import *
from models.m_transformers import *
from models.m_copy import *

def __main__(model_file, data_path, augment, p):
    '''
    This function takes a command line prompt with certain parameters and runs the model training used in my thesis research.

    input:
    - model_file: the name the model will have when saved locally
    - data_path: only supports a local path to the datasets being used
    - augment: the augmentation strategy to be used, if any.
    '''
    # Read-in data and make any necessary changes. ADD YOUR PATH TO DATA HERE
    ds = data_path.split("_")[0]
    data_path = "" + ds + "/" + data_path + ".csv"

    # Read-in the train, dev, and test datasets. ADD YOUR PATHS TO DATA HERE
    train = pd.read_csv(data_path, header = 0, index_col = None).sample(frac = p)
    dev = pd.read_csv("" + ds + "/" + ds + "_dev.csv", header = 0, index_col = None)
    test = pd.read_csv("" + ds + "/" + ds + "_test.csv", header = 0, index_col = None)

    # The WebNLG dataset has two categories of content in its test dataset. The majority are subjects that were seen in the training data and
    # the minority are unseen categories which would be theoretically more challenging for a model to handle.
    if (ds == "webnlg"):
        test = test[test["no_ref"] == False]

        test_seen = test[test["seen_category"] == True]
        test_unseen = test[test["seen_category"] == False]

    # Select the appropriate model to train
    if ("t5" in model_file):
        print("T5 model selected")
        model = T5Model(ds, file = model_file, epochs = 3)
        model.build(train, dev, augment)
    elif ("transformer" in model_file):
        print("Transformer model selected")
        model = TransformerModel(ds, file = model_file)
        model.build(train, dev, augment)
    elif ("copy" in model_file):
        print("Copy approach selected")
        model = Copying(ds, file = model_file)
        model.build(train, dev)

    # Generate the predictions and references as a dictionary structure.
    pred_ref = model.gen(test, seen_category = "All")
    
    # Evaluate with n-gram metrics
    model.ngram_eval(pred_ref)

    # Evaluate with BERTScore metric
    model.bs_eval(pred_ref)

__main__(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])