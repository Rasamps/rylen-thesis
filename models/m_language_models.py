# General libaries
import os
import numpy as np
import pickle
import torch
from time import time

# Class libraries
from abc import ABCMeta, abstractmethod

# Dataset libraries
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset

# Modelling libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
from transformers.optimization import Adafactor
from augmentation.augmentation import *

# Scoring libraries
# from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
# from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from evaluate import load

device = torch.device("cpu")

class E2EDataset(Dataset):
    '''
    This forms a Dataset object that is loadable by PyTorch's native model training capabilities. Specific naming conventions for the E2E dataset.

    input:
    - source data (meaning representations)
    - target text
    '''
    def __init__(self, mrs, texts):
        self.mrs = mrs
        self.texts = texts

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "src_data": self.mrs[idx],
            "tgt_text": self.texts[idx],
        }

class WebNLGDataset(Dataset):
    '''
    This forms a Dataset object that is loadable by PyTorch's native model training capabilities. Specific naming conventions for the WebNLG dataset.

    input:
    - source data (semantic triples)
    - target text
    '''
    def __init__(self, trips, texts):
        # self.ids = ids
        self.trips = trips # input --> numerical after 
        self.texts = texts # output
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "src_data": self.trips[idx],
            "tgt_text": self.texts[idx],
        }

class Model(metaclass = ABCMeta):
    '''
    A generic class for all models using the HuggingFace API to be built upon offering shared functionality.
    '''
    def __init__(self, name):
        self.name = name

    # Abstract method that's passed down to the specific model classes
    @abstractmethod
    def build(self):
        pass

    # @abstractmethod
    def gen(self, test, seen_category):
        '''
        A generation function that takes the test dataset and using a HuggingFace language model, generates predictions.
        '''
        if (seen_category == True):
            seen_category = test["seen_category"].unique()[0]
            
        references, predictions = [], []
        
        # Depending on the dataset, the maximum sequence length that can be generated or the total number of references might change.
        if (self.ds == "webnlg"):
            test_set = WebNLGDataset(test.input.values, test.output.values)
            padded_refs = 5
            max_len = 130
        else: # self.ds == e2e
            test_set = E2EDataset(test.input.values, test.output.values)
            padded_refs = 45
            max_len = 70

        # Load the test dataset into a PyTorch DataLoader object to facilitate the train-test loop.
        test_loader = DataLoader(test_set, batch_size = 8, shuffle = False)

        self.model.to(device)
        self.model.eval()

        # Loop through the batches made by the DataLoader
        for i, batch in enumerate(test_loader):
            if (i % 100 == 0):
                print("Finished {} / {} batches.".format(i, len(test_loader)))

            # Tokenize the data and add tokens up to the longest item in the batch
            src = self.tokenizer.batch_encode_plus(batch["src_data"], padding = "longest", return_tensors = "pt")

            # These Beam Search parameters were selected using cross-validation. For other models or data, these may need to be re-tuned.
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids = src["input_ids"].to(device),
                    max_length = max_len,
                    min_length = 5,
                    length_penalty = -1.0,
                    repetition_penalty = 2.5,
                    num_beams = 5,
                ).to(device)

            # Store the predictions and references into lists to return
            predictions.extend([self.tokenizer.decode(output, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output in outputs])
            references.extend([item.split(" <N> ") for item in batch["tgt_text"]])

        # Add mock references to work within the n-gram evaluation libraries
        for r, ref in enumerate(references):
            if (len(ref) < padded_refs):
                references[r] = ref + [""] * (padded_refs - len(ref))
        
        return {"predictions": predictions, "references": references, "observed": seen_category}

    def ngram_eval(self, pred_ref):
        '''
        Taking the predictions made by a model and evaluating using metrics such as BLEU, ROUGE, and METEOR
        Will compute and return via the command line, but also write to a .csv

        input:
        - pred_ref: list of lists, one is the predictions and the other the references.
        '''
        self.model.eval()

        # Load in the n-gram evaluation functions
        sacrebleu = load("sacrebleu")
        rouge = ROUGEScore()
        meteor = load("meteor")

        predictions = pred_ref["predictions"]
        references = pred_ref["references"]

        # Compute the BLEU score
        bleu = sacrebleu.compute(predictions = predictions, references = references,
                                 tokenize = "13a")
        print("\nThe corpus bleu score is:", bleu["score"],"\n")

        # Compute all ROUGE (Longest Common Substring) scores
        rouge_scores = rouge(predictions, references)
        print("These are the ROUGE score metrics:")
        print("rougeL F measure =", rouge_scores["rougeLsum_fmeasure"].item())
        print("rougeL precision =", rouge_scores["rougeLsum_precision"].item())
        print("rougeL recall =", rouge_scores["rougeLsum_recall"].item(),"\n")

        # Compute the METEOR score
        meteor_score = meteor.compute(predictions = predictions, references = references)
        print("The meteor score is:", meteor_score["meteor"],"\n")

        # Read-in an existing results spreadsheet and add the scores as well as the current model details
        # ADD YOUR PATH TO RESULTS HERE
        results = pd.read_csv("" + self.ds + "_results.csv", header = 0, index_col = None)
        to_add = {}
        to_add["model"] = [self.name]
        to_add["bleu"] = [np.round(bleu["score"], 4)]
        to_add["rougeL_F"] = [np.round(rouge_scores["rougeLsum_fmeasure"].item() * 100, 4)]
        to_add["rougeL_P"] = [np.round(rouge_scores["rougeLsum_precision"].item() * 100, 4)]
        to_add["rougeL_R"] = [np.round(rouge_scores["rougeLsum_recall"].item() * 100, 4)]
        to_add["meteor"] = [np.round(meteor_score["meteor"] * 100, 4)]
        to_add["seen"] = [pred_ref["observed"]]
        to_add["n"] = [self.n]
        to_add["ds"] = [self.ds]
        to_add["cl"] = [self.cl]
        results = pd.concat([results, pd.DataFrame.from_dict(to_add)])
        # ADD YOUR PATH TO RESULTS HERE
        results.to_csv("" + self.ds + "_results.csv", header = True, index = False)

    def bs_eval(self, pred_ref):
        '''
        Evaluates the model's performance with the BERTScore metric
        '''
        bertscore = load("bertscore")

        predictions = pred_ref["predictions"]
        references = pred_ref["references"]

        bert_scores = bertscore.compute(predictions = predictions, references = references, lang = "en")
        print("The BERT score is:", np.mean(bert_scores["f1"]),"\n")
        
        # Add the score to the metrics file.
        # ADD YOUR PATH TO RESULTS HERE
        results = pd.read_csv("/bertscore_results.csv", header = 0, index_col = None)
        to_add = {}
        to_add["model"] = [self.name]
        to_add["bertscore"] = [np.round(100 * np.mean(bert_scores["f1"]), 4)]
        to_add["seen"] = [pred_ref["observed"]]
        to_add["n"] = [self.n]
        to_add["ds"] = [self.ds]
        to_add["cl"] = [self.cl]
        results = pd.concat([results, pd.DataFrame.from_dict(to_add)])
        # ADD YOUR PATH TO RESULTS HERE
        results.to_csv("/bertscore_results.csv", header = True, index = False)

    def predict(self, src_data):
        '''
        Takes source data not within either of the datasets and generates the most likely text describing that source data.

        input:
        - source data: can be of any shape or form, but likely will have deteriorating quality on unseen formats.
        '''
        self.model.eval()
        self.model.to(device)
        
        to_return = self.tokenizer.encode_plus("Describe the following data: {}".format(src_data), return_tensors = "pt").to(device)
        outputs = self.model.generate(input_ids = to_return["input_ids"], attention_mask = to_return["attention_mask"],
                                        max_length = 300, early_stopping = True, length_penalty = 1.0, repetition_penalty = 2.5,
                                        num_beams = 4)
        predictions = self.tokenizer.decode(outputs[0])
        
        print("Input:", src_data, "\nOutput:", predictions)

class T5Model(Model):
    '''
    Class for a transformer-based language model that leverags the T5 architecture. The architecture is loaded in using HuggingFace.
    This Class also inherits from the general Model class.

    input:
    - name: the name of the model to be saved as in a directory
    - model: a class parameter to store the HuggingFace or PyTorch model to be called from
    - tokenizer: the associated tokenizer from whichever library the model originates from
    - src_length: maximum number of tokens that can be in the source data
    - tgt_length: maximum number of tokens that the model can generate
    - epochs: the number of epochs to train for
    - n: the number of examples within the training dataset to use
    - ds: the dataset being used in the current experiment
    
    functions:
    - __init__(): Initializes an instance of this class.
    - build(): Loads in training data and fine-tunes the model with it. Returns a train and validated model.
    '''
    def __init__(self, ds, file = "temp", epochs = 3):
        self.name = file
        self.model = None
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.src_length = 275
        self.tgt_length = 130
        self.epochs = int(epochs)
        self.n = 0
        self.ds = ds
        
    def build(self, train, dev, augment):
        '''
        Trains or fine-tunes a transformer / language model on a dataset for data-to-text generation. The model is then validated on a second set of data.

        input:
        - train: the training split of the dataset being used (compatible with PyTorch DataLoader)
        - dev: the secondary split of data used to estimate the model's performance (compatible with PyTorch DataLoader)
        - augment: a string that dictates what data augmentation strategy is going to be used, if any

        output:
        - returns a trained / fine-tuned model that can be used on a test set or further experiments.
        '''

        # Check to see if there's an existing model, otherwise initialize a new one.
        if (os.path.isdir("models/" + str(self.name))):
            print("Existing model loaded.")
            # ADD YOUR PATH TO SAVED MODELS HERE
            self.model = T5ForConditionalGeneration.from_pretrained("/" + str(self.name))
        else:
            print("Beginning to train a vanilla T5 language model for conditional generation.")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

            # Depending on the determined dataset, load in the data with the respective data Class
            self.n = len(train)
            if (self.ds == "webnlg"):
                train = WebNLGDataset(train.input.values,
                                    train.output.values)
                dev = WebNLGDataset(dev.input.values,
                                    dev.output.values)
            else: #(self.ds == "e2e"):
                train = E2EDataset(train.input.values,
                                   train.output.values)
                dev = E2EDataset(dev.input.values,
                                 dev.output.values)

            # Set-up the PyTorch dataloaders
            train_loader = DataLoader(train, batch_size = 4, shuffle = not self.cl)
            dev_loader = DataLoader(dev, batch_size = 4, shuffle = False)

            # Freeze the model parameters in the encoder to ensure that the benefits of transfer learning do not dissipate
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            # Set-up the HuggingFace optimizer 
            optimizer = Adafactor(
                self.model.decoder.parameters(),
                lr = 1e-3, eps = (1e-30, 1e-3), clip_threshold = 1.0,
                decay_rate = -0.8, beta1 = None, weight_decay = 0.0,
                relative_step = False, scale_parameter = False, warmup_init = False,
            )

            self.model.to(device)

            # Lists for tracking the training and development loss
            total_train_loss, total_dev_loss = [], []
            min_dev_loss = np.inf

            # Begin the train-validate loop for a number of epochs
            for epoch in range(self.epochs):
                self.model.train()
                # Reset training loss for the current epoch
                train_loss = 0.0
                # Loop through the batches in the training dataset
                for i, batch in enumerate(train_loader):
                    # Select and apply any data augmentation strategies
                    if (augment in ["paraphrase", "erase_src", "erase_tgt", "sub_src", "sub_tgt"]):
                        if (augment == "paraphrase"):   batch = paraphrasing(batch)
                        elif (augment == "erase_src"):  batch = erase_src(batch, self.ds)
                        elif (augment == "erase_tgt"):  batch = erase_tgt(batch)
                        else:   batch = sub_tgt(batch)

                    # Tokenize the source data and target text to be used within the language model architecture
                    src = self.tokenizer.batch_encode_plus(batch["src_data"], padding = "longest", return_tensors = "pt")
                    tgt = self.tokenizer.batch_encode_plus(batch["tgt_text"], padding = "longest", return_tensors = "pt")
                    # Specific to the T5 model
                    tgt["input_ids"][tgt["input_ids"] == 0] = -100

                    # Get the model outputs from the last network layer
                    outputs = self.model(input_ids = src["input_ids"].to(device), 
                                         attention_mask = src["attention_mask"].to(device),
                                         labels = tgt["input_ids"].to(device))

                    # Obtain and add the loss to our training loss variable
                    loss = outputs[0]
                    train_loss += loss.item()

                    # Every 150 batches print out the epoch, batch, and current training loss
                    if (i % 150 == 0):
                        print("Epoch: {} / {}, Batch: {} / {}, Loss: {}".format(epoch, self.epochs, i, len(train_loader), loss.item()))

                    # Reset the optimizer and gradient for the next batch.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Do development testing at the end of each epoch. Code follows the same routine as training, but with the model set to eval() mode
                dev_loss = 0.0
                self.model.eval()
                for j, batch_t in enumerate(dev_loader):
                    src = self.tokenizer.batch_encode_plus(batch_t["src_data"], padding = "longest", return_tensors = "pt")
                    tgt = self.tokenizer.batch_encode_plus(batch_t["tgt_text"], padding = "longest", return_tensors = "pt")
                    tgt["input_ids"][tgt["input_ids"] == 0] = -100

                    outputs = self.model(input_ids = src["input_ids"].to(device), labels = tgt["input_ids"].to(device))
                    dev_loss += outputs[0].item()

                # Print out details at the end of an epoch
                print("End of epoch {}".format(epoch))
                print("The average training loss is: {}".format(train_loss / len(train_loader)))
                print("The average loss on the dev set is: {}".format(dev_loss / len(dev_loader)))
                if (min_dev_loss > dev_loss / len(dev_loader)):
                    print("The dev loss has decreased since the last epoch \n")

                total_train_loss.append(train_loss / len(train_loader))
                total_dev_loss.append(dev_loss / len(dev_loader))

            # Report model training and validation loss results to a .csv tracker
            # ADD YOUR PATH TO RESULTS HERE
            loss_results = pd.read_csv("", header = 0, index_col = None)
            to_add = {}
            to_add["model"] = [self.name + str(self.n)]
            to_add["n"] = [self.n]
            to_add["ds"] = [self.ds]
            to_add["cl"] = [self.cl]
            to_add["train_loss"] = [", ".join(map(str, total_train_loss))]
            to_add["dev_loss"] = [", ".join(map(str, total_dev_loss))] 
            loss_results = pd.concat([loss_results, pd.DataFrame.from_dict(to_add)])
            # ADD YOUR PATH TO RESULTS HERE
            loss_results.to_csv("", header = True, index = False)

            # Write the model to a directory and report that training is finished
            # ADD YOUR PATH TO SAVED MODELS HERE
            self.model.save_pretrained("/" + str(self.name))
            print("Model saved.")