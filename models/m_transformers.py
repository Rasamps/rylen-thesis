# General libaries
import os
import numpy as np
import math
import random
import pickle

# Dataset libraries
import torch
from torch.utils.data import DataLoader
import pandas as pd
from vocabulary import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

# Modelling libraries
from torch import nn
import torch.nn.functional as F
from transformers.optimization import Adafactor
from torch import optim
from torch.optim import Adam, Adagrad, Adadelta
from torch.nn import CrossEntropyLoss

# Scoring libraries
from torchmetrics.text.rouge import ROUGEScore
from evaluate import load

device = torch.device("cpu")

'''
-----------------------------------------------------------------------------------------------
The next three datasets were all used to test that the Transformer model was working properly on easy tasks such as:
- copying a sequence of numbers
- shifting numbers in a sequence up one
- a special test dataset that has a few different rules
'''
class CopyDataset(Dataset):
    def __init__(self, n):
        self.src, self.tgt = self.make_data(n)
        self.n = n
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {
            "src_ids": self.src[idx],
            "tgt_ids": self.tgt[idx]
        }
    
    def make_data(self, n):
        sos_token = np.array([1])
        eos_token = np.array([2])
        length = 18
        domain = [3, 4, 5, 6, 7, 8]
        data = []

        for i in range(n):
            values = np.random.choice(domain, np.random.randint(low = 1, high = length))
            if (len(values) < 18):
                pad = np.zeros(18-len(values))
            x = np.concatenate((sos_token, values, eos_token, pad))
            y = x.copy()
            data.append([x, y])
        
        np.random.shuffle(data)
        data = torch.tensor(data).type(torch.LongTensor)
        return data[:, 0, :], data[:, 1, :]

class ShiftUpDataset(Dataset):
    def __init__(self, n):
        self.src, self.tgt = self.make_data(n)
        self.n = n
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {
            "src_ids": self.src[idx],
            "tgt_ids": self.tgt[idx]
        }
    
    def make_data(self, n):
        sos_token = np.array([1])
        eos_token = np.array([2])
        length = 18
        domain = [3, 4, 5, 6, 7, 8]
        data = []

        for i in range(n):
            values = np.random.choice(domain, np.random.randint(low = 1, high = length))
            values_y = values + 1
            if (len(values) < 18):
                pad = np.zeros(18-len(values))
            x = np.concatenate((sos_token, values, eos_token, pad))
            y = np.concatenate((sos_token, values_y, eos_token, pad))
            data.append([x, y])
        
        np.random.shuffle(data)
        data = torch.tensor(data).type(torch.LongTensor)
        return data[:, 0, :], data[:, 1, :]

class TestDataset(Dataset):
    def __init__(self, n):
        self.src, self.tgt = self.make_data(n)
        self.n = n

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {
            "src_ids": self.src[idx],
            "tgt_ids": self.tgt[idx]
        }

    def make_data(self, n):
        sos_token = np.array([0])
        eos_token = np.array([2])
        length = 8
        data = []

        # 3, 3, 3, 3 --> 3, 3, 3, 3
        for i in range(n // 3):
            x = np.concatenate((sos_token, np.repeat(3, length), eos_token))
            y = np.concatenate((sos_token, np.repeat(3, length), eos_token))
            data.append([x,y])

        for i in range(n // 3):
            x = np.concatenate((sos_token, np.repeat(4, length), eos_token))
            y = np.concatenate((sos_token, np.repeat(4, length), eos_token))
            data.append([x,y])

        for i in range(n // 3):
            x = np.repeat(3, length)
            start = random.randint(0,1)
            x[start::2] = 4

            y = np.repeat(3, length)
            if x[-1] == 3:
                y[::2] = 4
            else:
                y[1::2] = 4

            x = np.concatenate((sos_token, x, eos_token))
            y = np.concatenate((sos_token, y, eos_token))
            data.append([x,y])

        np.random.shuffle(data)
        data = torch.tensor(data)
        return data[:, 0, :], data[:, 1, :]

'''
-----------------------------------------------------------------------------------------------
'''

class TransformersDataset(Dataset):
    '''
    Class that prepares the WebNLG dataset for training. Similar to those for the language model
    '''
    def __init__(self, trips, texts):
        self.trips = trips
        self.texts = texts
        # YOU WILL HAVE TO CREATE AN INPUT VOCABULARY FOR THE TRANSFORMER AND ADD PATH TO IT HERE. RECOMMEND JSON
        with open("") as vocab_file:
            self.input_vocab = json.load(vocab_file)
        # YOU WILL HAVE TO CREATE AN OUTPUT VOCABULARY FOR THE TRANSFORMER AND ADD PATH TO IT HERE. RECOMMEND JSON
        with open("vocabulary/webnlg_output_vocab.json") as vocab_file:
            self.output_vocab = json.load(vocab_file)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        trip = [1] + [self.input_vocab[token] if (token in self.input_vocab.keys()) else 3 for token in input_tokenizer(self.trips[idx])] + [2]
        text = [1] + [self.output_vocab[token] if (token in self.output_vocab.keys()) else 3 for token in output_tokenizer(self.texts[idx])] + [2]

        trip = torch.tensor(trip)
        text = torch.tensor(text)

        return {
            "ids": idx,
            "src_data": self.trips[idx],
            "src_ids": trip.squeeze(),
            "tgt_text": self.texts[idx],
            "tgt_ids": text.squeeze(),
        }
    
'''
----------------------------------------------------------------------------------------------------------------------------------------------------
Vanilla transformer architecture implementation as seen in:
- https://github.com/hyunwoongko/transformer
- https://pytorch.org/tutorials/beginner/translation_transformer.html
- https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

Check out these links for further information and explanation on how the code works
'''
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout_p, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, hidden_dim)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0)) / hidden_dim) # 1000^(2i/hidden_dim)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/hidden_dim))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/hidden_dim))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers = 6, num_heads = 8, dropout = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.src_embedding = nn.Embedding(input_dim, hidden_dim, padding_idx = 0, device = device)
        self.tgt_embedding = nn.Embedding(output_dim, hidden_dim, padding_idx = 0, device = device)

        self.positional_encoder = PositionalEncoding(hidden_dim, dropout)


        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model = hidden_dim, nhead = num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)


        self.transformer = nn.Transformer(d_model = hidden_dim, nhead = num_heads,
                                          num_encoder_layers = num_layers, num_decoder_layers = num_layers,
                                          dropout = dropout, device = device)

        self.fc = nn.Linear(hidden_dim, output_dim, device = device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask).to(device)

        seq_mask = self.make_tgt_mask(tgt).to(device)
        pad_mask = self.make_tgt_pad_mask(tgt).to(device)
        
        output = self.decode(tgt, memory, src_mask, seq_mask, pad_mask)
        output = self.fc(output)
        return output
      
    def forward_pass(self, src, tgt):
        src_mask = self.make_src_mask(src)
        seq_mask = self.make_tgt_mask(tgt).to(device)
        pad_mask = self.make_tgt_pad_mask(tgt).to(device)

        src = self.src_embedding(src).permute(1, 0, 2) * math.sqrt(self.hidden_dim)
        src = self.positional_encoder(src)
        tgt = self.tgt_embedding(tgt).permute(1, 0, 2) * math.sqrt(self.hidden_dim)
        tgt = self.positional_encoder(tgt)

        output = self.transformer(src = src, tgt = tgt, tgt_mask = seq_mask, src_key_padding_mask = src_mask,
                                  tgt_key_padding_mask = pad_mask, memory_key_padding_mask = src_mask)
        output = self.fc(output)
        return output

    def encode(self, src, src_mask):
        src = self.src_embedding(src).permute(1, 0, 2) * math.sqrt(self.hidden_dim)
        src = self.positional_encoder(src)
        enc_src = self.encoder(src = src, src_key_padding_mask = src_mask)

        return enc_src

    def decode(self, tgt, enc_src, memory_mask, seq_mask, pad_mask = None):
        tgt = self.tgt_embedding(tgt).permute(1, 0, 2) * math.sqrt(self.hidden_dim)
        tgt = self.positional_encoder(tgt)
        
        output = self.decoder(tgt = tgt, memory = enc_src, tgt_mask = seq_mask,
                              tgt_key_padding_mask = pad_mask, memory_key_padding_mask = memory_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src == 0)
        return src_mask

    def make_tgt_mask(self, tgt):
        seq_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal = 1)
        seq_mask = seq_mask.masked_fill(seq_mask == 1, float("-inf"))
        return seq_mask

    def make_tgt_pad_mask(self, tgt):
        pad_mask = (tgt == 0)
        return pad_mask

'''
----------------------------------------------------------------------------------------------------------------------------------------------------
'''

class TransformerModel():
    '''
    Class that implements the training process for the Transformer model. Helper classes above with further details on the model's inner-workings

    input:
    - name: the name of the model to be saved as in a directory
    - model: a class parameter to store the HuggingFace or PyTorch model to be called from
    - epochs: the number of epochs to train for
    - n: the number of examples within the training dataset to use
    - ds: the dataset being used in the current experiment
    - input_dim: maximum possible vocabulary for the source data
    - output_dim: maximum possible vocabulary for the target text
    '''
    def __init__(self, ds, file = "temp", epochs = 10):
        self.name = file
        self.model = None
        self.epochs = int(epochs)
        self.n = 0
        self.ds = ds
        if (ds == "webnlg"):
            self.input_dim = 4257
            self.output_dim = 9354
        else:
            self.input_dim = 127
            self.output_dim = 3135

    def build(self, train, dev, augment):
        '''
        The build function trains a Transformer model using the PyTorch Dataloader

        input:
        - train: the training dataset set-up with a PyTorch dataloader
        - dev: the secondary dataset used to validate the trained model, set-up as a PyTorch dataloader object
        - augment: the type of data augmentation strategy
        '''
        def collate_fct_pad(batch):
            '''
            Helper function to dynamically pad batches to the longest sequence in the bunch
            '''
            to_return = {}
            keys = list(batch[0].keys())
            for k in keys:
                if (k == "src_ids" or k == "tgt_ids"):
                    to_add = [b[k] for b in batch]
                    to_return[k] = pad_sequence(to_add, batch_first = True, padding_value = 0)
                else:
                    to_return[k] = [b[k] for b in batch]
            return to_return

        # Empty the cache in advance
        torch.cuda.empty_cache()

        # If a model exists already then load it in
        if (os.path.isfile("models/" + self.name)):
            print("Existing model is being loaded...")
            # ADD PATH TO SAVED MODELS
            with open ("/" + self.name, "rb") as model_file:
                self.model = pickle.load(model_file).to(device)

        # Otherwise train a new one
        else:
            # Set-up the training and validation datasets
            self.n = len(train)
            train = TransformersDataset(
                train.input.values,
                train.output.values,
            )
            dev = TransformersDataset(
                dev.input.values,
                dev.output.values
            )

            train_loader = DataLoader(train, batch_size = 4, shuffle = not self.cl, collate_fn = collate_fct_pad)
            dev_loader = DataLoader(dev, batch_size = 2, shuffle = False, collate_fn = collate_fct_pad)

            print("Training new transformer model...")
            self.model = Transformer(
                input_dim = 4257,
                output_dim = 9354,
                hidden_dim = 256
            ).to(device)

            # Set-up the optimizer and loss function
            optimizer = Adafactor(
                self.model.parameters(),
                lr = 3e-6, eps = (1e-30, 1e-3), clip_threshold = 1.0,
                decay_rate = -0.8, beta1 = None, weight_decay = 0.0,
                relative_step = False, scale_parameter = False, warmup_init = False,
            )
            loss_function = CrossEntropyLoss(ignore_index = 0)

            # Set-up lists to track the loss over the course of the training loop
            avg_loss = []
            total_train_loss, total_dev_loss = [], []
            min_dev_loss = np.inf
            
            # Loop for a number of epochs to ensure sufficient training on the data for the model
            for epoch in range(self.epochs):
                self.model.train()
                # Reset the training loss for each epoch
                train_loss = 0.0
                
                # Loop through the batches and update the model's weights. Similar process to the language models
                for i, batch in enumerate(train_loader):
                    src_data = batch["src_ids"].to(device)
                    tgt_input = batch["tgt_ids"][:, :-1].to(device)
                    tgt_output = batch["tgt_ids"][:, 1:].to(device)

                    output = self.model.forward_pass(src = src_data, tgt = tgt_input)
                    output_2d = output.permute(1,2,0)

                    # Reset the gradient and loss for each batch
                    optimizer.zero_grad()
                    loss = loss_function(output_2d, tgt_output)
                    train_loss += loss
                    loss.backward()
                    optimizer.step()

                    # Print updates to the terminal. Verbosity
                    if (i % 500 == 0):
                        print("Epoch: {} / {}, Batch: {} / {}, Loss: {}".format(epoch, self.epochs, i, len(train_loader), loss))

                avg_loss.append(train_loss / len(train_loader))

                # Empty the cache before validation loop
                torch.cuda.empty_cache()

                # Run the validation split through the model to gauge if the model is still learning.
                dev_loss = 0.0
                self.model.eval()
                for j, batch_v in enumerate(dev_loader):
                    src_data = batch_v["src_ids"].to(device)
                    tgt_input = batch_v["tgt_ids"][:, :-1].to(device)
                    tgt_output = batch_v["tgt_ids"][:, 1:].to(device)

                    with torch.no_grad():
                        pred = self.model.forward_pass(src = src_data, tgt = tgt_input)
                        pred = pred.permute(1,2,0)
                    
                    loss = loss_function(pred, tgt_output)
                    dev_loss += loss

                # Update at the end of an epoch
                print("End of epoch {}".format(epoch))
                print("The average training loss is: {}".format(train_loss / len(train_loader)))
                print("The average loss on the dev set is: {}".format(dev_loss / len(dev_loader)))
                if (min_dev_loss > dev_loss / len(dev_loader)):
                    print("The dev loss has decreased since the last epoch \n")
                    min_dev_loss = dev_loss / len(dev_loader)

                # Add the tracked loss to a list for saving
                total_train_loss.append(train_loss / len(train_loader))
                total_dev_loss.append(dev_loss / len(dev_loader))

            # Save the loss results to a .csv for tracking
            # ADD PATH TO RESULTS
            loss_results = pd.read_csv("/loss_results.csv", header = 0, index_col = None)
            to_add = {}
            to_add["model"] = [self.name]
            to_add["n"] = [self.n]
            to_add["ds"] = [self.ds]
            to_add["cl"] = [self.cl]
            to_add["train_loss"] = [", ".join(map(str, total_train_loss))]
            to_add["dev_loss"] = [", ".join(map(str, total_dev_loss))] 
            loss_results = pd.concat([loss_results, pd.DataFrame.from_dict(to_add)])
            # ADD PATH TO RESULTS
            loss_results.to_csv("loss_results.csv", header = True, index = False)

            # Save the trained model with pickle
            # ADD PATH TO SAVED MODELS
            with open("/" + self.name, "wb") as model_file:
                pickle.dump(self.model, model_file)

    def gen(self, test, seen_category):
        '''
        Class function used to generate predictions using the saved transformer model

        input:
        - test: the test dataset in PyTorch dataloader format
        '''
        if (seen_category == True):
            seen_category = test["seen_category"].unique()[0]

        # Set-up lists for the predictions and references
        predictions, references = [], []
        
        # Load the output vocabulary for this model
        # ADD PATH TO OUTPUT VOCABULARY
        with open("/" + self.ds + ".json") as vocab_file:
            output_vocab = json.load(vocab_file)

        # Set-up dataset for prediction
        test = TransformersDataset(
            test.input.values,
            test.output.values
        )
        # I never got the Transformer prediction function working for more than batch sizes of one
        test_loader = DataLoader(test, batch_size = 1, shuffle = False)

        self.model.to(device)
        self.model.eval()

        # Beam search algorithm.
        for i, batch in enumerate(test_loader):
            if (i % 100 == 0):
                print("Finished {} / {} batches.".format(i, len(test_loader)))

            src_data = batch["src_ids"].to(device)
            y_pred = torch.tensor([[1]], device = device).type_as(src_data)

            for l in range(80):
                out = self.model.forward_pass(src_data, y_pred)
                prob = F.log_softmax(out, dim = -1)[l, :, :]
                
                _, next_word = torch.max(prob, dim = 1)
                next_word = next_word.item()

                y_pred = torch.cat([y_pred, torch.ones(1,1).type_as(src_data.data).fill_(next_word)], dim = 1)

                if (next_word == 2):
                    break
            
            # Format the predicted sentences for comparison and evaluation
            sentence = " ".join([list(output_vocab.keys())[list(output_vocab.values()).index(t)] for t in y_pred[0] if (t not in [0,1,2])])
            predictions.append(sentence)
            references.append(batch["tgt_text"][0].split(" <N> "))
            
        # Pad any too short references
        for r, ref in enumerate(references):
            if (len(ref) < 5):
                references[r] = ref + [""] * (5 - len(ref))

        pred_ref = {"predictions": predictions, "references": references, "observed": seen_category}
        return pred_ref
    
    def ngram_eval(self, pred_ref):
        '''
        Similar ngram evaluation function to the language model. See there for further details
        '''
        self.model.eval()

        sacrebleu = load("sacrebleu")
        rouge = ROUGEScore()
        meteor = load("meteor")

        predictions = pred_ref["predictions"]
        references = pred_ref["references"]

        bleu = sacrebleu.compute(predictions = predictions, references = references,
                                 tokenize = "13a")
        print("\nThe corpus bleu score is:", bleu["score"],"\n")

        rouge_scores = rouge(predictions, references)
        print("These are the ROUGE score metrics:")
        print("rougeL F measure =", rouge_scores["rougeLsum_fmeasure"].item())
        print("rougeL precision =", rouge_scores["rougeLsum_precision"].item())
        print("rougeL recall =", rouge_scores["rougeLsum_recall"].item(),"\n")

        meteor_score = meteor.compute(predictions = predictions, references = references)
        print("The meteor score is:", meteor_score["meteor"],"\n")

        # ADD PATH TO RESULTS
        results = pd.read_csv("/webnlg_results.csv", header = 0, index_col = None)
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
        # ADD PATH TO RESULTS
        results.to_csv("/" + self.ds + ".csv", header = True, index = False)