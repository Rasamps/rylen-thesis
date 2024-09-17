# Import necessary libraries to perform the data augmentations
import string
import math
import re
import random
from time import time
import numpy as np
from collections import OrderedDict
import pandas as pd
import torch
from torch.nn import Softmax
from torch.nn import functional as F
from torch.nn import CosineSimilarity
from transformers import BertTokenizer, BertForMaskedLM, BartForConditionalGeneration, BartTokenizer
import spacy
import copy

device = "Please set-up your own CUDA device within your system."

# Include any additional models required for the more complex data augmentation
p_tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
p_model = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase").eval().to(device)
nlp = spacy.load("en_core_web_sm")
s_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", additional_special_tokens = [" [SEP] "])
s_model = BertForMaskedLM.from_pretrained("bert-base-uncased",
                                            return_dict = True,
                                            output_hidden_states = True,
                                            output_attentions = True).eval()
cos = CosineSimilarity(dim = 0)

'''
All of the data augmentation functions adhere to the following structure...
input:
- batch: the current batch in the model training loop to be augmented
- ds (optional): depending on the augmentation strategy, particular adaptations need to be made depending on the dataset

output:
- augmented batch: the augmented batch returns the source data and target text from the original batch along with augmented copies - in general
                   size(augmented batch) > size(batch)
'''

def erase_src(batch, ds):
    '''
    Takes the current batch and ~erases~ parts of the source data. See: https://arxiv.org/pdf/1708.04896

    The WebNLG and E2E dataset require different tokenization approaches to obtain erasable chunks.
    The erased chunks are replaced by random character segments of a similar length.
    '''
    # Lists to store the new batch in to return.
    src_to_add = []
    tgt_to_add = []
    if (ds == "webnlg"):
        for src, tgt in zip(batch["src_data"], batch["tgt_text"]):
            src = re.split('\[SEP\] | \| ', src)
            # Erase 33% of the source data
            for idx in random.sample(range(len(src)), math.ceil(len(src) / 3)):
                src[idx] = "".join(random.choices(string.ascii_lowercase, k = len(src[idx])))
            # Re-join the tokenized source data as seen in the original data
            src = " [SEP] ".join([src[i] + " | " + src[i+1] + " | " + src[i+2] for i in range(0, len(src), 3) if i < len(src) - 2])
            src_to_add.append(src)
            tgt_to_add.append(tgt)

    # The process for the E2E dataset is the same.
    else: #(ds == "e2e")
        reg = r'\[.*?\]'
        for src, tgt in zip(batch["src_data"], batch["tgt_text"]):
            src = src.split(", ")
            for idx in random.sample(range(len(src)), math.ceil(len(src) / 3)):
                src[idx] = re.sub(reg, "[" + "".join(random.choices(string.ascii_lowercase, k = len(src[idx]))) + "]", src[idx])
            src = ", ".join(src)
            src_to_add.append(src)
            tgt_to_add.append(tgt)

    # Add the augmented data to the original batch then return.
    batch["src_data"] += src_to_add
    batch["tgt_text"] += tgt_to_add

    return batch
    
def erase_tgt(batch):
    '''
    Takes the current batch and ~erases~ parts of the target text. See: https://arxiv.org/pdf/1708.04896 (same as last function)

    The WebNLG and E2E dataset require different tokenization approaches to obtain erasable chunks.
    The erased chunks are replaced by random character segments of a similar length.
    '''
    # Lists to store the new batch in to return.
    src_to_add = []
    tgt_to_add = []

    # As the target text is natural language across both datasets, one solution for random erasing works for both.
    for src, tgt in zip(batch["src_data"], batch["tgt_text"]):
        tgt = tgt.split()
        # More conservatively erase the target text (20%)
        for idx in random.sample(range(len(tgt)), math.ceil(len(tgt) / 5)):
            tgt[idx] = "".join(random.choices(string.ascii_lowercase, k = len(tgt[idx])))
        # Re-join the text.
        tgt = " ".join(tgt)
        src_to_add.append(src)
        tgt_to_add.append(tgt)

    # Add to the original batch and return to the training loop.
    batch["src_data"] += src_to_add
    batch["tgt_text"] += tgt_to_add

    return batch

def paraphrasing(batch):
    '''
    Takes the current batch and paraphrases the target text resulting in a new, unseen verbalization of the source data.
    While this "round-trip" translation is usually done into then out of another language, it was thought this might lose
    too much of the original context whereas paraphrasing would limit this loss: https://www.sciencedirect.com/science/article/pii/S2666651022000080

    As this data augmentation specifically focuses on the target text, there's no need to differentiate between datasets.
    '''
    texts = batch["tgt_text"]

    # Run the original target texts through the pre-trained paraphrasing model.
    encoding = p_tokenizer.batch_encode_plus(texts, padding = "longest", return_tensors = "pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # Generate up to three new paraphrased versions of the target text
    outputs = p_model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256, do_sample=True, top_k=120, top_p=0.85,
            early_stopping=True, num_return_sequences = 3
        )
    outputs = p_tokenizer.batch_decode(outputs, skip_special_tokens = True, clean_up_tokenization_spaces = True)

    # Perform a simple check to ensure that the paraphrased copies of the target text differ from the originals.
    for t in range(0, len(texts)):
        new_text = list(set(outputs[t * 3: t * 3 + 3]))
        new_text = [text for text in new_text if text not in texts]
        batch["tgt_text"] += new_text
        batch["src_data"] += np.repeat(batch["src_data"][t], len(new_text)).tolist()
    
    # We add to the batch then return it to the model training loop.
    return batch
    
def sub_tgt(batch):
    '''
    Takes the current batch and substitutes candidate words in the target text with synonyms selected from within BERT's contextual embeddings.
    Candidate words must:
    - Be a NOUN, ADJECTIVE, ADVERB, or NUMBER
    And the resulting synonyms must be:
    - Not the same word as the one it replaces
    - Have a cosine similarity of 0.9

    As this data augmentation specifically focuses on the target text, there's no need to differentiate between datasets.
    Ref: https://aclanthology.org/P19-1328/
    '''
    def check_candidate(new_ids, new_indices, old_tokens):
        '''
        check_candidate looks at the proposed synonyms and ensures they are not invalid. I.e. not punctuation that ruins a sentence,
        be a single character, or be the same word as before.

        input:
        - List of words converted to IDs
        - The index of the words in the sentence
        - The old tokens within the sentence
        '''
        # Loop through the list of IDs for the proposed synonyms
        for item in range(len(new_ids)):
            # Convert these IDs to their text-based tokens
            tokens = s_tokenizer.convert_ids_to_tokens(new_ids[item])
            # Initialize a list consisting of old tokens to be deleted
            to_delete = []
            # Looping through the tokens provided by the current proposed IDs
            for t, token in enumerate(tokens):
                token = token.strip("#")
                # Check if the token is punctuation, a single character, or a word in the old sentence
                if (token in string.punctuation or # Not punctuation
                len(token) == 1 or
                token in old_tokens[item] or
                ("##" + token) in old_tokens[item] or
                token == "."): # Not a single character
                    to_delete.append(t)
            new_indices[item] = np.delete(new_indices[item], to_delete)
            new_ids[item] = np.delete(new_ids[item], to_delete)
            # Return the retained candidate words
        return new_indices, new_ids

    # Using BERT, encode the original sentence into a sequence of IDs. Add the original tokens into the dictionary as well.
    encoding = s_tokenizer.batch_encode_plus(batch["tgt_text"], return_tensors = "pt", padding = True)
    encoding["tokens"] = [s_tokenizer.convert_ids_to_tokens(seq, skip_special_tokens = True) for seq in encoding["input_ids"]]
    
    # With SpaCy, tokenize the text again and check for tokens that have a part-of-speech of Noun, Adjective, Adverb, or Number.
    docs = [nlp(text) for text in batch["tgt_text"]]
    pos = [[str(token) for token in doc if (str(token.pos_) in ["NOUN","ADJ","ADVERB","NUM"])] for doc in docs]

    # The paper this methodology was pulled from indicated that for candidate words, we apply dropout to those positions in the sentence
    encoding["dropout_mask"] = [[0 if token.strip("##") in p else 1 for token in seq] for seq, p in zip(encoding["tokens"],pos)]
    # Collect the embeddings for each word in the sentence. Those with a dropout mask are replaced with zero-vectors which allows BERT to
    # generate the synonyms without prior influence.
    embeddings = torch.stack([s_model.get_input_embeddings()(id)[:, :] for id in encoding["input_ids"]])

    for mask in range(len(encoding["dropout_mask"])):
        for m in range(len(encoding["dropout_mask"][mask])):
            if (encoding["dropout_mask"][mask][m] == 0):
                embeddings[mask, m, random.sample(range(768), 768 // 3)] = 0.0

    # With no gradient adjustment, given the changed embeddings we reconstruct the sentence with possible candidates at the dropout sites.
    with torch.no_grad():
            dropout_output = s_model(inputs_embeds = embeddings, attention_mask = encoding["attention_mask"], token_type_ids = encoding["token_type_ids"])
            og_output = s_model(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"], token_type_ids = encoding["token_type_ids"])

    # Find the indices where dropout occurred.
    indices_to_swap = [np.where(np.array(encoding["dropout_mask"][t_i]) == 0)[0] for t_i in range(len(batch["tgt_text"]))]
    indices_to_swap = [indices - 1 for indices in indices_to_swap]  
    # Collect the top candidate provided by BERT for each word to be substituted and send it off to the check_candidate() function
    top_token_ids = [np.argsort(-dropout_output.logits[t_i, indices_to_swap[t_i], :].detach().numpy())[:, 1] for t_i in range(len(batch["tgt_text"]))]
    indices_to_swap, top_token_ids = check_candidate(top_token_ids, indices_to_swap, encoding["tokens"])
    # If the synonym is accepted the add the new word to the original sentence.
    top_tokens = [s_tokenizer.convert_ids_to_tokens(tokens) for tokens in top_token_ids]

    encoding_prime = encoding["input_ids"].clone()
    for t in range(len(batch["tgt_text"])):
        encoding_prime[t, indices_to_swap[t]] = torch.tensor(top_token_ids[t])

    with torch.no_grad():
        new_outputs = s_model(input_ids = encoding_prime, attention_mask = encoding["attention_mask"], token_type_ids = encoding["token_type_ids"])

    # Initialize lists to save the augmented batch to eventually return
    src_to_add = []
    tgt_to_add = []
    # For the augmented target text, we check if it is similar to the original using the Softmax as follows in the paper.
    for t, text in enumerate(batch["tgt_text"]):
        with torch.no_grad():
            softmax = Softmax(dim = 2)
            # See paper for reasoning and explicit formula. ------------------
            d_i = torch.cat([og_output["hidden_states"][-4:][state][t,:,:] for state in range(4)], dim = 1)
            d_prime_i = torch.cat([new_outputs["hidden_states"][-4:][state][t,:,:] for state in range(4)], dim = 1)
           
            attentions = og_output.attentions

            attention_stack = torch.cat([attentions[l][0,:,:,:] for l in range(12)], dim = 0)
            self_attention = torch.matmul(attention_stack, torch.transpose(attention_stack,1,2))
            self_attention = softmax(self_attention)

            if (len(indices_to_swap) > 0):
                changed = False
                for index_i, new_token in zip(indices_to_swap[t], top_tokens[t]):
                    ranking_sum = 0
                    for i in range(self_attention.size(-1)):
                        attn_avg = torch.mean(self_attention[:,i,:index_i + 1])
                        similarity = cos(d_i[i,:], d_prime_i[i, :])
                        ranking_sum += attn_avg * similarity
            # -------------------------------------------------------------------------------
                    if (ranking_sum >= 0.9): # If all checks pass then the augmented text is kept
                            changed = True
                            text = text.lower().replace(encoding["tokens"][t][index_i-1].strip("#"), new_token.strip("#"), 1)
                            encoding["tokens"][t][index_i - 1] = new_token.strip("#")

                # Add everything back to the batch to be return
                if (changed == True):
                    src_to_add.append(batch["src_data"][t])
                    tgt_to_add.append(" ".join(encoding["tokens"][t]).replace(" ##", ""))

    batch["src_data"] += src_to_add
    batch["tgt_text"] += tgt_to_add
    return batch