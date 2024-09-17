import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
from evaluate import load
from torchmetrics.text.rouge import ROUGEScore

def webnlgCategories():
    sacrebleu = load("sacrebleu")
    rouge = ROUGEScore()
    meteor = load("meteor")

    test = pd.read_csv("(load_references_from_ds)", header = 0, index_col = None)
    test = test[test["no_ref"] == False].reset_index()
    with open("text/(load_model_predictions)", "r") as file:
        pred_ref = json.load(file)

    bleu_scores, rouge_scores, meteor_scores = [], [], []
    categories = list(test.category.unique())
    y_pos = np.arange(len(categories))
    
    for c in categories:
        idx = test[test.category == c].index.tolist()
        predictions = [pred_ref["predictions"][i] for i in idx]
        references = []
        for i in idx:
            references.append(pred_ref["references"][i] + [""] * (5 - len(pred_ref["references"][i])))

        bleu_scores.append(
            round(sacrebleu.compute(predictions = predictions,
                              references = references,
                              tokenize = "13a")["score"], 2)
                    )
        rouge_scores.append(
            round(100 * rouge(predictions, 
                              references)["rougeLsum_fmeasure"].item(), 2)
        )

        meteor_scores.append(
            round(100 * meteor.compute(predictions = predictions,
                                       references =  references)["meteor"], 2)
        )
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, sharex = True)
    ax1.barh(y_pos, bleu_scores, color = "#C2E8F7")
    ax2.barh(y_pos, rouge_scores, color = "#FFE2BB")
    ax3.barh(y_pos, meteor_scores, color = "#CCE7CF")
    
    ax1.set_yticks(y_pos, labels = categories)
    ax1.set_ylabel("Categories")

    ax1.set_xlabel("BLEU")
    ax2.set_xlabel("ROUGE")
    ax3.set_xlabel("METEOR")

    plt.savefig("figures/(name_of_file_comparing_webnlg_categories).pdf", dpi = 300.0, format = "pdf", bbox_inches = "tight", pad_inches = 0.0)

def loss_plot():
    df = pd.read_csv("(path_to_file_with_training_loss)/loss_results.csv", header = 0, index_col = None)
    webnlg_model_names = ["t5_webnlg_vanilla", "t5_webnlg_erase_tgt", "t5_webnlg_sub_tgt", "t5_webnlg_paraphrase"]
    webnlg_to_strategy = {
        "t5_webnlg_vanilla": "None",
        "t5_webnlg_erase_tgt": "Random Erasing",
        "t5_webnlg_sub_tgt": "Lexical Substitution",
        "t5_webnlg_paraphrase": "Paraphrasing"
    }
    e2e_model_names = ["t5_e2e_vanilla", "t5_e2e_erase_tgt", "t5_e2e_sub_tgt", "t5_e2e_paraphrase"]
    e2e_to_strategy = {
        "t5_e2e_vanilla": "None",
        "t5_e2e_erase_tgt": "Random Erasing",
        "t5_e2e_sub_tgt": "Lexical Substitution",
        "t5_e2e_paraphrase": "Paraphrasing"
    }

    colours = ["#C2E8F7", "#FFE2BB", "#CCE7CF", "#F2F4C1"]

    webnlg = df[df.model.isin(webnlg_model_names)]
    e2e = df[df.model.isin(e2e_model_names)]

    epochs = [1, 2, 3]

    webnlg_y = np.array([l.split(", ") for l in webnlg.dev_loss]).astype(float)
    e2e_y = np.array([l.split(", ") for l in e2e.dev_loss]).astype(float)

    fig, (ax1, ax2) = plt.subplots(1,2, sharex = True)
    for i in range(4):
        ax1.plot(epochs, webnlg_y[i, :], label = webnlg_to_strategy[webnlg.iloc[i].model], color = colours[i])
        ax2.plot(epochs, e2e_y[i, :], label = e2e_to_strategy[e2e.iloc[i].model], color = colours[i])

    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")

    ax1.set_title("WebNLG")
    ax2.set_title("E2E")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,1,2,0]

    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = "upper left", fontsize = "8")

    plt.savefig("figures/(name_of_file_to_save_loss).pdf", dpi = 300.0, format = "pdf", bbox_inches = "tight", pad_inches = 0.0)

loss_plot()