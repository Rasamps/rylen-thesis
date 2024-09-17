# Data augmentation for text generation from structured data
**Foreword:**This is the main portion of code that I used in my Master's thesis at Simon Fraser University from 2021-2023. While some smaller analyses were done on an ad hoc basis, this can be used to reproduce the majority of the results and figures.

The thesis can be found in the Simon Fraser University library here: https://summit.sfu.ca/item/37861

## Abstract
Data-to-text generation, a subfield of natural language generation, increases the usability of structured data and knowledge bases. However, data-to-text generation datasets are not readily available in most domains and those that exist are arduously small. One solution is to include more data, though usually not a straightforward option. Alternatively, data augmentation consists of strategies which artificially enlarge the training data by incorporating slightly varied copies of the original data in order to diversify a dataset that is otherwise lacking. This work investigates augmentation as a remedy for training data-to-text generation models on small datasets. Natural language generation metrics are used to assess the quality of the generated text with and without augmentation. Experiments demonstrated that, with augmentation, models achieved equal performance despite the generated text exhibiting different properties. This suggests that data augmentation can be a useful step in training data-to-text generation models with limited data.

## Set-up
### Data availability
The datasets are not stored on GitHub due to repository size limits. However, both the WebNLG and E2E datasets are publicly available for download from their respective research pages and are easy to manage on your local system. For this research, the most recent versions of these datasets were used although depending on motivations you may want a different version. The most likely difference between versions is the formatting of the _source data_.

### Requirements
Although this repository contains a requirements.txt file, I will highlight some of the main Python libraries used in this work:
- PyTorch
- HuggingFace (transformers)
- Pickle
- Evaluate

Furthermore, between the data and saving any trained models, the code requires adequate memory. While I have tried my best to ensure everything is as efficient as possible, this research does involve large language models (LLMs) which are by definition costly.

### Getting started
There are a couple ways in which this code can be quickly set-up and ran.
1) Downloading the files to your local computer and calling run.py from the command line with the requisite arguments. The main caveat with this approach is this code was developed using a GPU and may take a long time without one which many local computers do not.
2) Load the code to a platform such as AWS or Azure with GPU support and modify run.py to work accordingly in the environment.

Furthermore, you'll want to quickly comb through the Python files to ensure all the directories and paths are properly re-written for your environment.

Lastly, a command line call of run.py might look something like this:
![GitHub Fig drawio](https://github.com/user-attachments/assets/d2ef0add-4420-426f-bbd4-639fe6b5cb13)

## Methodology
The following section will briefly highlight some of the methodology choices in this research.

### Models
The following repository offers implementations for the following three models, although given each one, variations can be quickly added due to the organization of the code. Each model is represented using a Class that offers auxiliary functionality for generating text, processing the _source data_ and _target text_ as well as handling the train-validation cycle of a model.

1) **T5:** this model is implemented using the HuggingFace API and follows the usage in their [documentation]([url](https://huggingface.co/docs/transformers/en/model_doc/t5)) wrapped up into an easy to call class. This was the best performing model of the three and warrants further investigation.
2) **Transformer:** this model is implemented following the architecture outlined in Vaswani et al. (2017) without additional variations except being implemented for data-to-text generation. This model was implemented using PyTorch primarily and is accompanied by datasets to confirm it functions correctly.
3) **Copy-based Approach:** this model largely acted as a baseline for comparison on n-gram metrics which can often be deceiving. This model removes any specific formatting within the _source data_ and generates a prediction based on the resulting _source data_ that resembles fragmented natural language.

### Augmentation strategies
In the research, we focused on three data augmentation strategies - progressively iterating them to what we felt could be plausible solutions. Two of the strategies only operate on the _target text_.

1) **Random erasing:** this augmentation strategy operates on both the _source data_ and _target text_ by randomly replacing segments of either one with a noisy sequence (random characters). This augmentation strategy offered the best improvements. The motivation of this strategy is to reinforce connections between related parts of the _source data_ and _target text_ for better natural language generation.
2) **Lexical substitution:** this augmentation strategy operates on the _target text_ by identifying potential words that do not affect the context and then replacing them with context-specific synonyms. Given BERT's adeptness for contextual embeddings, we can retrieve these synonyms from a standalone BERT model. This provides the data-to-text generation model with a more broad target space, hypothetically diversifying its output.
3) **Paraphrasing:** this augmentation strategy also operates on the _target text_ and provides a similar benefit of broadening the target space for more tailored natural language generation. By running the _target text_ through a secondary paraphrasing model, we generate contextually unchanged but new verbalizations of the _target text_.
