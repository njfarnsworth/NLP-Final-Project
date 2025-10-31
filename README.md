# NLP-Final-Project

This is the main gthub repository for Nayda and Judah's COSC 426 (NLP) Final Project, Fall 2025

In this project we evalaute how large language models handle the formal and mathematical logical notions of monotonicity and symmetricity in the context of natural language inference.


## Dataset Preparation

 - MoNLI: The pre-labeled training and testing data are available on github, the data is split between positive and negated sentences, and a test/train split is provided for the negated sentences, being the sentences of interest in the original paper. The files are currently stores as .jsonl, so simple conversions with a python script will be required to input them to NLPScholar. To complete the training set, closed source LLM's will be used to generate templated sentence pairs using examples from the original dataset as context. By limiting the scope of the LLM usage to templated generation, we gain the guarantees of templated synthetic data with greater data diversity than is feasible by hand-creation. This process will be handled by a python script calling the openAI api. Because the original data is pre-formatted for an NLI task, the processing the original data is fairly simple


 - Symmetricity:


 ### Analysis Overview

 - Independent Analysis: For each logical notion of interest, we will evaluate the 3 models (base, NLI finetune, our data finetune on base, our data finetune on NLI) on the test data using a text classification task on NLP Scholar. The analysis mode of NLPScholar will generate F1,recall, and precision metrics, and we will report accuracy using the results from evaluate (in part to compare to previous results with the MoNLI). The accuracy and F1 will be reported for each model.

 - Generalization Analysis: In the same approach as the independent analysis, except now we will compare the generalization performance of 2 models (base finetuned, NLI Finetuned) on data set from one logical task, and evaluate performance on the test dataset from the other task. The accuracy and F1 will be reported for each model

 (All data will be presented in tables, with a separate table for indepdnednt and generalization analysis)


