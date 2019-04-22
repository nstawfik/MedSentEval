# MedSentEval
## Introduction
Motivated by the work of Conneau et al. [1], in their efforts to evaluate sentence representations in a fair and structured approach, this projects aims at replicating their evaluations in a domain-specific settings. We aimed to build on their work and adapt it to fit medical and clinical purposes, we also integrated extra embedding techniques not available in the original toolkit such as ELMO and BERT. 
Our goal is to evaluate text representations algorithms intended mainly for general purpose English language, and evaluate their transferability to biomedical domain tasks. We also experiment with the same models after pre-training them on published articles from PubMed. We use embeddings as input features for solving various downstream problems of biomedical computational linguistics. We hope that our in-depth evaluations along with the toolkit will benefit the BioNLP community in selecting suitable embeddings for different application tasks. 
Codes and experiments in this repository are based on the original SentEval code with adapttaion to the biomedical domain. 

** Source code for Tawfik and Spruit Manuscript "E********* S******* R************** f** B********* T***:  M****** a** E*********** R******". (under review as of April 2019).

## Refrences
[1] Conneau, Alexis, and Douwe Kiela. "Senteval: An evaluation toolkit for universal sentence representations." arXiv preprint arXiv:1803.05449 (2018).
