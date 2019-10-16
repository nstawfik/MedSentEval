# MedSentEval
## Introduction
Motivated by the work of Conneau et al. [1], in their efforts to evaluate sentence representations in a fair and structured approach, this projects aims at replicating their evaluations in a domain-specific settings. We aimed to build on their work and adapt it to fit medical and clinical purposes, we also integrated extra embedding techniques not available in the original toolkit such as ELMO and BERT. 
Our goal is to evaluate text representations algorithms intended mainly for general purpose English language, and evaluate their transferability to biomedical domain tasks. We also experiment with the same models after pre-training them on published articles from PubMed. We use embeddings as input features for solving various downstream problems of biomedical computational linguistics. We hope that our in-depth evaluations along with the toolkit will benefit the BioNLP community in selecting suitable embeddings for different application tasks. 
Codes and experiments in this repository are based on the original SentEval code with adapttaion to the biomedical domain. 

** Source code for Tawfik and Spruit Manuscript "E********* S******* R************** f** B********* T***:  M****** a** E*********** R******". (under review as of April 2019).

## Included Tasks
 | Dataset  | Task  | Source  | Example |  Label  |
 | ------------- | ------------- |------------- | ------------------ | ------------- | 
 | MedNLI | Textual Entailment | Patient records | H1:During hospitalization , patient became progressively more dyspnic requiring BiPAP and then a NRB<br />P2:The patient is on room air.  | Contradiction | 
 | RQE | Question Entailment | Doctor questions  | Q1: What should I do with this patient whose biopsy report shows carcinoma in situ of the vulva? <br />Q2: What to do with this patient, biopsy shows carcinoma in situ of the vulva? | True |  
 | PUBMED20K | Sentence Classification | Medical articles | Text:Transient intraocular pressure elevation and cataract progression occurred.  |  Background
 | PICO | Sentence Classification | Medical articles | Text: Classes included CRC survivors and people with CVD.  | Intervention
 | PatientSA | Sentiment Analysis | Patient tweets | Text: Don't forget to also vaccinate your sons. It is potentially even more important. #HPV #vaccineswork  | Positive
 | CitationSA | Sentiment Analysis | Medical articles | Text: Patrek et al [C] examined 13 factors influencing uid drainage. | Neutral
 | BioASQ | Question Answering | Medical articles | Q:Is osteocrin expressed exclusively in the bone? <br />A:Evolution of Osteocrin as an activity-regulated factor in the primate brain. | No
 | BioC | Question Answering | Medical articles| Q:In women with pre-eclampsia, is mutation in renin-angiotensin gene associated with pre-eclampsia? <br />A:The variants(A{>C) of 1166 polymorphism site of AT1RG predisposes increased risk of PIH. | Yes
 | C-STS | Semantic Similarity | Patient records | S1: Use information was down loaded from the patient's PAP device and reviewed with the patient. <br /> S2:I discussed the indications, contraindications and side effects of doxycycline with the patient. | 0.5
 | BIOSSES | Semantic Similarity | Medical articles | S1: The oncogenic activity of mutant Kras appears dependent on functional Craf. <br /> S2: Oncogenic KRAS mutations are common in cancer. | 1

 ### Download datasets
 To get all the transfer tasks datasets, run (in data/):
```
./download_data.bash
```
The script will download all datasets that are publically avaiable without permission. For other data, fulfill the requirments to gain permission and put it under ./data/ 

## Embedding models
| Embedding  | Description  | Source |
| ------------- | ------------- |------------- | 

### Download Pre-trained models:
YOu can download state-of-the-art open-domain and biomedical embeddings individually to the ./embedding/ or run the following script
```
./download_embeddings.bash
```

## Usage Examples
In the ./examples folder we provide jupyter notebooks to evaluate soome embeddings models on different datasets. Modify 
## Add a New Sentence Encoder
As required by SentEval, this script implements two functions: prepare (optional) and batcher (required) that turn text sentences into sentence embeddings. Then SentEval takes care of the evaluation on the transfer tasks using the embeddings as features.

How to use SentEval: examples
examples/bow.py
In examples/bow.py, we evaluate the quality of the average of word embeddings.

<!---##References--->
%Please considering citing [1] if using this code for evaluating sentence embedding methods.

%SentEval: An Evaluation Toolkit for Universal Sentence Representations
%[1] A. Conneau, D. Kiela, SentEval: An Evaluation Toolkit for Universal Sentence Representations

%@article{conneau2018senteval,
  %title={SentEval: An Evaluation Toolkit for Universal Sentence Representations},
  %author={Conneau, Alexis and Kiela, Douwe},
  %journal={arXiv preprint arXiv:1803.05449},
 % year={2018}
%}

## Refrences
[1] Conneau, Alexis, and Douwe Kiela. "Senteval: An evaluation toolkit for universal sentence representations." arXiv preprint arXiv:1803.05449 (2018).
