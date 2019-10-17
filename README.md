# MedSentEval
## Introduction
Motivated by the work of Conneau et al. [1], in their efforts to evaluate sentence representations in a fair and structured approach, this projects aims at replicating their evaluations in a domain-specific settings. We aimed to build on their work and adapt it to fit medical and clinical purposes, we also integrated extra embedding techniques not available in the original toolkit such as ELMO and BERT. 
Our goal is to evaluate text representations algorithms intended mainly for general purpose English language, and evaluate their transferability to biomedical domain tasks. We also experiment with the same models after pre-training them on published articles from PubMed. We use embeddings as input features for solving various downstream problems of biomedical computational linguistics. We hope that our in-depth evaluations along with the toolkit will benefit the BioNLP community in selecting suitable embeddings for different application tasks. 
Codes and experiments in this repository are based on the original SentEval code with adapttaion to the biomedical domain. 

** Source code for Tawfik and Spruit Manuscript "E********* S******* R************** f** B********* T***:  M****** a** E*********** R******". (under review as of April 2019).

[1] Conneau, Alexis, and Douwe Kiela. "Senteval: An evaluation toolkit for universal sentence representations." arXiv preprint arXiv:1803.05449 (2018).
## Installation
'''
!git clone https://github.com/nstawfik/MedSentEval
%cd ./MedSentEval
!python setup.py install
'''

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
| Embedding Model | Description  | Source |
| ------------- | ------------- |------------- |  
| GloVE  | Common Crawl  | http://nlp.stanford.edu/data/glove.840B.300d.zip | 
| GloVe_PubMed | 2016 PubMed  | https://slate.cse.ohio-state.edu/BMASS/PubMed_Glove.bin | 
| FastText  | Common Crawl  | https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip | 
| FastText_PubMed | 2016 PubMed  | http://www.llwang.net/pubmed_fasttext/pubmed_noncomm_fasttext_model.vec.tar.gz | 
| BERT |   |https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip |
| ELMo | Wikipedia and news crawl data | https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 <br /> https://s3-us-west2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json |
| ELMO_PubMed | PubMed| https://s3-us-west2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5 <br /> https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json |
| BioBERT | PubMed| |
| SciBERT | biomedical and computer sciences | https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz |
| Flair | web, Wikipedia, and Subtitles for the English language | https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt <br /> https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt |
| Flair_PubMed| PubMED | https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/pubmed-2015-fw-lm.pt <br />      https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/pubmed-2015-bw-lm.pt |
|InferSent | SNLI | https://dl.fbaipublicfiles.com/infersent/infersent2.pklhttps://dl.fbaipublicfiles.com/infersent/infersent2.pkl |
|Bio_InferSent | MedNLI |  |
|USE | Wikipedia, news, questions/answers, SNLI| https://tfhub.dev/google/universal-sentence-encoder-large/3 |






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
<!---%Please considering citing [1] if using this code for evaluating sentence embedding methods.--->

<!---%SentEval: An Evaluation Toolkit for Universal Sentence Representations--->
<!---%[1] A. Conneau, D. Kiela, SentEval: An Evaluation Toolkit for Universal Sentence Representations--->

<!---@article{conneau2018senteval,--->
  <!---title={SentEval: An Evaluation Toolkit for Universal Sentence Representations},--->
  <!---author={Conneau, Alexis and Kiela, Douwe},--->
  <!---journal={arXiv preprint arXiv:1803.05449},--->
 <!---year={2018}--->
<!---}--->

