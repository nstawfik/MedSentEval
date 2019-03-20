from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import torch
import keras
import os

import random
import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.embeddings import DocumentPoolEmbeddings
import argparse

parser = argparse.ArgumentParser(description='Flair Embeddings')

parser.add_argument("--data_path", type=str, default='./data', help="Path to data (default ./data)")
parser.add_argument('--embeddings', '-flair', nargs='+', default=['mix-forward', 'mix-backward'],help="Flair models to be used default ('mix-forward', 'mix-backward')")
parser.add_argument("--nhid", type=int, default=0, help="number of hidden layers: 0 for Logistic Regression or >0 for MLP (default 0)")
parser.add_argument('--tasks', nargs='+', default= ['BioC','CitationSA','ClinicalSA','BioASQ','PICO','PUBMED20K','RQE','ClinicalSTS','BIOSSES','MEDNLI'] ,help="Bio Tasks to evaluate (default ALL TASKS)")
parser.add_argument("--folds", type=int, default=10, help="number of k-folds for cross validations(default 10)")
params, _ = parser.parse_known_args()

print(params)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = params.data_path


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.info("-------------------------------------FLAIR MODEL-------------------------------------"+"\nPATH_TO_DATA: " + str(PATH_TO_DATA) +"\nFlair model: "+ str(params.embeddings)+"\nTASKS: "+ str(params.tasks))



# import senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': params.folds}


f=[]
for i in params.embeddings:
  f.append(FlairEmbeddings(i))
  #f.append(eval(i))
flair_encoder = DocumentPoolEmbeddings(f)
params_senteval['flair'] = flair_encoder
print(params_senteval['flair'])

nhid=params.nhid
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam','batch_size': 64, 'tenacity': 5,'epoch_size': 4}


def prepare(params, samples):
       
    return


def batcher(params, batch):
    """
    """
    embeddings = []
    sentences=[]
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    for sent in batch:
      sentence = Sentence(' '.join(w for w in sent))
      sentences.append(sentence)
      
    #print(batch)
    params_senteval['flair'].embed(sentences)
    #flair_encoder.embed(sentences)
      
    for sent in  sentences: 
        embeddings.append(sent.embedding.numpy())
        
    embeddings = np.vstack(embeddings)
    return embeddings
  


if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = params.tasks
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
