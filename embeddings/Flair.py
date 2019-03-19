from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import torch
import keras
import os
#from allennlp.commands.elmo import ElmoEmbedder
import random
import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.embeddings import DocumentPoolEmbeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
import argparse

parser = argparse.ArgumentParser(description='Flair Embeddings')

parser.add_argument("--data_path", type=str, default='./data', help="Path to data folder")
parser.add_argument('--embeddings', '-flair', nargs='+', default=[FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')],help="Types of embeddin to be used")
parser.add_argument("--nhid", type=int, default=0, help="number of hidden layers: 0 for Logistic Regression or >0 for MLP")

params, _ = parser.parse_known_args()

print(params)
logging.getLogger("flair").disabled=True
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

logging.info("FLAIR MODEL [https://github.com/zalandoresearch/flair]")


# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = params.data_path


# import SentEval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}


f=[]
for i in params.embeddings:
  #f.append(FlairEmbeddings(i))
  f.append(eval(i))
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
    transfer_tasks = ['BIOSSES','ClinicalSTS','MEDNLI']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
