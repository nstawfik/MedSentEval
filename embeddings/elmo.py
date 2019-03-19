from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import torch
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser(description='Flair Embeddings')

parser.add_argument("--data_path", type=str, default='./data', help="Path to data (default ./data)")
parser.add_argument("--nhid", type=int, default=0, help="number of hidden layers: 0 for Logistic Regression or >0 for MLP (default 0)")
parser.add_argument('--tasks', nargs='+', default=['BIOSSES', 'ClinicalSTS', 'PICO' ,'PUBMED20K','RQE','MEDNLI','ClinicalSTS2'] ,help="Bio Tasks to evaluate (default [BIOSSES ClinicalSTS PICO PUBMED20K RQE MEDNLI RQE] )")
parser.add_argument('--model', type=str, choices=['original','small','pubmed'],default= 'original',help="ELMO Model (default original)")
params, _ = parser.parse_known_args()
# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = params.data_path
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.info("-------------------------------------ELMO [ALLENNLP]-------------------------------------"+"\nPATH_TO_DATA: " + str(PATH_TO_DATA) +"\nELMO_model: "+ str(params.model)+"\nTASKS: "+ str(params.tasks))

nhid=params.nhid

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 

#Load ELMO model
if (params.model=='pubmed'):
	options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
	weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'
elif (params.model=='small'):
	options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
	weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
elif  (params.model=='original'):
	options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
	weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

elmo_encoder = ElmoEmbedder(options_file,weight_file,cuda_device=0)
params_senteval['elmo'] = elmo_encoder
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam','batch_size': 64, 'tenacity': 5,'epoch_size': 4}
def s_embedding(word_embeds, rule='MEAN'):
    '''
    defines the type of sentence embedding
    @param word_embeds: word embeddings - np array of arrays
    @param rule: type of sentence embedding
    @return sentence_embedding
    '''
    if rule == 'MEAN':
        return np.mean(word_embeds, axis=0)

    if rule == 'SUM':
        return np.sum(word_embeds, axis=0)

    return 0

def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    return


def batcher(params, batch):
    """
    
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    #print(batch)

       
    #for elmo_embedding in params.elmo.embed_sentences(batch):  
    for elmo_embedding in params_senteval['elmo'].embed_sentences(batch):  
        # Average the 3 layers returned from ELMo #1024
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)
        
        #concatenate the 3 layers returned from ELMo #3072
        comb_elmo_embedding = np.concatenate(elmo_embedding, axis=1)
        
        mowe_elmo=np.mean(comb_elmo_embedding, axis=0)   
        embeddings.append(mowe_elmo)
        
    embeddings = np.vstack(embeddings)
    return embeddings
  





if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks=[]
    for i in params.tasks:
        transfer_tasks.append(i)
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
    print("\n")
