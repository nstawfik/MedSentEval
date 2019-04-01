# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import sys
from models import InferSent
import argparse

parser = argparse.ArgumentParser(description='InferSent Embeddings')

parser.add_argument("--data_path", type=str, default='./data', help="Path to data (default ./data)")
parser.add_argument('--embedding_path', type=str, default= './embeddings/glove/glove.840B.300d.txt',help="Path to embeddings (default ./embeddings/glove/glove.840B.300d.txt")
parser.add_argument('--model_path', type=str, default= './embeddings/infersent/infersent1.pkl',help="Path to InferSent model (default ./embeddings/infersent/infersent1.pkl")
parser.add_argument("--nhid", type=int, default=0, help="number of hidden layers: 0 for Logistic Regression or >0 for MLP (default 0)")
parser.add_argument('--tasks', nargs='+', default= ['BioC','CitationSA','ClinicalSA','BioASQ','PICO','PUBMED20K','RQE','ClinicalSTS','BIOSSES','MEDNLI'] ,help="Bio Tasks to evaluate (default ALL TASKS)")
parser.add_argument("--folds", type=int, default=10, help="number of k-folds for cross validations(default 10)")
parser.add_argument("--version", type=int, default=1, help="Infersent version(default 1)")
parser.add_argument("--usescikitlearn", action='store_false', default=True, help="Logistic regression from the scikit-learn (default Pytorch is used)")

params, _ = parser.parse_known_args()
# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = params.data_path
PATH_TO_W2V =  params.embedding_path
MODEL_PATH = params.model_path
V=params.version
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': params.usescikitlearn, 'kfold': params.folds}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.info("-------------------------------------INFERSENT MODEL-------------------------------------"+"\nPATH_TO_DATA: " + str(PATH_TO_DATA) +"\nPATH_TO_VEC: "+ str(PATH_TO_W2V)+"\nTASKS: "+ str(params.tasks))


nhid=params.nhid
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam','batch_size': 64, 'tenacity': 5,'epoch_size': 4}


#assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
#    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""
    
if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)

    params_senteval['infersent'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = params.tasks
    results = se.eval(transfer_tasks)
    print(results)
