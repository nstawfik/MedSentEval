# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division

import os
import sys
import logging
import tensorflow as tf
import tensorflow_hub as hub
tf.logging.set_verbosity(0)



# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.info("ELMO MODEL (params: Path to Data & Num of Hidden Layers[optional} ) ")
logging.info("\n\n\nPATH_TO_DATA: " + str(sys.argv[1])+ "\n\n")

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = sys.argv[1]  # '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# tensorflow session
session = tf.Session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['elmo'](batch)
    return embeddings

def make_embed_fn(module):
  with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    elmo = hub.Module(module,trainable=True)
    embeddings = elmo(sentences,signature='default',as_dict=True)["default"]
    session = tf.train.MonitoredSession()
  return lambda x: session.run(embeddings, {sentences: x})

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}

if (len(sys.argv)>3):
    nhid = int(sys.argv[3])
else:
    nhid=0

#params_senteval['classifier'] = {'nhid':nhid , 'optim': 'rmsprop', 'batch_size': 128,'tenacity': 3, 'epoch_size': 2}
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam','batch_size': 64, 'tenacity': 5,'epoch_size': 4}

# Start TF session and load Google Universal Sentence Encoder
encoder = make_embed_fn("https://tfhub.dev/google/elmo/2")


params_senteval['elmo'] = encoder

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MEDNLI','ClinicalSTS','BIOSSES','ClinicalSTS2']
    results = se.eval(transfer_tasks)
    print(results)
