from __future__ import absolute_import, division, unicode_literals,print_function 
import sys
import numpy as np
import logging
import sklearn
import torch
import keras
import os
import sys
import numpy as np
import random


import codecs
import collections
import json
import re
from utils import modeling
from utils import tokenization
import tensorflow as tf
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='Flair Embeddings')

parser.add_argument("--data_path", type=str, default='./data', help="Path to data (default ./data)")
parser.add_argument("--nhid", type=int, default=0, help="number of hidden layers: 0 for Logistic Regression or >0 for MLP (default 0)")
parser.add_argument('--tasks', nargs='+', default= ['BioC','CitationSA','ClinicalSA','BioASQ','PICO','PUBMED20K','RQE','ClinicalSTS','BIOSSES','MEDNLI'] ,help="Bio Tasks to evaluate (default ALL TASKS)")
parser.add_argument('--model_path', type=str, default= 'None',help="BERT Model directory (default None)")
parser.add_argument("--folds", type=int, default=10, help="number of k-folds for cross validations(default 10)")
parser.add_argument("--master", type=str, default=None, help="TPU address (default None)")
parser.add_argument("--bert_batch_size", type=int, default=32, help="Bert Batch Size (default 32)")
parser.add_argument("--usescikitlearn", action='store_false', default=True, help="Logistic regression from the scikit-learn (default Pytorch is used)")
parser.add_argument("--useTPU", action='store_true', default=False, help="Logistic regression from the scikit-learn (default Pytorch is used)")


params, _ = parser.parse_known_args()

PATH_TO_BERT= params.model_path

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = params.data_path


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
#logging.info("-------------------------------------BERT [GOOGLE]-------------------------------------"+"\nPATH_TO_DATA: " + str(PATH_TO_DATA) +"\nELMO_model: "+ str(params.model)+"\nTASKS: "+ str(params.tasks))

nhid=params.nhid

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch':params.usescikitlearn, 'kfold': params.folds,'batch_size':100000} #params.folds}

vocab_file=os.path.join(PATH_TO_BERT, 'vocab.txt')
bert_config_file=os.path.join(PATH_TO_BERT, 'bert_config.json')
init_checkpoint=os.path.join(PATH_TO_BERT, 'bert_model.ckpt')
layers=-1
max_seq_length=128
batch_size=params.bert_batch_size
do_lower_case=True
use_tpu=params.useTPU
master=params.master
num_tpu_cores=8
use_one_hot_embeddings=False
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam',  'batch_size': 64, 'tenacity': 5,'epoch_size': 4}



tf.logging.set_verbosity(tf.logging.INFO)

layer_indexes = [layers]

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

tokenizer = tokenization.FullTokenizer(
vocab_file=vocab_file, do_lower_case=do_lower_case)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
master=master,
tpu_config=tf.contrib.tpu.TPUConfig(
num_shards=num_tpu_cores,
per_host_input_for_training=is_per_host))
#####bert 





class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]


    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    #if ex_index < 5:
      #tf.logging.info("*** Example ***")
      #tf.logging.info("unique_id: %s" % (example.unique_id))
      #tf.logging.info("tokens: %s" % " ".join(
          #[tokenization.printable_text(x) for x in tokens]))
      #tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      #tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      #tf.logging.info(
       #   "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  for sent in input_file:
    sent=' '.join(w for w in sent)
    line = tokenization.convert_to_unicode(sent)
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
      text_a = line
    else:
      text_a = m.group(1)
      text_b = m.group(2)
    #print(text_a)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    unique_id += 1
  
  return examples
  
  

def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    #tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      #init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
  
  

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
    print("NewBatch", len(batch))
    embeddings = []
    examples = read_examples(batch)
    #print(examples)
	

    features = convert_examples_to_features(examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)
    #print(features)

    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature

 
    input_fn = input_fn_builder(
      features=features, seq_length=max_seq_length)

    for result in estimator.predict(input_fn, yield_single_examples=True):
      #print('NEW')
      sent_vec,tokens=[],[]
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      output_json = collections.OrderedDict()
      output_json["linex_index"] = unique_id
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
          token_vec=[round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
        
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        if token not in ['[SEP]','[CLS]'] :
          sent_vec.append(token_vec)
          tokens.append(token)
          #print(token,token_vec)
        
        all_features.append(features)
      sent_vecNP=np.array([np.array(xi) for xi in sent_vec])  
      mowe_bert=np.mean(sent_vecNP, axis=0)   
      print(sent_vecNP.shape,mowe_bert.shape)
      embeddings.append(mowe_bert)
        
      
      #all_features.append(sentVec)
      #output_json["features"] = all_features
      #writer.write(json.dumps(output_json) + "\n")
      #print(tokens,mowe_bert.shape)
    
    embeddings = np.vstack(embeddings)
    return embeddings


if __name__ == "__main__":
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=batch_size,
	  train_batch_size=1024)

  se = senteval.engine.SE(params_senteval, batcher, prepare)
  transfer_tasks=[]
  for i in params.tasks:
      transfer_tasks.append(i)
    # senteval prints the results and returns a dictionary with the scores
  results = se.eval(transfer_tasks)
  print(results)
