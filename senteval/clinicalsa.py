# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import logging

from senteval.tools.validation import InnerKFoldClassifier


class ClinicalSAEval(object):
    logging.info('***** Transfer task :Vaccination Tweets Sentiment Analysis *****\n\n')
    def __init__(self, task_path, seed=1111):
        self.seed = seed
        self.data = self.loadFile(os.path.join(task_path, 'train.txt'))
        #self.n_samples = len(self.samples)
        #print(self.n_samples)

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        samples = self.data['X']
        return prepare(params, samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        data = {'X': [], 'y': []}
        tgt2idx = {'Positive': 0, 'Neutral': 1, 'Unrelated': 2,'NegSafety': 3, 'NegOthers': 4,'NegEfficacy': 5, 'NegCost': 6,'NegResistant': 7}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                try:
                  id,label,text = line.split('\t')
                  data['X'].append(text.split(' '))
                  data['y'].append(tgt2idx[label])
                except:
                  pass
        #print(len(data['X']),len(data['y']))
        return data

    def run(self, params, batcher):
        enc_input = []
        # Sort to reduce padding
        sorted_corpus = sorted(zip(self.data['X'], self.data['y']),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples = [x for (x, y) in sorted_corpus]
        sorted_labels = [y for (x, y) in sorted_corpus]
        logging.info('Generating sentence embeddings')
        for ii in range(0, len(self.data['X']), params.batch_size):
            batch = sorted_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            enc_input.append(embeddings)
        enc_input = np.vstack(enc_input)
        logging.info('Generated sentence embeddings')

        config = {'nclasses': 8, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = InnerKFoldClassifier(enc_input, np.array(sorted_labels), config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc}
