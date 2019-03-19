# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
BioContradiction : Recognizing Question Entailment for Medical Question Answering 
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io
from senteval.tools.validation import InnerKFoldClassifier
from sklearn.metrics import f1_score


class BIOCEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : BioContradiction *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'train.txt'))
        self.data = {'train': train}
        

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.data['train']['quest'] + \
                  self.data['train']['claim'] 
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = {'quest': [], 'claim': [], 'label': []}
        tgt2idx = {'NO': 0, 'YS': 1}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                try:
                  #print(text[0],"-",text[1],"-",text[2],"-",text[3])
                  data['quest'].append(text[0].split(' '))
                  data['claim'].append(text[1].split(' '))
                  data['label'].append(tgt2idx[text[2]])
                except:
                  pass
        return data

    def run(self, params, batcher):
        bioC_embed = {'train': {}}

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.data[key]['quest'],
                                       self.data[key]['claim'],
                                       self.data[key]['label']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))
            text_data['quest'] = [x for (x, y, z) in sorted_corpus]
            text_data['claim'] = [y for (x, y, z) in sorted_corpus]
            text_data['label'] = [z for (x, y, z) in sorted_corpus]
            for txt_type in ['quest', 'claim']:
                bioC_embed[key][txt_type] = []
                for ii in range(0, len(text_data['label']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    bioC_embed[key][txt_type].append(embeddings)
                bioC_embed[key][txt_type] = np.vstack(bioC_embed[key][txt_type])
            bioC_embed[key]['label'] = np.array(text_data['label'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainQ = bioC_embed['train']['quest']
        trainC = bioC_embed['train']['claim']
        trainQC = np.c_[np.abs(trainQ - trainC), trainQ * trainC]
        trainY = bioC_embed['train']['label']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = InnerKFoldClassifier(trainQC, trainY, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc}
