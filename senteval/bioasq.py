# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
RQE : Recognizing Question Entailment for Medical Question Answering 
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score


class RQEEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : RQE *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'rqe_train.txt'))
        test = self.loadFile(os.path.join(task_path,
                             'rqe_test.txt'))
        
        self.rqe_data = {'train': train, 'test': test}
        

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.rqe_data['train']['chq'] + \
                  self.rqe_data['train']['faq'] + \
                  self.rqe_data['test']['chq'] + self.rqe_data['test']['faq']
        return prepare(params, samples)

    def loadFile(self, fpath):
        rqe_data = {'chq': [], 'faq': [], 'label': [],'pid':[]}
        tgt2idx = {'false': 0, 'true': 1}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                
                try:
                  #print(text[0],"-",text[1],"-",text[2],"-",text[3])
                  rqe_data['faq'].append(text[3].split(' '))
                  rqe_data['chq'].append(text[2].split(' '))
                  rqe_data['label'].append(tgt2idx[text[1]])
                  rqe_data['pid'].append(text[0])
                except:
                  pass
        return rqe_data

    def run(self, params, batcher):
        rqe_embed = {'train': {}, 'test': {}}

        for key in self.rqe_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.rqe_data[key]['chq'],
                                       self.rqe_data[key]['faq'],
                                       self.rqe_data[key]['label'],
                                       self.rqe_data[key]['pid']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))
            text_data['chq'] = [x for (x, y, z, w) in sorted_corpus]
            text_data['faq'] = [y for (x, y, z, w) in sorted_corpus]
            text_data['label'] = [z for (x, y, z, w ) in sorted_corpus]
            text_data['pid'] = [w for (x, y, z, w ) in sorted_corpus]
            for txt_type in ['chq', 'faq']:
                rqe_embed[key][txt_type] = []
                for ii in range(0, len(text_data['label']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    rqe_embed[key][txt_type].append(embeddings)
                rqe_embed[key][txt_type] = np.vstack(rqe_embed[key][txt_type])
            rqe_embed[key]['label'] = np.array(text_data['label'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainC = rqe_embed['train']['chq']
        trainF = rqe_embed['train']['faq']
        trainCF = np.c_[np.abs(trainC - trainF), trainC * trainF]
        trainY = rqe_embed['train']['label']

        # Test
        testC = rqe_embed['test']['chq']
        testF = rqe_embed['test']['faq']
        testCF = np.c_[np.abs(testC - testF), testC * testF]
        testY = rqe_embed['test']['label']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainCF, 'y': trainY},
                              test={'X': testCF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for RQE.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainCF), 'ntest': len(testCF)}

