# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.rqe import RQEEval
from senteval.trec import TRECEval
from senteval.rct20k import RCT20KEval
from senteval.bioc import BIOCEval
from senteval.pico import PICOEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.clinicalsa import ClinicalSAEval
from senteval.citationsa import CitationSAEval
from senteval.sts import STS14Eval, STSBenchmarkEval,ClinicalSTSEval,BIOSSESEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.probing import *

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['BioC','CitationSA','ClinicalSA','BIOC','CHEMPROT','PICO','PUBMED20K','MEDNLI', 'RQE','STSBenchmark','ClinicalSTS','ClinicalSTS2','BIOSSES']
                       

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'CR':
            self.evaluation = CREval(tpath + '/downstream/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/downstream/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/downstream/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/downstream/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/downstream/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/downstream/SST/fine', nclasses=5, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/downstream/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/downstream/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'ClinicalSTS2':
            self.evaluation = STSBenchmarkEval(tpath + '/ClinicalSTS2', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/downstream/SNLI', seed=self.params.seed)
        elif name == 'MEDNLI':
            self.evaluation = SNLIEval(tpath + '/MEDNLI', seed=self.params.seed)
        elif name == 'RQE':
            self.evaluation = RQEEval(tpath + '/RQE', seed=self.params.seed)
        elif name == 'BioC':
            self.evaluation = BIOCEval(tpath + '/BIOC', seed=self.params.seed)   
        elif name == 'ClinicalSA':
            self.evaluation = ClinicalSAEval(tpath + '/ClinicalSA', seed=self.params.seed) 
        elif name == 'CitationSA':
            self.evaluation = CitationSAEval(tpath + '/CitationSA', seed=self.params.seed) 
        elif name == 'RCT20K':
            self.evaluation = RCT20KEval(tpath + '/ RCT20K', seed=self.params.seed)
        elif name == 'PICO':
            self.evaluation = PICOEval(tpath + '/PICO', seed=self.params.seed)
        elif name == 'CHEMPROT':
            self.evaluation = CHEMPROTEval(tpath + '/CHEMPROT', seed=self.params.seed)
        elif name in ['STS14']:
            fpath ='sts-en-test-gs-2014'
            self.evaluation = eval(name + 'Eval')(tpath + '/STS14' , seed=self.params.seed)
        elif name in ['ClinicalSTS']:
            self.evaluation = eval(name + 'Eval')(tpath + '/ClinicalSTS', seed=self.params.seed)
        elif name in ['BIOSSES']:
            self.evaluation = eval(name + 'Eval')(tpath + '/BIOSSES', seed=self.params.seed)                                   
                           
        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
