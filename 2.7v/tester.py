from __future__ import absolute_import
import pprint
import os
import sys
import numpy as np
from params import get_cfg
from rnn_base import RNN
from mem_sel_rnn import MemRNN
from NoEmbedRNN import OpSel
from NoEmbedRNN import MemSel
from NoEmbedRNN import RNN as oldRNN
from NoEmbedRNN import MemRNN  as oldMemRNN
from NoEmbedRNN import HistoryRNN
from rl_rnn import RLRNN
from rl_rnn_mem import RLRNNMEM
from ops import Operations
from session import *
from data_gen import *
import pickle
import tensorflow as tf
from io import open

tf.flags.DEFINE_string(u"path", u"", u"path to the summaries dir")

path = tf.flags.FLAGS.path
test_1000 = True
old = False

model_path = path+u'/model'
cfg_path = path+u'/cfg.p'

#get the global configuration
cfg = pickle.load(open(cfg_path, u'rb'))


#generate data 
x,y = samples_generator(cfg[u'train_fn'], (cfg[u'num_samples'], cfg[u'num_features']) , cfg[u'samples_value_rng'], cfg[u'seed'])
x_train, x_test, y_train, y_test = split_train_test (x, y , cfg[u'test_ratio'])

if cfg[u'model'] == u"RNN":
        ops = Operations(cfg)
        ops.ops = cfg[u"used_ops_obj"]
        ops.num_of_ops = len(ops.ops)
        if u'used_ops_obj_mem' in cfg:
                ops.ops_mem = cfg[u"used_ops_obj_mem"]
                ops.num_of_ops_mem = len(ops.ops_mem)
        #instantiante the mem selection RNN
        if old: mem = oldMemRNN(cfg, ops)
        else:   mem = MemRNN(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        if old: model = oldRNN(cfg, ops, mem)
        else:   model = RNN(cfg, ops, mem)
        res = restore_selection_matrixes2RNNS(model, cfg, x_train, x_test, y_train, y_test, model_path)
elif cfg[u'model'] == u"HistoryRNN":
        ops = Operations(cfg)
        ops.ops = cfg[u"used_ops_obj"]
        ops.num_of_ops = len(ops.ops)
        #instantiante the mem and op selection
        mem_sel = MemSel(cfg, ops)
        op_sel = OpSel(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        model = HistoryRNN(cfg, ops, mem_sel, op_sel)
        res = restore_selection_matrixes_HistoryRNNS(model, cfg, x_train, x_test, y_train, y_test, model_path)
elif cfg[u'model'] == u"RLRNN":
        ops_env = cfg[u"used_ops_env"]
        mem = RLRNNMEM(cfg, ops_env) 
        model = RLRNN(cfg, ops_env, mem) 
        res = restore_selection_RL_RNN(model, cfg, x_train, x_test, y_train, y_test, model_path, test_1000)
else:
        raise Exception(u'did not find the model')