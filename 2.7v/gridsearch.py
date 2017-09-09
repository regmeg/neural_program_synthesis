from __future__ import absolute_import
import os
import subprocess
import sys
import random
import time
import tensorflow as tf
import itertools
from collections import OrderedDict
from itertools import izip

u'''
This module simulates the gridsearch funtionality, in order to tune the hyperparmaters
'''

tf.flags.DEFINE_integer(u"seed", 0, u"the global simulation seed for np and tf")
tf.flags.DEFINE_string(u"type", u"RL", u"model type")

FLAGS = tf.flags.FLAGS

def gen_cmd(cfg_dict, seed):
    string = u"python ./model.py"
    name = u" --name="
    for key, val in cfg_dict.items():
        if key == u'grad_clip_val':
            string += u" --"+unicode(key)+u"_min="+unicode(val[0])
            string += u" --"+unicode(key)+u"_max="+unicode(val[1])
            name += unicode(key)+u"#"+unicode(val[0])+u"*"+unicode(val[1])+u"~"
        else:
            string += u" --"+unicode(key)+u"="+unicode(val)
            if key == u'max_output_ops' or key == u'train_fn' or key == u'model': continue
            name += unicode(key)+u"#"+unicode(val)+u"~"
    if seed == 0: seed = int(round(random.random()*100000))
    name += u"seed#" + unicode(seed)
    seed  = u" --seed="+unicode(seed)
    return string + seed + name

#cfg to iterate over
u'''
params=OrderedDict(
    state_size = [50, 100, 200],
    num_samples = [1500, 5000, 30000],
    batch_size  = [50, 100, 500],
    learning_rate = [0.01, 0.005],
    #learning_rate = [0.01, 0.005, 0.0005],
    grad_norm = [10e2],
    #grad_norm = [10e1, 10e2, 10e3],
    max_output_ops = [5],
    num_features = [3],
    train_fn = ["np_mult"],
    model = ["RNN"],
    norm = [True]
)
'''
#gridsearch for supervised models

if FLAGS.type == u"RNN":
    params=OrderedDict()
    params[u'total_num_epochs'] = [40000]
    params[u'state_size'] = [300, 300, 300, 300]
    params[u'test_ratio'] = [0.5]
    params[u'num_samples'] = [1500]
    params[u'batch_size']  = [100]
    params[u'learning_rate'] = [0.01]
    params[u'epsilon'] = [1e-3]
    params[u'max_output_ops'] = [5]
    params[u'num_features'] = [4]
    params[u'train_fn'] = [u"np_avg_val", u"np_center"]
    params[u'model'] = [u"RNN"]
    params[u'norm'] = [True]
    params[u'clip'] = [False]
    params[u'softmax_sat'] = [100]
    params[u'state_fn'] = [u"relu"]
    params[u'smax_pen_r'] = [0.0]
    params[u'augument_grad'] = [True]
    params[u'relaunch'] = [True]

elif FLAGS.type == u"RL":
    #cfg for RL models
    params=OrderedDict()
    params[u'total_num_epochs'] = [80000]
    params[u'state_size'] = [200, 200, 200, 200]
    params[u'test_ratio'] = [0.5]
    params[u'num_samples'] = [1500]
    params[u'batch_size']  = [100]
    params[u'learning_rate'] = [0.005]
    params[u'epsilon'] = [1e-3]
    params[u'max_output_ops'] = [5]
    params[u'num_features'] = [4]
    params[u'train_fn'] = [u"np_avg_val", u"np_center"]
    params[u'model'] = [u"RLRNN"]
    params[u'state_fn'] = [u"relu"]
    params[u'pen_sofmax'] = [False]
    params[u'augument_grad'] = [False]
    params[u'max_reward'] = [1000]
    params[u'relaunch'] = [True]
else:
    raise Exception(u'Wrong model specified to be run')

#cfg which unlinkely is going to be iterated, but still can be configured

#seed
seed = FLAGS.seed
#for n in range(len(cfg)):
cfg_dicts = [OrderedDict(izip(params, x)) for x in itertools.product(*params.values())]

cmds = [gen_cmd(cdict, seed) for cdict in cfg_dicts]

for ind,cmd in enumerate(cmds):
    print u"Lnch[" + unicode(ind+1) +u"]: " + cmd
    subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)