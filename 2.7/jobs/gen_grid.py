from __future__ import absolute_import
import os
import datetime
import io
import subprocess
import sys
import random
import time
import tensorflow as tf
import itertools
from collections import OrderedDict
import re
from itertools import izip

u'''
This module simulates the gridsearch funtionality, in order to tune the hyperparmaters
'''

tf.flags.DEFINE_integer(u"seed", 0, u"the global simulation seed for np and tf")
tf.flags.DEFINE_integer(u"amount", 1, u"num_of_sims_to_run")
tf.flags.DEFINE_string(u"type", u"RL", u"model type")

FLAGS = tf.flags.FLAGS


def gen_cmd(cfg_dict, seed):
    string = u"python /home/rb7e15/2.7v/model.py"
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
    #learning_rate = [0.01, 0.005],
    learning_rate = [0.01, 0.005, 0.0005],
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
    params[u'total_num_epochs'] = [5000, 21000]
    #params[u'total_num_epochs'] = [15]
    params[u'state_size'] = [300 for _ in range(FLAGS.amount)]
    params[u'test_ratio'] = [0.33]
    params[u'num_samples'] = [3500]
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
    params[u'total_num_epochs'] = [140000]
    #params[u'total_num_epochs'] = [15]
    params[u'state_size'] = [250 for _ in range(FLAGS.amount)]
    params[u'test_ratio'] = [0.33]
    params[u'num_samples'] = [150]
    params[u'batch_size']  = [10]
    params[u'learning_rate'] = [0.005]
    params[u'epsilon'] = [1e-3]
    params[u'max_output_ops'] = [5]
    params[u'num_features'] = [4]
    #params[u'train_fn'] = [u"np_avg_val", u"np_center"]
    params[u'train_fn'] = [ u"np_center"]
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
    #job_name
    start_time = datetime.datetime.now().strftime(u"%Y_%m_%d_%H%M%S")
    fname = "job_"+FLAGS.type+start_time+'_'+unicode(ind)
    #gen job launch scripts
    f = open(u'./'+fname, u'w')
    f.write('#!/bin/bash\n')
    f.write('\n')
    f.write('echo "source the env"\n')
    f.write('module load python\n')
    f.write('export CC=/home/rb7e15/gcc/bin/gcc\n')
    f.write('export LD_LIBRARY_PATH=/home/rb7e15/gcc/lib64:$LD_LIBRARY_PATH\n')
    f.write('source /home/rb7e15/2.7v/TFenv/bin/activate\n')
    f.write('\n')
    f.write('echo  "launch the command"\n')
    f.write(cmd)
    f.write('\n')


    #launch the job itself
    print u"Lnch[" + unicode(ind+1) +u"]: " + cmd

    #set wall time 
    num_epochs = re.search(r'(--total_num_epochs=)([0-9]*)( --)', cmd).groups()[1]
    run_time = int(0.6*int(num_epochs))            #2 seconds per epoch
    if run_time < 5*60: run_time = 5*60     #set min run time 5 min
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
  
    if h > 24:
       h, m, s = 24, 0, 0
    run_len = "%d:%02d:%02d" % (h, m, s)
  
    cmd_qsub = 'qsub -l walltime='+str(run_time)+' -l nodes=1:ppn=16 '+fname
    print cmd_qsub
    subprocess.Popen(cmd_qsub , shell=True, stderr=subprocess.STDOUT)
 
