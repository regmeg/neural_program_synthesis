import os
import subprocess
import sys
import random
import time
import tensorflow as tf
import itertools
from collections import OrderedDict

'''
This module simulates the gridsearch funtionality, in order to tune the hyperparmaters
'''

tf.flags.DEFINE_integer("seed", int(round(random.random()*100000)), "the global simulation seed for np and tf")
FLAGS = tf.flags.FLAGS

def gen_cmd(cfg_dict, seed):
    string = "python3 ./model.py"
    name = " --name="
    for key, val in cfg_dict.items():
        string += " --"+str(key)+"="+str(val)
        if key == 'max_output_ops' or key == 'train_fn' or key == 'model': continue
        name += str(val)+"_"+str(key)+"-"
    name = name[:-1]
    seed  = " --seed="+str(seed)
    return string + seed + name

#cfg to iterate over
'''
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

params=OrderedDict()
params['state_size'] = [20, 50, 200]
params['num_samples'] = [1500]
params['batch_size']  = [100]
params['learning_rate'] = [0.005]
params['grad_norm'] = [10e2, 10e3]
params['max_output_ops'] = [5]
params['num_features'] = [3]
params['train_fn'] = ["np_mult"]
params['model'] = ["RNN"]
params['norm'] = [True]
#cfg which unlinkely is going to be iterated, but still can be configured

#seed
seed = FLAGS.seed
print (params)
#for n in range(len(cfg)):
cfg_dicts = [OrderedDict(zip(params, x)) for x in itertools.product(*params.values())]

cmds = [gen_cmd(cdict, seed) for cdict in cfg_dicts]

for ind,cmd in enumerate(cmds):
    print("Lnch[" + str(ind+1) +"]: " + cmd)
    #subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)