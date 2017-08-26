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

tf.flags.DEFINE_integer("seed", 0, "the global simulation seed for np and tf")
FLAGS = tf.flags.FLAGS

def gen_cmd(cfg_dict, seed):
    string = "python3 ./model.py"
    name = " --name="
    for key, val in cfg_dict.items():
        if key == 'grad_clip_val':
            string += " --"+str(key)+"_min="+str(val[0])
            string += " --"+str(key)+"_max="+str(val[1])
            name += str(key)+"_"+str(val[0])+"_"+str(val[1])+"-"
        else:
            string += " --"+str(key)+"="+str(val)
            if key == 'max_output_ops' or key == 'train_fn' or key == 'model': continue
            name += str(key)+"_"+str(val)+"-"
    if seed == 0: seed = int(round(random.random()*100000))
    name += "seed_" + str(seed)
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
params['state_size'] = [200]
params['num_samples'] = [1500]
params['batch_size']  = [100]
params['learning_rate'] = [0.001]
params['epsilon'] = [1e-3]
#params['loss_weight'] = [0.5]
params['max_output_ops'] = [5]
params['num_features'] = [10]
params['train_fn'] = ["np_avg_val"]
#params['model'] = ["HistoryRNN", "RNN"]
params['model'] = ["RNN"]
params['norm'] = [True]
#params['grad_norm'] = [10e2]
#params['softmax_sat'] = [1.0, 100.0, 1000.0]
params['softmax_sat'] = [1]
params['clip'] = [False]
#params['grad_clip_val'] = [[-10e2, 10e2],[-10e3, 10e3], [-10e4, 10e4]]
params['share_state'] = [True]
params['rnns_same_state'] = [False]
#params['state_fn'] = ["relu", "tanh"]
params['state_fn'] = ["relu"]
params['pen_sofmax'] = [True]
params['smax_pen_r'] = [0.05]
params['loss_swap_per'] = [30, 50, 80]
#params['test_ratio'] = [0.5]
#cfg which unlinkely is going to be iterated, but still can be configured

#seed
seed = FLAGS.seed
#for n in range(len(cfg)):
cfg_dicts = [OrderedDict(zip(params, x)) for x in itertools.product(*params.values())]

cmds = [gen_cmd(cdict, seed) for cdict in cfg_dicts]

for ind,cmd in enumerate(cmds):
    print("Lnch[" + str(ind+1) +"]: " + cmd)
    subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)