'''
this script helps to automate scp copying for the VM generated models
'''
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
from functools import reduce

"""
paths = [

["1500RNNnp_avg_val",
"/home/rb7e15/2.7v/summaries/1500samples/RNN/np_avg_val-5ops/total_num_epochs#17000~state_size#300~test_ratio#0.33~num_samples#1500~batch_size#100~learning_rate#0.01~epsilon#0.001~num_features#4~norm#True~clip#False~softmax_sat#100~state_fn#relu~smax_pen_r#0.0~augument_grad#True~relaunch#True~seed#14102"],

["1500RNNnp_center",
"/home/rb7e15/2.7v/summaries/1500samples/RNN/np_center-5ops/total_num_epochs#17000~state_size#300~test_ratio#0.33~num_samples#1500~batch_size#100~learning_rate#0.01~epsilon#0.001~num_features#4~norm#True~clip#False~softmax_sat#100~state_fn#relu~smax_pen_r#0.0~augument_grad#True~relaunch#True~seed#47364"],

["3500RNNnp_avg_val",
"/home/rb7e15/2.7v/summaries/RNN/np_avg_val-5ops/total_num_epochs#27000~state_size#300~test_ratio#0.33~num_samples#3500~batch_size#100~learning_rate#0.01~epsilon#0.001~num_features#4~norm#True~clip#False~softmax_sat#100~state_fn#relu~smax_pen_r#0.0~augument_grad#True~relaunch#True~seed#4369"],

["3500RNNnp_center",
"/home/rb7e15/2.7v/summaries/RNN/np_center-5ops/total_num_epochs#4000~state_size#300~test_ratio#0.33~num_samples#3500~batch_size#100~learning_rate#0.01~epsilon#0.001~num_features#4~norm#True~clip#False~softmax_sat#100~state_fn#relu~smax_pen_r#0.0~augument_grad#True~relaunch#True~seed#8859"],

["1500RLRNNnp_avg_val",
"/home/rb7e15/2.7v/summaries/1500samples/RLRNN/np_avg_val-5ops/total_num_epochs#80000~state_size#200~test_ratio#0.33~num_samples#1500~batch_size#100~learning_rate#0.005~epsilon#0.001~num_features#4~state_fn#relu~pen_sofmax#False~augument_grad#False~max_reward#1000~relaunch#False~seed#69690"],

["1500RLRNNnp_center",
"/home/rb7e15/2.7v/summaries/1500samples/RLRNN/np_center-5ops/total_num_epochs#3000~state_size#200~test_ratio#0.33~num_samples#1500~batch_size#100~learning_rate#0.005~epsilon#0.001~num_features#4~state_fn#relu~pen_sofmax#False~augument_grad#False~max_reward#1000~relaunch#True~seed#44082"],

["3500RLRNNnp_avg_val",
"/home/rb7e15/2.7v/summaries/RLRNN/np_avg_val-5ops/total_num_epochs#8000~state_size#200~test_ratio#0.33~num_samples#3500~batch_size#100~learning_rate#0.005~epsilon#0.001~num_features#4~state_fn#relu~pen_sofmax#False~augument_grad#False~max_reward#1000~relaunch#True~seed#26966"],

["3500RLRNNnp_center",
"/home/rb7e15/2.7v/summaries/RLRNN/np_center-5ops/total_num_epochs#8000~state_size#200~test_ratio#0.33~num_samples#3500~batch_size#100~learning_rate#0.005~epsilon#0.001~num_features#4~state_fn#relu~pen_sofmax#False~augument_grad#False~max_reward#1000~relaunch#True~seed#94097"]
]
"""
paths = [
["3500RLRNNnp_avgvalconv",
"/home/rb7e15/2.7v/summaries/RLRNN/np_avg_val-5ops/total_num_epochs#21000~state_size#200~test_ratio#0.33~num_samples#3500~batch_size#100~learning_rate#0.005~epsilon#0.001~num_features#4~state_fn#relu~pen_sofmax#False~augument_grad#False~max_reward#1000~relaunch#True~seed#35428"]
]
for elem in paths:
	cmd = "sshpass -f /home/user/.psw scp -r rb7e15@lyceum2.soton.ac.uk:"+elem[1]+" ./scp_pulled/"+elem[0]
	print(cmd)
	p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
	p.wait()
print("********")		
print("Finihsed")
