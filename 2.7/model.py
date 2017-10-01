from __future__ import absolute_import
import pprint
import os
import sys
import pickle
import io
from params import get_cfg
from rnn_base import RNN
from mem_sel_rnn import MemRNN
from ops import Operations
from session import run_session_2RNNS, run_session_HistoryRNN, run_session_RL_RNN
from data_gen import samples_generator, split_train_test, OpsEnv
from rl_rnn import RLRNN
from rl_rnn_mem import RLRNNMEM
from io import open

def main():
    #get the global configuration
    cfg = get_cfg()
    init_cfg = cfg

    if init_cfg[u'rerun_cfg'] != u"":
        cfg_path = cfg[u'rerun_cfg']+u'cfg.p'
        cfg = pickle.load(open(cfg_path, u'rb')) 
        cfg[u'name'] = init_cfg[u'name']
        cfg[u'logoff'] = init_cfg[u'logoff']
        cfg[u'dst'] = cfg[u'model'] + u"/" + cfg[u'train_fn'].__name__ + u"-" + unicode(cfg[u'max_output_ops']) +u"ops/" + cfg[u'name']

    #instantiate containter with the operations avail for the selection
    ops = Operations(cfg)
    
    #save ops as obj
    if init_cfg[u'rerun_cfg'] != u"":
        ops.ops = cfg[u"used_ops_obj"]
        ops.num_of_ops = len(ops.ops)
    
    #cfg[u"used_ops_obj"] = ops.ops
    if cfg[u'model'] == u"RLRNN":
        ops_env = OpsEnv(cfg)
        #cfg[u"used_ops_env"] = ops_env

    #craete log and dumpl globals
    try:
        os.makedirs(u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'])
    except FileExistsError, err:
        raise Exception(u'Dir already exists, saving resultsi n the same dir will result in unreadable graphs')
    
    #dump cfg        
    pickle.dump( cfg, open( u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst']+  u'/cfg.p', u'wb' ) )


    stdout_org = sys.stdout
    log = io.open(u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst']  + u'/log.log', u'wb')
    sys.stdout =  log
  
    print u'###########CFG dict is###########'
    cfg_u = {unicode(k): unicode(v)  for k, v in cfg.items()}
    pprint.pprint(cfg_u, depth=3)
    print u'#############################'
    
    if cfg[u'logoff']:
        sys.stdout = stdout_org

    #generate data 
    x,y = samples_generator(cfg[u'train_fn'], (cfg[u'num_samples'], cfg[u'num_features']) , cfg[u'samples_value_rng'], cfg[u'seed'])
    x_train, x_test, y_train, y_test = split_train_test (x, y , cfg[u'test_ratio'])
    if cfg[u'model'] == u"RNN":
        #instantiante the mem selection RNN
        mem = MemRNN(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        model = eval(cfg[u'model']+u"(cfg, ops, mem)")
        run_session_2RNNS(model, cfg, x_train, x_test, y_train, y_test)
    elif cfg[u'model'] == u"HistoryRNN":
        #instantiante the mem and op selection
        mem_sel = MemSel(cfg, ops)
        op_sel = OpSel(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        model = eval(cfg[u'model']+u"(cfg, ops, mem_sel, op_sel)")
        run_session_HistoryRNN(model, cfg, x_train, x_test, y_train, y_test)
    elif cfg[u'model'] == u"RLRNN":
        mem = RLRNNMEM(cfg, ops_env) 
        model = RLRNN(cfg, ops_env, mem) 
        run_session_RL_RNN(model, cfg, x_train, x_test, y_train, y_test)
    else:
        raise Exception(u'Wrong model specified to be run')

if __name__ == u"__main__":
    main()
