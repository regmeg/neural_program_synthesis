import pprint
import os
import sys
import pickle
from params import get_cfg
from rnn_base import RNN
from mem_sel_rnn import MemRNN
from NoEmbedRNN import OpSel
from NoEmbedRNN import MemSel
from NoEmbedRNN import RNN as oldRNN
from NoEmbedRNN import MemRNN  as oldMemRNN
from NoEmbedRNN import HistoryRNN
from ops import Operations
from session import run_session_2RNNS, run_session_HistoryRNN, run_session_RL_RNN
from data_gen import samples_generator, split_train_test, OpsEnv
from rl_rnn import RLRNN
from rl_rnn_mem import RLRNNMEM

def main():
    #get the global configuration
    cfg = get_cfg()
    init_cfg = cfg

    if init_cfg['rerun_cfg'] != "":
        cfg_path = cfg['rerun_cfg']+'cfg.p'
        cfg = pickle.load(open(cfg_path, 'rb')) 
        cfg['name'] = init_cfg['name']
        cfg['logoff'] = init_cfg['logoff']
        cfg['dst'] = cfg['model'] + "/" + cfg['train_fn'].__name__ + "-" + str(cfg['max_output_ops']) +"ops/" + cfg['name']

    #instantiate containter with the operations avail for the selection
    ops = Operations(cfg)
    
    #save ops as obj
    if init_cfg['rerun_cfg'] != "":
        ops.ops = cfg["used_ops_obj"]
        ops.num_of_ops = len(ops.ops)
    
    cfg["used_ops_obj"] = ops.ops
    if cfg['model'] == "RLRNN":
        ops_env = OpsEnv(cfg)
        cfg["used_ops_env"] = ops_env

    #craete log and dumpl globals
    try:
        os.makedirs('./summaries/' + cfg['dst'])
    except FileExistsError as err:
        raise Exception('Dir already exists, saving resultsi n the same dir will result in unreadable graphs')
    
    #dump cfg        
    pickle.dump( cfg, open( './summaries/' + cfg['dst']+  '/cfg.p', "wb" ) )


    stdout_org = sys.stdout
    sys.stdout = open('./summaries/' + cfg['dst']  + '/log.log', 'w')
    print("###########Global dict is###########")
    pprint.pprint(globals(), depth=3)
    print("###########CFG dict is###########")
    pprint.pprint(cfg, depth=3)
    print("#############################")
    if cfg['logoff']:
        sys.stdout = stdout_org

    #generate data 
    x,y = samples_generator(cfg['train_fn'], (cfg['num_samples'], cfg['num_features']) , cfg['samples_value_rng'], cfg['seed'])
    x_train, x_test, y_train, y_test = split_train_test (x, y , cfg['test_ratio'])
    if cfg['model'] == "RNN":
        #instantiante the mem selection RNN
        mem = MemRNN(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        model = eval(cfg['model']+"(cfg, ops, mem)")
        run_session_2RNNS(model, cfg, x_train, x_test, y_train, y_test)
    elif cfg['model'] == "HistoryRNN":
        #instantiante the mem and op selection
        mem_sel = MemSel(cfg, ops)
        op_sel = OpSel(cfg, ops)
        # instanitae the model graph with the main OP selection RNN
        model = eval(cfg['model']+"(cfg, ops, mem_sel, op_sel)")
        run_session_HistoryRNN(model, cfg, x_train, x_test, y_train, y_test)
    elif cfg['model'] == "RLRNN":
        mem = RLRNNMEM(cfg, ops_env) 
        model = RLRNN(cfg, ops_env, mem) 
        run_session_RL_RNN(model, cfg, x_train, x_test, y_train, y_test)
    else:
        raise Exception('Wrong model specified to be run')

if __name__ == "__main__":
    main()