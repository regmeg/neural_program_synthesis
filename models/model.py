import pprint
import os
import sys
import pickle
from params import get_cfg
from rnn_base import RNN
from rnn_sel_rnn import MemRNN
from ops import Operations
from session import run_session
from data_gen import samples_generator, split_train_test


def main():
    #get the global configuration
    cfg = get_cfg()
    
    #craete log and dumpl globals
    try:
        os.makedirs('./summaries/' + cfg['dst'])
    except FileExistsError as err:
        raise Exception('Dir already exists, saving resultsi n the same dir will result in unreadable graphs')
        
    pickle.dump( cfg, open( './summaries/' + cfg['dst']+  '/cfg.p', "wb" ) )


    stdout_org = sys.stdout
    sys.stdout = open('./summaries/' + cfg['dst']  + '/log.log', 'w')
    print("###########Global dict is###########")
    pprint.pprint(globals(), depth=3)
    print("###########CFG dict is###########")
    pprint.pprint(cfg, depth=3)
    print("#############################")
    #sys.stdout = stdout_org
    
    #dump cfg

    #instantiate containter with the operations avail for the selection
    ops = Operations(cfg)
    #generate data 
    x,y = samples_generator(cfg['train_fn'], (cfg['num_samples'], cfg['num_features']) , cfg['samples_value_rng'], cfg['seed'])
    x_train, x_test, y_train, y_test = split_train_test (x, y , cfg['test_ratio'])
    #instantiante the mem selection RNN
    mem = MemRNN(cfg, ops)
    # instanitae the model graph with the main OP selection RNN
    model = eval(cfg['model']+"(cfg, ops)")
    model.set_mem(mem)
    #run the tensorflow session with the selectted model
    run_session(model, cfg, x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()