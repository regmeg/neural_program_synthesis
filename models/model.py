import pprint
import os
import sys
from params import get_cfg
from rnn_base import RNN
from ops import Operations
from session import run_session

def main():
    #get the global configuration
    cfg = get_cfg()
    
    #craete log and dumpl globals
    try:
        os.makedirs('./summaries/' + cfg['dst'])
    except FileExistsError as err:
        print("Dir already exists")

    stdout_org = sys.stdout
    sys.stdout = open('./summaries/' + cfg['dst']  + '/log.log', 'w')
    print("###########Global dict is###########")
    pprint.pprint(globals(), depth=3)
    print("###########CFG dict is###########")
    pprint.pprint(cfg, depth=3)
    print("#############################")
    #sys.stdout = stdout_org

    #instantiate containter with the operations avail for the selection
    ops = Operations(cfg)
    # instanitae the model
    model = RNN(cfg, ops)
    #run the tensorflow session with the selectted model
    run_session(model, cfg)

if __name__ == "__main__":
    main()