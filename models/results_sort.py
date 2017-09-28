'''
This script sorts the result logs produced by the gridserach
'''
##python2

import time
import os
import subprocess
import decimal 
import argparse
import re

def error_handl(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser()
parser.add_argument("path", help="parent dir of the dirs with lgos")
parser.add_argument("mode", help="use tail or cat - to sort all or last")
parser.add_argument("key", help="Hardmax, Softmax")

args = parser.parse_args()

com = ""
if   args.mode == 'tail': com = "tail -c4000 "
elif args.mode == 'cat':  com = "cat "
else:
    assert False, "mode should be cat or tail, args.mode: %r" % args.mode

logs_cmd = "ls "+args.path+"/*/log.log"
logs = subprocess.Popen(logs_cmd, shell=True, stdout=subprocess.PIPE,)
logs_stdout = logs.communicate()[0]
logs_list = logs_stdout.split('\n')

logs = []
for log in logs_list:
    if (log != "" and log is not None):
        groups = re.search(r'(?:\/)(.*)(?:\/)(.*)(?:\/)', log).groups()
        logs.append((groups[0], groups[1], log))
        
floats = []
for log in logs:
    
    cmd = com+log[2]+" | grep '"+args.key+"' | sed 's/.*"+args.key+"\s*//' | sed  's/(.*//' | uniq"

    print "cmd is"
    print cmd

    numbers = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,)
    stdout = numbers.communicate()[0]
    stdout_list = stdout.split('\n')
    model = log[1]
    local_floats = [(float(num), model) for num in stdout_list if error_handl(num)]
    floats = floats + local_floats

floats = sorted(floats, key=lambda log: log[0], reverse = True)

model = ""
models = []
for val in floats:
    if val[1] != model: 
        model = val[1]
        print model+":"
        if model in models: models.remove(model)
        models.append(model)
    print str(val[0])

print "models in decesding error val are"
for modell in models:
    print modell


    
    