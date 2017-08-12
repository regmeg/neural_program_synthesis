'''
inject variables into the running simulations, so that they finish off quicker
works only with python2
'''
import time
import os
import subprocess
import pyrasite

process = subprocess.Popen("pgrep -f no_dist_model.py",
                             shell=True,
                             stdout=subprocess.PIPE,
                           )
stdout = process.communicate()[0]
stdout_list = stdout.split('\n')

for pid in stdout_list:
    print("##Injecting: ",pid)
    pyrasite.inject(int(pid), "inject.py", verbose=True)
