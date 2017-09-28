'''
This script launches the distributed  version of tensroflow simulations
'''
import os
import subprocess
import sys
import tensorflow as tf
import random
import time


def get_time_hhmmss(dif):
    m, s = divmod(dif, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

startTime = time.time()

tf.flags.DEFINE_integer("num_of_workers", 4, "num of workers to launch")
tf.flags.DEFINE_integer("num_threads", 16, "num of threads per worker")
tf.flags.DEFINE_integer("seed", int(round(random.random()*100000)), "the global simulation seed for np and tf")
FLAGS = tf.flags.FLAGS

#launch the param server
cmd = "python3 ./model.py --job_name=ps --task_index=0 --seed="+str(FLAGS.seed)+" --num_of_workers="+str(FLAGS.num_of_workers)+" --num_threads="+str(FLAGS.num_threads)
print "Lnch: ", cmd
subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
#launch workers
for worker in range(FLAGS.num_of_workers):
    cmd = "python3 ./model.py --job_name=worker --task_index="+str(worker)+" --seed="+str(FLAGS.seed)+" --num_of_workers="+str(FLAGS.num_of_workers)+" --num_threads="+str(FLAGS.num_threads)
    print "Lnch: ", cmd
    subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
    
#create a global clock
while True:
    time.sleep(1)
    print"##############################################"
    print"##############################################"
    print"[Launcher] time elapsed: ", get_time_hhmmss(time.time() - startTime)
    print"##############################################"
    print"##############################################"