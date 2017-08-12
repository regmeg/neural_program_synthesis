import tensorflow as tf
import random
import time
import os
import sys
import datetime
import data_gen  

#model flags
tf.flags.DEFINE_boolean("debug", False, "weather run in a dubg mode")

tf.flags.DEFINE_integer("state_size", 50, "weather to norm grads")    
tf.flags.DEFINE_integer("num_samples", 1500, "weather to norm grads")
tf.flags.DEFINE_integer("batch_size", 100, "weather to norm grads")

tf.flags.DEFINE_float("learning_rate", 0.005, "weather to norm grads")
tf.flags.DEFINE_float("grad_norm", 10e2, "weather to norm grads")
tf.flags.DEFINE_integer("max_output_ops", 5, "weather to norm grads")

tf.flags.DEFINE_integer("num_features", 3, "weather to norm grads")
tf.flags.DEFINE_string("train_fn", "np_mult", "weather to norm grads")
tf.flags.DEFINE_boolean("norm", True, "weather to norm grads")

tf.flags.DEFINE_integer("seed", round(random.random()*100000), "the global simulation seed for np and tf")
tf.flags.DEFINE_string("name", "predef_sim_name" , "name of the simulation")
tf.flags.DEFINE_string("dst", "predef_sim_name" , "name of the simulation")

FLAGS = tf.flags.FLAGS

def get_cfg():
    #configuraion constants
    global_cfg = dict(
        total_num_epochs = 10000000,
        iters_per_epoch = 1,
        num_of_operations = 3,
        samples_value_rng = (-100, 100),
        test_ratio = 0.33333,
        param_init = 0.1,
        epsilon=1e-6,
        test_cycle = 150,
        convergance_check_epochs = 1000,
        sim_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        datatype = tf.float64,

        #flagged
        state_size = FLAGS.state_size,
        num_samples = FLAGS.num_samples,
        batch_size  = FLAGS.batch_size,

        learning_rate = FLAGS.learning_rate,
        grad_norm = FLAGS.grad_norm,
        max_output_ops = FLAGS.max_output_ops,

        num_features = FLAGS.num_features,
        train_fn = data_gen.get_syn_fn(FLAGS.train_fn),
        norm = FLAGS.norm,

        seed = FLAGS.seed,
        name = FLAGS.name
    )
    global_cfg['num_epochs'] = global_cfg['total_num_epochs'] // global_cfg['iters_per_epoch']
    
    return global_cfg