from __future__ import absolute_import
import tensorflow as tf
import random
import time
import os
import sys
import datetime
import data_gen  

#model flags
tf.flags.DEFINE_boolean(u"debug", False, u"weather run in a dubg mode")

tf.flags.DEFINE_integer(u"total_num_epochs", 10000000, u"num_of_epochs")    
tf.flags.DEFINE_integer(u"state_size", 50, u"the state size of RNN")    
tf.flags.DEFINE_integer(u"num_samples", 1500, u"num of samples to generate")
tf.flags.DEFINE_integer(u"batch_size", 100, u"batch size for the input")

tf.flags.DEFINE_float(u"learning_rate", 0.005, u"learnig rate for the model")
tf.flags.DEFINE_float(u"epsilon", 1e-6, u"learnig rate for the model")
tf.flags.DEFINE_float(u"grad_norm", 10e2, u"amount of normalisation for the gradient")
tf.flags.DEFINE_integer(u"max_output_ops", 5, u"number of output from RNN")
tf.flags.DEFINE_float(u"loss_weight", 0.5, u"number of output from RNN")
tf.flags.DEFINE_float(u"softmax_sat", 1, u"number of output from RNN")
tf.flags.DEFINE_float(u"drop_rate", 0.15, u"dropout rate")
tf.flags.DEFINE_float(u"smax_pen_r", 0, u"sofmax penalty rate")
tf.flags.DEFINE_float(u"test_ratio", 0.33333, u"test train data split ratio")
tf.flags.DEFINE_integer(u"loss_swap_per", 50, u"periodicity of using just softmax penallty or math_error+softmax_r*smax_pen")
tf.flags.DEFINE_boolean(u"add_noise", True, u"whether to add grads to add random gausian decaying noise to the grads")

tf.flags.DEFINE_integer(u"num_features", 3, u"number of features per generated sample")
tf.flags.DEFINE_string(u"train_fn", u"np_add", u"the function which model has to learn")
tf.flags.DEFINE_string(u"model", u"RNN", u"what model to use for training")
tf.flags.DEFINE_string(u"state_fn", u"tanh", u"what state activation func to use")
tf.flags.DEFINE_boolean(u"share_state", True, u"whether to share RNN state between training and testing stages")
tf.flags.DEFINE_boolean(u"rnns_same_state", False, u"weather op and Mem RNNs are set to the same state before back and forward prop")

tf.flags.DEFINE_boolean(u"norm", True, u"weather to norm grads")
tf.flags.DEFINE_boolean(u"clip", False, u"weather to clip grads")
tf.flags.DEFINE_float(u"grad_clip_val_max", 10e3, u"grad_clip_val_max")
tf.flags.DEFINE_float(u"grad_clip_val_min", -10e3, u"grad_clip_val_min")

tf.flags.DEFINE_boolean(u"pen_sofmax", True, u"whether to compute sofmax penalties")
tf.flags.DEFINE_boolean(u"augument_grad", True, u"whether to augument grads involved in sofmax with sfotmax_sat val")

tf.flags.DEFINE_integer(u"seed", round(random.random()*100000), u"the global simulation seed for np and tf")
tf.flags.DEFINE_string(u"name", u"predef_sim_name" , u"name of the simulation")
tf.flags.DEFINE_boolean(u"logoff", False , u"stitch of loggin")
tf.flags.DEFINE_string(u"rerun_cfg", u"", u"rerun simulation with cfg path")
tf.flags.DEFINE_boolean(u"hardmax_break", True , u"if to finish on small hardmax error")
tf.flags.DEFINE_boolean(u"relaunch", False, u"whether to relaunch after instance of learning when model has not converged")
tf.flags.DEFINE_float(u"max_reward", 3000, u"max reward for RL env")

FLAGS = tf.flags.FLAGS

def get_cfg():
    #configuraion constants
    global_cfg = dict(
        total_num_epochs = FLAGS.total_num_epochs,
        iters_per_epoch = 1,
        samples_value_rng = (-100, 100),
        test_ratio = FLAGS.test_ratio,
        param_init = 0.1,
        epsilon= FLAGS.epsilon,
        test_cycle = 5,
        #test_cycle = 150,
        convergance_check_epochs = 500,
        sim_start_time = datetime.datetime.now().strftime(u"%Y_%m_%d_%H%M%S"),
        datatype = tf.float64,

        #flagged
        state_size = FLAGS.state_size,
        num_samples = FLAGS.num_samples,
        batch_size  = FLAGS.batch_size,

        drop_rate = FLAGS.drop_rate,
        learning_rate = FLAGS.learning_rate,
        grad_norm = FLAGS.grad_norm,
        max_output_ops = FLAGS.max_output_ops,
        loss_weight = FLAGS.loss_weight,
        rnns_same_state = FLAGS.rnns_same_state,
        softmax_sat = FLAGS.softmax_sat,
        
        pen_sofmax = FLAGS.pen_sofmax,
        smax_pen_r = FLAGS.smax_pen_r,
        
        num_features = FLAGS.num_features,
        train_fn = eval(u"data_gen."+FLAGS.train_fn),
        model = FLAGS.model,
        norm = FLAGS.norm,
        clip = FLAGS.clip,
        share_state = FLAGS.share_state,
        grad_clip_val_max = FLAGS.grad_clip_val_max,
        grad_clip_val_min = FLAGS.grad_clip_val_min,
        
        state_fn = FLAGS.state_fn,
        loss_swap_per = FLAGS.loss_swap_per,
        add_noise = FLAGS.add_noise,
        augument_grad = FLAGS.augument_grad,
        max_reward = FLAGS.max_reward,
        
        seed = FLAGS.seed,
        name = FLAGS.name,
        logoff = FLAGS.logoff,
        hardmax_break = FLAGS.hardmax_break,
        rerun_cfg = FLAGS.rerun_cfg,
        debug = FLAGS.debug,
        relaunch = FLAGS.relaunch
    )
    global_cfg[u'dst'] = global_cfg[u'model'] + u"/" + global_cfg[u'train_fn'].__name__ + u"-" + unicode(global_cfg[u'max_output_ops']) +u"ops/" + global_cfg[u'name']
    global_cfg[u'num_epochs'] = global_cfg[u'total_num_epochs'] // global_cfg[u'iters_per_epoch']
    
    #couple of quick cfg checks
    if global_cfg[u'norm'] is True or global_cfg[u'norm'] is True:
        assert global_cfg[u'norm'] is not global_cfg[u'clip'], u'cant norm and clip at the same time'
    if global_cfg[u'clip']:
        assert global_cfg[u'grad_clip_val_max'] >= global_cfg[u'grad_clip_val_min'], u'max clip val cannot be smaller than min'
    
    return global_cfg