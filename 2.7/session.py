from __future__ import with_statement
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops as tf_ops
import pprint
import random
import time
import os
import sys
import datetime
import subprocess
from data_gen import *
from params import get_cfg
from rnn_base import RNN
from ops import Operations
from itertools import izip
import re


def write_no_tf_summary(writer, tag, val, step):
   summary=tf.Summary()
   summary.value.add(tag=tag, simple_value = val)
   writer.add_summary(summary, step)
    

#helpder func
def get_time_hhmmss(dif):
    m, s = divmod(dif, 60)
    h, m = divmod(m, 60)
    if h > 24:
       h, m, s = 24, 0, 0
    time_str = u"%02d:%02d:%02d" % (h, m, s)
    return time_str

def determine_loss(epoch, cfg):
    period = cfg[u"loss_swap_per"]
    mod = epoch % (period*2)
    if mod < period: return True
    else :           return False


def gen_cmd_from_name(name_in, cfg):
    new_seed = int(round(random.random()*100000))
    tokens = name_in.split(u"~")
    string = u"python /home/rb7e15/2.7v/model.py"
    name = u" --name="
    for token in tokens:
        key, val = token.split(u"#")
        if key == u'grad_clip_val':
            val = val.split(u"*")
            string += u" --"+unicode(key)+u"_min="+unicode(val[0])
            string += u" --"+unicode(key)+u"_max="+unicode(val[1])
            name += unicode(key)+u"#"+unicode(val[0])+u"*"+unicode(val[1])+u"~"
        elif key == u'seed': continue
        else:
            string += u" --"+unicode(key)+u"="+unicode(val)
            name += unicode(key)+u"#"+unicode(val)+u"~"
    string += u" --max_output_ops="+unicode(cfg[u"max_output_ops"])
    string += u" --train_fn="+unicode(cfg[u"train_fn"].__name__)
    string += u" --model="+unicode(cfg[u"model"])
    name += u"seed#" + unicode(new_seed)
    seed  = u" --seed="+unicode(new_seed)
    return string + seed + name


def get_qsub_com(cmd, fname):
    num_epochs = re.search(r'(--total_num_epochs=)([0-9]*)( --)', cmd).groups()[1]
    run_time = int(4.2*int(num_epochs)) #2 seconds per epoch
    #set min run time 5 min
    if run_time < 5*60: run_time = 5*60
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    if h > 24:
       h, m, s = 24, 0, 0
    run_len = "%d:%02d:%02d" % (h, m, s)
    #cmd_qsub = 'qsub -l walltime='+run_len+' -l nodes=1:ppn=8 '+fname
    cmd_qsub = 'qsub -l walltime='+str(run_time)+' -l nodes=1:ppn=8 -o '+fname+'.o -e '+fname+'.e ' +fname
    print cmd_qsub
    return cmd_qsub

def gen_job_file(cmd, cfg):
    #job_name
    start_time = datetime.datetime.now().strftime(u"%Y_%m_%d_%H%M%S")
    fname = "/home/rb7e15/2.7v/jobs/job_"+cfg['model']+start_time+'_rel'
    #gen job launch scripts
    f = open(fname, u'w')
    f.write('#!/bin/bash\n')
    f.write('\n')
    f.write('echo "source the env"\n')
    f.write('module load python\n')
    f.write('export CC=/home/rb7e15/gcc/bin/gcc\n')
    f.write('export LD_LIBRARY_PATH=/home/rb7e15/gcc/lib64:$LD_LIBRARY_PATH\n')
    f.write('source /home/rb7e15/2.7v/TFenv/bin/activate\n')
    f.write('\n')
    f.write('echo  "launch the command"\n')
    f.write(cmd)
    f.write('\n')
    return fname
    
def run_session_2RNNS(m, cfg, x_train, x_test, y_train, y_test):
    #pre training setting
    np.set_printoptions(precision=3, suppress=True)
    #train_fn = np_mult
    #train_fn = np_stall
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    #model training

    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())

    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #define congergance check list
    last_train_losses = []

    with tf.Session(config=config) as sess:
        # Merge all the summaries and write them out 
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] ,sess.graph, max_queue=1)
        ##enable debugger if necessary
        if (cfg[u'debug']):
            print u"Running in a debug mode"
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(u"has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        #plt.ion()
        #plt.figure()
        #plt.show() 
        #Init vars:
        _W = sess.run([m.params[u'W']])
        _W2 = sess.run([m.params[u'W2']])
        _W3 = sess.run([m.params[u'W3']])
        
        _W_mem = sess.run([m.params[u'W_mem']])
        _W2_mem = sess.run([m.params[u'W2_mem']])
        #_W3_mem = sess.run([m.params['W3_mem']])
        
        print u"W1"
        print m.params[u'W'].eval()
        print u"W2"
        print m.params[u'W2'].eval()
        print u"W3"
        print m.params[u'W3'].eval()
        
        print u"W1_mem"
        print m.params[u'W_mem'].eval()
        print u"W2_mem"
        print m.params[u'W2_mem'].eval()
        
        #print("W3_mem")
        #print(m.params['W3_mem'].eval())
        
        _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        
        
        globalstartTime = time.time()
        for epoch_idx in xrange(cfg[u'num_epochs']):
            # reset variables
            startTime = time.time()
            loss_list_train_soft = [0,0]
            loss_list_train_hard = [0,0]
            loss_list_test_soft = [0,0]
            loss_list_test_hard = [0,0]
            
            math_error_train_soft = [0,0]
            math_error_train_hard = [0,0]
            math_error_test_soft = [0,0]
            math_error_test_hard = [0,0]
            
            u'''
            reduced_loss_train_soft = 0
            reduced_math_error_train_soft = 0
            pen_loss_train_soft = 0
            reduced_loss_train_hard = 0
            reduced_math_error_train_hard = 0
            pen_loss_train_hard = 0
            reduced_loss_test_soft = 0
            reduced_math_error_test_soft = 0
            pen_loss_test_soft = 0
            reduced_loss_test_hard = 0
            reduced_math_error_test_hard = 0
            pen_loss_test_hard = 0
            '''

            summary = None
            #shuffle data
            #x_train, y_train = shuffle_data(x_train, y_train)
            
            #determine if use both losses to train during current traiing
            use_both_losses = determine_loss(epoch_idx, cfg) 

                #backprop and test training set for softmax and hardmax loss
            for batch_idx in xrange(num_batches):
                
                    #if flag set, make op and mem selection rnn use exaclty the same state
                    if cfg[u'rnns_same_state'] is True:
                        _current_state_train_mem = _current_state_train
                        _current_state_test_mem  = _current_state_test
                            
                    start_idx = cfg[u'batch_size'] * batch_idx
                    end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]
                    
                                
                    #set states
                    _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            

                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data
                    if epoch_idx % cfg[u'test_cycle'] != 0 :                       
                      
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _output_train,\
                        _grads,\
                        _softmaxes_train,\
                        _math_error_train = sess.run([m.total_loss_train, 
                                                      m.train_step,
                                                      m.train[u"current_state"], 
                                                      m.train[u"current_state_mem"], 
                                                      m.train[u"output"], 
                                                      m.grads, 
                                                      m.train[u"softmaxes"],
                                                      m.math_error_train],
                        feed_dict={
                            m.init_state:_current_state_train,
                            m.mem.init_state:_current_state_train_mem,
                            m.batchX_placeholder:batchX,
                            m.batchY_placeholder:batchY,
                            m.use_both_losses: use_both_losses
                        })
                        loss_list_train_soft.append(_total_loss_train)
                        math_error_train_soft.append(_math_error_train)

                    else :
                    #for testing cylce, do one forward and back prop with 1 batch with training data, plus produce summary and hardmax result
                    
                        summary,\
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _output_train,\
                        _grads,\
                        _softmaxes_train,\
                        _math_error_train = sess.run([merged,
                                                      m.total_loss_train,
                                                      m.train_step,
                                                      m.train[u"current_state"],
                                                      m.train[u"current_state_mem"],
                                                      m.train[u"output"],
                                                      m.grads,
                                                      m.train[u"softmaxes"],
                                                      m.math_error_train],
                        feed_dict={
                            m.init_state:_current_state_train,
                            m.mem.init_state:_current_state_train_mem,
                            m.batchX_placeholder:batchX,
                            m.batchY_placeholder:batchY,
                            m.use_both_losses: use_both_losses
                        })
                        loss_list_train_soft.append(_total_loss_train)
                        math_error_train_soft.append(_math_error_train)
                        
                        _total_loss_test,\
                        _current_state_test,\
                        _current_state_test_mem,\
                        _output_test,\
                        _softmaxes_test,\
                        _math_error_test = sess.run([m.total_loss_test,
                                                     m.test[u"current_state"],
                                                     m.test[u"current_state_mem"],
                                                     m.test[u"output"],
                                                     m.test[u"softmaxes"],
                                                     m.math_error_test],
                            feed_dict={
                                m.init_state:_current_state_test,
                                m.mem.init_state:_current_state_test_mem,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY,
                                m.use_both_losses: use_both_losses
                            })
                        loss_list_train_hard.append(_total_loss_test)
                        math_error_train_hard.append(_math_error_test)
                        
            ##save loss for the convergance chassing        
            reduced_loss_train_soft = reduce(lambda x, y: x+y, loss_list_train_soft)
            last_train_losses.append(reduced_loss_train_soft)
            ##every 'test_cycle' epochs test the testing set for sotmax/harmax loss
            if epoch_idx % cfg[u'test_cycle'] == 0 : 
            
                for batch_idx in xrange(num_test_batches):
            
                        #if flag set, make op and mem selection rnn use exaclty the same state
                        if cfg[u'rnns_same_state'] is True:
                            _current_state_train_mem = _current_state_train
                            _current_state_test_mem  = _current_state_test
                            
                        start_idx = cfg[u'batch_size'] * batch_idx
                        end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]
                        
                        
                        #set states
                        _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                        _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                        _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                        _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                        
                        _total_loss_train,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _math_error_train        = sess.run([m.total_loss_train,
                                                             m.train[u"current_state"],
                                                             m.train[u"current_state_mem"],
                                                             m.math_error_train],
                            feed_dict={
                                m.init_state:_current_state_train,
                                m.mem.init_state:_current_state_train_mem,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY,
                                m.use_both_losses: use_both_losses
                            })
                        loss_list_test_soft.append(_total_loss_train)
                        math_error_test_soft.append(_math_error_train)

                        _total_loss_test,\
                        _current_state_test,\
                        _current_state_test_mem,\
                        _math_error_test        = sess.run([m.total_loss_test,
                                                            m.test[u"current_state"],
                                                            m.test[u"current_state_mem"],
                                                            m.math_error_test],
                            feed_dict={
                                m.init_state:_current_state_test,
                                m.mem.init_state:_current_state_test_mem,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY,
                                m.use_both_losses: use_both_losses
                            })
                        loss_list_test_hard.append(_total_loss_test)
                        math_error_test_hard.append(_math_error_test)
                            
                #save model            
                saver.save(sess, u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] + u'/model/',global_step=epoch_idx)
                #write variables/loss summaries after all training/testing done
            reduced_math_error_train_soft = reduce(lambda x, y: x+y, math_error_train_soft)
            pen_loss_train_soft = reduced_loss_train_soft - reduced_math_error_train_soft

            reduced_loss_train_hard = reduce(lambda x, y: x+y, loss_list_train_hard)
            reduced_math_error_train_hard = reduce(lambda x, y: x+y, math_error_train_hard)
            pen_loss_train_hard = reduced_loss_train_hard - reduced_math_error_train_hard
             
            reduced_loss_test_soft = reduce(lambda x, y: x+y, loss_list_test_soft)
            reduced_math_error_test_soft = reduce(lambda x, y: x+y, math_error_test_soft)
            pen_loss_test_soft = reduced_loss_test_soft - reduced_math_error_test_soft
                
            reduced_loss_test_hard = reduce(lambda x, y: x+y, loss_list_test_hard)
            reduced_math_error_test_hard = reduce(lambda x, y: x+y, math_error_test_hard)
            pen_loss_test_hard = reduced_loss_test_hard - reduced_math_error_test_hard
                
            if epoch_idx % cfg[u'test_cycle'] == 0 :
                #train_writer.add_summary(summary, epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_train_loss",      reduced_loss_train_soft, epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_math_train_loss", reduced_math_error_train_soft , epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_pen_train_loss",  pen_loss_train_soft, epoch_idx)
                
                #write_no_tf_summary(train_writer, u"Hardmax_train_loss",      reduced_loss_train_hard, epoch_idx)
                #write_no_tf_summary(train_writer, u"Hardmax_math_train_loss", reduced_math_error_train_hard, epoch_idx)
                #write_no_tf_summary(train_writer, u"Hardmax_pen_train_loss",  pen_loss_train_hard, epoch_idx)
                
                #write_no_tf_summary(train_writer, u"Softmax_test_loss",      reduced_loss_test_soft, epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_math_test_loss", reduced_math_error_test_soft, epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_pen_test_loss",  pen_loss_test_soft, epoch_idx)
                
                #write_no_tf_summary(train_writer, u"Hardmax_test_loss",      reduced_loss_test_hard, epoch_idx)
                #write_no_tf_summary(train_writer, u"Hardmax_math_test_loss", reduced_math_error_test_hard, epoch_idx)
                #write_no_tf_summary(train_writer, u"Hardmax_pen_test_loss",  pen_loss_test_hard, epoch_idx)

            	print u""
            #harmax test
            u'''
            print(" ")
            print("output_train\t\t\t\t\toutput_test")
            print(np.column_stack((_output, _output_test)))
            print("x\t\t\\t\t\t\t\y")
            print(np.column_stack((batchX, batchY)))
            print("softmaxes_train\t\t\t\t\softmaxes_test")
            print(np.column_stack((_softmaxes, _softmaxes_test)))
            print("mat_error_train\t\t\t\t\math_error_test")
            print(np.column_stack((_math_error, _math_error_test)))
            '''
            print u"Epoch",epoch_idx, u"use_both_losses", use_both_losses
            print u"Softmax train loss\t", reduced_loss_train_soft, u"(m:",reduced_math_error_train_soft,u"p:",pen_loss_train_soft,u")"
            print u"Hardmax train loss\t", reduced_loss_train_hard, u"(m:",reduced_math_error_train_hard,u"p:",pen_loss_train_hard,u")"
            print u"Sotfmax test loss\t", reduced_loss_test_soft, u"(m:",reduced_math_error_test_soft,u"p:",pen_loss_test_soft,u")"
            print u"Hardmax test loss\t", reduced_loss_test_hard, u"(m:",reduced_math_error_test_hard,u"p:",pen_loss_test_hard,u")"
            print u"Epoch time: ", ((time.time() - startTime) % 60), u" Global Time: ",  get_time_hhmmss(time.time() - globalstartTime)
            print u"func: ", cfg[u'train_fn'].__name__, u"max_ops: ", cfg[u'max_output_ops'], u"sim_seed", cfg[u'seed'], u"tf seed", tf_ops.get_default_graph().seed
            #print("grads[0] - W", _grads[0][0])
            #print("grads[1] - b", _grads[1][0])
            #print("grads[2] - W2", _grads[2][0])
            #print("grads[3] - b2", _grads[3][0])
            #print("W", W.eval())
            #print("w2" , W2.eval())
            #record execution timeline
            ##check convergance over last 5000 epochs
            if epoch_idx % cfg[u'convergance_check_epochs'] == 0 and epoch_idx >= cfg[u'convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print u"#################################"
                    print u"Model has converged, breaking ..."
                    print u"#################################"
                    break
                else:
                    print u"Reseting the loss conv array"
                    last_train_losses = []

            #as well check early stopping options, once hardmax train error is small enough - there is not point to check softmax, as its combinations of math error and penalties
            if cfg[u'hardmax_break']:
                if (epoch_idx % cfg[u'test_cycle'] == 0) and ((reduced_loss_train_hard < 0.01) or (reduced_loss_test_hard < 0.01)):
                        ## check thousand random samples
                        print u"@@checking random thousand samples"

                        #gen new seed
                        test_seed = round(random.random()*100000)
                        num_tests = 1000
                        print u"test_seed", test_seed, u"num_tests", num_tests
                        x_sample, y_sample = samples_generator(cfg[u'train_fn'], (num_tests, cfg[u'num_features']) , cfg[u'samples_value_rng'], test_seed)
                        match_count = 0
                        for i in xrange(num_tests):
                            batchX = np.zeros((cfg[u'batch_size']-1, cfg[u'num_features']))
                            batchX = np.concatenate(([x_sample[i]], batchX), axis=0)  
                            output = sess.run([m.test[u'output']],
                                feed_dict={
                                        m.init_state:np.zeros((cfg[u'batch_size'], cfg[u'state_size'])),
                                        m.mem.init_state:np.zeros((cfg[u'batch_size'], cfg[u'state_size'])),
                                        m.batchX_placeholder:batchX
                                    })
                            match = np.allclose(y_sample[i], output[0][0])
                            print u"i", i, u"match", match
                            print u"input", list(x_sample[i])
                            print u"expect", list(y_sample[i])
                            print u"actual", list(output[0][0].tolist())
                            if match:
                                match_count = match_count + 1
                        print
                        print match_count, u"out of", num_tests,u"matched"
                        
                        print u"#################################"
                        print u"Model reached hardmax, breaking ..."
                        print u"#################################"                                                     

          
                        
                        ##break
                        break

    if cfg[u'relaunch']:
            cmd = gen_cmd_from_name(cfg[u"name"], cfg)
            fname = gen_job_file(cmd, cfg)
            cmd_qsub = get_qsub_com(cmd, fname)
            print u"ReLnch: " + cmd
            subprocess.Popen(cmd_qsub, shell=True, stderr=subprocess.STDOUT)


def restore_selection_matrixes2RNNS(m, cfg, x_train, x_test, y_train, y_test, path):
    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())
    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #def batches num
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    with tf.Session(config=config) as sess:
        ##enable debugger if necessary
        if (cfg[u'debug']):
            print u"Running in a debug mode"
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(u"has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))
        

        #get soft and hardmaxes out of the model for the last batches
        total_loss_traind_train = []
        
        outputs_traind_train = []
        outputs_traind_train_mem = []
        
        softmaxes_traind_train = []
        softmaxes_traind_train_mem = []
        
        total_loss_traind_test = []
        
        outputs_traind_test = []
        outputs_traind_test_mem = []
        
        softmaxes_traind_test = []
        softmaxes_traind_test_mem = []

        _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))

        #FOR THE TRAINING DATA
        for batch_idx in xrange(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]

               
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_traind_train,\
                _outputs_traind_train,\
                _outputs_traind_train_mem,\
                _softmaxes_traind_train,\
                _softmaxes_traind_train_mem,\
                _current_state_train,\
                _current_state_train_mem = sess.run([m.total_loss_train,
                                                     m.train[u"outputs"],
                                                     m.train[u"outputs_mem"],
                                                     m.train[u"softmaxes"],
                                                     m.train[u"softmaxes_mem"],
                                                     m.train[u"current_state"],
                                                     m.train[u"current_state_mem"]],
                                                
                                                
                feed_dict={
                    m.init_state:_current_state_train,
                    m.mem.init_state:_current_state_train_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
                total_loss_traind_train.append(_total_loss_traind_train)
                outputs_traind_train.append(_outputs_traind_train)
                outputs_traind_train_mem.append(_outputs_traind_train_mem)
                softmaxes_traind_train.append(_softmaxes_traind_train)
                softmaxes_traind_train_mem.append(_softmaxes_traind_train_mem)
                
                #FOR THE HARDMAX SELECTION
                _total_loss_traind_test,\
                _outputs_traind_test,\
                _outputs_traind_test_mem,\
                _softmaxes_traind_test,\
                _softmaxes_traind_test_mem,\
                _current_state_test,\
                _current_state_test_mem = sess.run([m.total_loss_test,
                                                     m.test[u"outputs"],
                                                     m.test[u"outputs_mem"],
                                                     m.test[u"softmaxes"],
                                                     m.test[u"softmaxes_mem"],
                                                     m.test[u"current_state"],
                                                     m.test[u"current_state_mem"]],
                    feed_dict={
                        m.init_state:_current_state_test,
                        m.mem.init_state:_current_state_test_mem,
                        m.batchX_placeholder:batchX,
                        m.batchY_placeholder:batchY
                    })
                total_loss_traind_test.append(_total_loss_traind_test)
                outputs_traind_test.append(_outputs_traind_test)
                outputs_traind_test_mem.append(_outputs_traind_test_mem)
                softmaxes_traind_test.append(_softmaxes_traind_test)
                softmaxes_traind_test_mem.append(_softmaxes_traind_test_mem)
                
        
        last_softmax_state_train = _current_state_train
        last_hardmax_state_train = _current_state_test 
        last_softmax_state_train_mem = _current_state_train
        last_hardmax_state_train_mem = _current_state_test 
        
        #produce results ith with the testing data
        total_loss_testd_train = []
        
        outputs_testd_train = []
        outputs_testd_train_mem = []
        
        softmaxes_testd_train = []
        softmaxes_testd_train_mem = []
        
        total_loss_testd_test = []
        
        outputs_testd_test =[]
        outputs_testd_test_mem =[]
        
        softmaxes_testd_test =[]
        softmaxes_testd_test_mem =[]
        
        if cfg[u'share_state'] is False:
            _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        
        #FOR THE TESTING DATA
        for batch_idx in xrange(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_testd_train,\
                _outputs_testd_train,\
                _outputs_testd_train_mem,\
                _softmaxes_testd_train,\
                _softmaxes_testd_train_mem,\
                _current_state_train,\
                _current_state_train_mem = sess.run([m.total_loss_train,
                                                     m.train[u"outputs"],
                                                     m.train[u"outputs_mem"],
                                                     m.train[u"softmaxes"],
                                                     m.train[u"softmaxes_mem"],
                                                     m.train[u"current_state"],
                                                     m.train[u"current_state_mem"]],
                feed_dict={
                    m.init_state:_current_state_train,
                    m.mem.init_state:_current_state_train_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
                total_loss_testd_train.append(_total_loss_testd_train)
                outputs_testd_train.append(_outputs_testd_train)
                outputs_testd_train_mem.append(_outputs_testd_train_mem)
                softmaxes_testd_train.append(_softmaxes_testd_train)
                softmaxes_testd_train_mem.append(_softmaxes_testd_train_mem)
                
                #FOR THE HARDMAX SELECTION
                _total_loss_testd_test,\
                _outputs_testd_test,\
                _outputs_testd_test_mem,\
                _softmaxes_testd_test,\
                _softmaxes_testd_test_mem,\
                _current_state_test,\
                _current_state_test_mem = sess.run([m.total_loss_test,
                                                     m.test[u"outputs"],
                                                     m.test[u"outputs_mem"],
                                                     m.test[u"softmaxes"],
                                                     m.test[u"softmaxes_mem"],
                                                     m.test[u"current_state"],
                                                     m.test[u"current_state_mem"]],
                    feed_dict={
                        m.init_state:_current_state_test,
                        m.mem.init_state:_current_state_test_mem,
                        m.batchX_placeholder:batchX,
                        m.batchY_placeholder:batchY
                    })
                total_loss_testd_test.append(_total_loss_testd_test)
                outputs_testd_test.append(_outputs_testd_test)
                outputs_testd_test_mem.append(_outputs_testd_test_mem)
                softmaxes_testd_test.append(_softmaxes_testd_test)
                softmaxes_testd_test_mem.append(_softmaxes_testd_test_mem)
                
        last_softmax_state_test = _current_state_train
        last_hardmax_state_test = _current_state_test
        last_softmax_state_test_mem = _current_state_train
        last_hardmax_state_test_mem = _current_state_test 
        
        
        #produce batches for reference        
        batchesX_train = []
        batchesY_train = []
        
        #FOR THE TRAINING DATA
        for batch_idx in xrange(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                batchesX_train.append(batchX)
                batchesY_train.append(batchY)
                
                
        batchesX_test = []
        batchesY_test = []
        
        #FOR THE TESTING DATA
        for batch_idx in xrange(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                batchesX_test.append(batchX) 
                batchesY_test.append(batchY) 
        
        return dict(
                #outputs for the train data
                total_loss_traind_train = total_loss_traind_train,
                outputs_traind_train = outputs_traind_train,
                outputs_traind_train_mem = outputs_traind_train_mem,
                softmaxes_traind_train = softmaxes_traind_train,
                softmaxes_traind_train_mem = softmaxes_traind_train_mem,
                total_loss_traind_test = total_loss_traind_test,
                outputs_traind_test = outputs_traind_test,
                outputs_traind_test_mem = outputs_traind_test_mem,
                softmaxes_traind_test = softmaxes_traind_test,
                softmaxes_traind_test_mem = softmaxes_traind_test_mem,
                last_softmax_state_train = last_softmax_state_train,
                last_hardmax_state_train = last_hardmax_state_train,
                last_softmax_state_train_mem = last_softmax_state_train_mem,
                last_hardmax_state_train_mem = last_hardmax_state_train_mem,
            
                #outputs for the testing data
                total_loss_testd_train = total_loss_testd_train,
                outputs_testd_train = outputs_testd_train,
                outputs_testd_train_mem = outputs_testd_train_mem,
                softmaxes_testd_train = softmaxes_testd_train,
                softmaxes_testd_train_mem = softmaxes_testd_train_mem,
                total_loss_testd_test = total_loss_testd_test,
                outputs_testd_test = outputs_testd_test,
                outputs_testd_test_mem = outputs_testd_test_mem,
                softmaxes_testd_test = softmaxes_testd_test,
                softmaxes_testd_test_mem = softmaxes_testd_test_mem,
                last_softmax_state_test = last_softmax_state_test,
                last_hardmax_state_test = last_hardmax_state_test,
                last_softmax_state_test_mem = last_softmax_state_test_mem,
                last_hardmax_state_test_mem = last_hardmax_state_test_mem,
            
                #return divided up batches
                batchesX_train = batchesX_train,
                batchesY_train = batchesY_train,
                batchesX_test = batchesX_test,
                batchesY_test = batchesY_test
        )

def predict_form_sess(m, cfg, x, state, state_mem, path, mode=u"hard"):
    #create a saver to restore saved model
    saver=tf.train.Saver(var_list=tf.trainable_variables())

    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:

        #init the var
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))      
        
        batchX = np.zeros((cfg[u'batch_size']-x.shape[0], cfg[u'num_features']))
        batchY = np.zeros((cfg[u'batch_size'], cfg[u'num_features']))
        
        batchX = np.concatenate((x, batchX), axis=0)

        if mode == u"soft":
            output = sess.run([m.train[u'output']],
                feed_dict={
                    m.init_state:state,
                    m.mem.init_state:state_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
        elif mode == u"hard":
            output = sess.run([m.test[u'output']],
                feed_dict={
                    m.init_state:state,
                    m.mem.init_state:state_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
        else: raise(u"Wrong mode selected for predicting variable")
            
    return output

def run_session_HistoryRNN(m, cfg, x_train, x_test, y_train, y_test):
    #pre training setting
    np.set_printoptions(precision=3, suppress=True)
    #train_fn = np_mult
    #train_fn = np_stall
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    #model training

    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())

    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #define congergance check list
    last_train_losses = []

    with tf.Session(config=config) as sess:
        # Merge all the summaries and write them out 
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] ,sess.graph, max_queue=1)
        ##enable debugger if necessary
        if (cfg[u'debug']):
            print u"Running in a debug mode"
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(u"has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        #plt.ion()
        #plt.figure()
        #plt.show() 
        #Init vars:
        _W = sess.run([m.params[u'W_hist']])
        _W2 = sess.run([m.params[u'W2_mem']])
        print m.params[u'W_hist'].eval()
        print m.params[u'W2_mem'].eval()
        globalstartTime = time.time()
        for epoch_idx in xrange(cfg[u'num_epochs']):
            # reset variables
            startTime = time.time()
            loss_list_train_soft = [0,0]
            loss_list_train_hard = [0,0]
            loss_list_test_soft = [0,0]
            loss_list_test_hard = [0,0]
            summary = None
            #shuffle data
            #x_train, y_train = shuffle_data(x_train, y_train)

            _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            
                #backprop and test training set for softmax and hardmax loss
            for batch_idx in xrange(num_batches):
                            
                    start_idx = cfg[u'batch_size'] * batch_idx
                    end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]

                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data
                    if epoch_idx % cfg[u'test_cycle'] != 0 :                       
                      
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _output_train,\
                        _grads,\
                        _math_error_train = sess.run([m.total_loss_train, 
                                                      m.train_step,
                                                      m.train[u"current_state"], 
                                                      m.train[u"output"], 
                                                      m.grads, 
                                                      m.math_error_train],
                        feed_dict={
                            m.init_state:_current_state_train,
                            m.batchX_placeholder:batchX,
                            m.batchY_placeholder:batchY
                        })
                        loss_list_train_soft.append(_total_loss_train)

                    else :
                    #for testing cylce, do one forward and back prop with 1 batch with training data, plus produce summary and hardmax result
                    
                        summary,\
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _output_train,\
                        _grads,\
                        _math_error_train = sess.run([merged,
                                                      m.total_loss_train,
                                                      m.train_step,
                                                      m.train[u"current_state"],
                                                      m.train[u"output"],
                                                      m.grads,
                                                      m.math_error_train],
                        feed_dict={
                            m.init_state:_current_state_train,
                            m.batchX_placeholder:batchX,
                            m.batchY_placeholder:batchY
                        })
                        loss_list_train_soft.append(_total_loss_train)

                        _total_loss_test,\
                        _current_state_test,\
                        _output_test,\
                        _math_error_test = sess.run([m.total_loss_test,
                                                     m.test[u"current_state"],
                                                     m.test[u"output"],
                                                     m.math_error_test],
                            feed_dict={
                                m.init_state:_current_state_test,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY
                            })
                        loss_list_train_hard.append(_total_loss_test)

            ##save loss for the convergance chassing        
            reduced_loss_train_soft = reduce(lambda x, y: x+y, loss_list_train_soft)
            last_train_losses.append(reduced_loss_train_soft)
            ##every 'test_cycle' epochs test the testing set for sotmax/harmax loss
            if epoch_idx % cfg[u'test_cycle'] == 0 :
                
                #if sharing state, share state between training and testing data, 
                #else completely reset the state
                if cfg[u'share_state'] is False:
                    _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            
                for batch_idx in xrange(num_test_batches):           
                           
                        start_idx = cfg[u'batch_size'] * batch_idx
                        end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]

                        _total_loss_train,\
                        _current_state_train = sess.run([m.total_loss_train,
                                                             m.train[u"current_state"]],
                            feed_dict={
                                m.init_state:_current_state_train,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY
                            })
                        loss_list_test_soft.append(_total_loss_train)

                        _total_loss_test,\
                        _current_state_test = sess.run([m.total_loss_test,
                                                        m.test[u"current_state"]],
                            feed_dict={
                                m.init_state:_current_state_test,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY
                            })
                        loss_list_test_hard.append(_total_loss_test)

                #save model            
                saver.save(sess, u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] + u'/model/',global_step=epoch_idx)
                #write variables/loss summaries after all training/testing done
                #train_writer.add_summary(summary, epoch_idx)
                #write_no_tf_summary(train_writer, u"Softmax_train_loss", reduced_loss_train_soft, epoch_idx)
                #write_no_tf_summary(train_writer, u"Hardmax_train_loss", reduce(lambda x, y: x+y, loss_list_train_hard), epoch_idx)
                #write_no_tf_summary(train_writer, u"Sotfmax_test_loss", reduce(lambda x, y: x+y, loss_list_test_soft), epoch_idx)
                write_no_tf_summary(train_writer, u"Hardmax_test_loss", reduce(lambda x, y: x+y, loss_list_test_hard), epoch_idx)

            print u""
            #harmax test
            u'''
            print(" ")
            print("output_train\t\t\t\t\toutput_test")
            print(np.column_stack((_output, _output_test)))
            print("x\t\t\\t\t\t\t\y")
            print(np.column_stack((batchX, batchY)))
            print("softmaxes_train\t\t\t\t\softmaxes_test")
            print(np.column_stack((_softmaxes, _softmaxes_test)))
            print("mat_error_train\t\t\t\t\math_error_test")
            print(np.column_stack((_math_error, _math_error_test)))
            '''
            print u"Epoch",epoch_idx
            print u"Softmax train loss\t", reduced_loss_train_soft
            print u"Hardmax train loss\t", reduce(lambda x, y: x+y, loss_list_train_hard)
            print u"Sotfmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_soft)
            print u"Hardmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_hard)
            print u"Epoch time: ", ((time.time() - startTime) % 60), u" Global Time: ",  get_time_hhmmss(time.time() - globalstartTime)
            print u"func: ", cfg[u'train_fn'].__name__, u"max_ops: ", cfg[u'max_output_ops'], u"sim_seed", cfg[u'seed'], u"tf seed", tf_ops.get_default_graph().seed
            #print("grads[0] - W", _grads[0][0])
            #print("grads[1] - b", _grads[1][0])
            #print("grads[2] - W2", _grads[2][0])
            #print("grads[3] - b2", _grads[3][0])
            #print("W", W.eval())
            #print("w2" , W2.eval())
            #record execution timeline
            ##check convergance over last 5000 epochs
            if epoch_idx % cfg[u'convergance_check_epochs'] == 0 and epoch_idx >= cfg[u'convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print u"#################################"
                    print u"Model has converged, breaking ..."
                    print u"#################################"
                    break
                else:
                    print u"Reseting the loss conv array"
                    last_train_losses = []
                    
def restore_selection_matrixes_HistoryRNNS(m, cfg, x_train, x_test, y_train, y_test, path):
    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())
    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #def batches num
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    with tf.Session(config=config) as sess:
        ##enable debugger if necessary
        if (cfg[u'debug']):
            print u"Running in a debug mode"
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(u"has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))
        

        #get soft and hardmaxes out of the model for the last batches
        total_loss_traind_train = []
        
        outputs_traind_train_op = []
        outputs_traind_train_mem = []
        
        softmaxes_traind_train_op = []
        softmaxes_traind_train_mem = []
        
        total_loss_traind_test = []
        
        outputs_traind_test_op = []
        outputs_traind_test_mem = []
        
        softmaxes_traind_test_op = []
        softmaxes_traind_test_mem = []

        _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
    

        #FOR THE TRAINING DATA
        for batch_idx in xrange(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]

               
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_traind_train,\
                _outputs_traind_train_op,\
                _outputs_traind_train_mem,\
                _softmaxes_traind_train_op,\
                _softmaxes_traind_train_mem,\
                _current_state_train     = sess.run([m.total_loss_train,
                                                     m.train[u"outputs_op"],
                                                     m.train[u"outputs_mem"],
                                                     m.train[u"softmaxes_op"],
                                                     m.train[u"softmaxes_mem"],
                                                     m.train[u"current_state"]],
                                                
                                                
                feed_dict={
                    m.init_state:_current_state_train,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
                total_loss_traind_train.append(_total_loss_traind_train)
                outputs_traind_train_op.append(_outputs_traind_train_op)
                outputs_traind_train_mem.append(_outputs_traind_train_mem)
                softmaxes_traind_train_op.append(_softmaxes_traind_train_op)
                softmaxes_traind_train_mem.append(_softmaxes_traind_train_mem)
                
                #FOR THE HARDMAX SELECTION
                _total_loss_traind_test,\
                _outputs_traind_test_op,\
                _outputs_traind_test_mem,\
                _softmaxes_traind_test_op,\
                _softmaxes_traind_test_mem,\
                _current_state_test      = sess.run([m.total_loss_test,
                                                     m.test[u"outputs_op"],
                                                     m.test[u"outputs_mem"],
                                                     m.test[u"softmaxes_op"],
                                                     m.test[u"softmaxes_mem"],
                                                     m.test[u"current_state"]],
                    feed_dict={
                        m.init_state:_current_state_test,
                        m.batchX_placeholder:batchX,
                        m.batchY_placeholder:batchY
                    })
                total_loss_traind_test.append(_total_loss_traind_test)
                outputs_traind_test_op.append(_outputs_traind_test_op)
                outputs_traind_test_mem.append(_outputs_traind_test_mem)
                softmaxes_traind_test_op.append(_softmaxes_traind_test_op)
                softmaxes_traind_test_mem.append(_softmaxes_traind_test_mem)
                
        
        last_softmax_state_train = _current_state_train
        last_hardmax_state_train = _current_state_test 
        
        #produce results ith with the testing data
        total_loss_testd_train = []
        
        outputs_testd_train_op = []
        outputs_testd_train_mem = []
        
        softmaxes_testd_train_op = []
        softmaxes_testd_train_mem = []
        
        total_loss_testd_test = []
        
        outputs_testd_test_op =[]
        outputs_testd_test_mem =[]
        
        softmaxes_testd_test_op =[]
        softmaxes_testd_test_mem =[]
        
        if cfg[u'share_state'] is False:
            _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        
        #FOR THE TESTING DATA
        for batch_idx in xrange(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_testd_train,\
                _outputs_testd_train_op,\
                _outputs_testd_train_mem,\
                _softmaxes_testd_train_op,\
                _softmaxes_testd_train_mem,\
                _current_state_train     = sess.run([m.total_loss_train,
                                                     m.train[u"outputs_op"],
                                                     m.train[u"outputs_mem"],
                                                     m.train[u"softmaxes_op"],
                                                     m.train[u"softmaxes_mem"],
                                                     m.train[u"current_state"]],
                feed_dict={
                    m.init_state:_current_state_train,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
                total_loss_testd_train.append(_total_loss_testd_train)
                outputs_testd_train_op.append(_outputs_testd_train_op)
                outputs_testd_train_mem.append(_outputs_testd_train_mem)
                softmaxes_testd_train_op.append(_softmaxes_testd_train_op)
                softmaxes_testd_train_mem.append(_softmaxes_testd_train_mem)
                
                #FOR THE HARDMAX SELECTION
                _total_loss_testd_test,\
                _outputs_testd_test_op,\
                _outputs_testd_test_mem,\
                _softmaxes_testd_test_op,\
                _softmaxes_testd_test_mem,\
                _current_state_test      = sess.run([m.total_loss_test,
                                                     m.test[u"outputs_op"],
                                                     m.test[u"outputs_mem"],
                                                     m.test[u"softmaxes_op"],
                                                     m.test[u"softmaxes_mem"],
                                                     m.test[u"current_state"]],
                                
                    feed_dict={
                        m.init_state:_current_state_test,
                        m.batchX_placeholder:batchX,
                        m.batchY_placeholder:batchY
                    })
                total_loss_testd_test.append(_total_loss_testd_test)
                outputs_testd_test_op.append(_outputs_testd_test_op)
                outputs_testd_test_mem.append(_outputs_testd_test_mem)
                softmaxes_testd_test_op.append(_softmaxes_testd_test_op)
                softmaxes_testd_test_mem.append(_softmaxes_testd_test_mem)
                
        last_softmax_state_test = _current_state_train
        last_hardmax_state_test = _current_state_test
        
        #produce batches for reference
        
        batchesX_train = []
        batchesY_train = []
        
        #FOR THE TRAINING DATA
        for batch_idx in xrange(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                batchesX_train.append(batchX)
                batchesY_train.append(batchY)
                
                
        batchesX_test = []
        batchesY_test = []
        
        #FOR THE TESTING DATA
        for batch_idx in xrange(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                batchesX_test.append(batchX) 
                batchesY_test.append(batchY) 
                
        return dict(
                #outputs for the train data
                total_loss_traind_train = total_loss_traind_train,
                outputs_traind_train = outputs_traind_train_op,
                outputs_traind_train_mem = outputs_traind_train_mem,
                softmaxes_traind_train = softmaxes_traind_train_op,
                softmaxes_traind_train_mem = softmaxes_traind_train_mem,
                total_loss_traind_test = total_loss_traind_test,
                outputs_traind_test = outputs_traind_test_op,
                outputs_traind_test_mem = outputs_traind_test_mem,
                softmaxes_traind_test = softmaxes_traind_test_op,
                softmaxes_traind_test_mem = softmaxes_traind_test_mem,
                last_softmax_state_train = last_softmax_state_train,
                last_hardmax_state_train = last_hardmax_state_train,
            
                #outputs for the testing data
                total_loss_testd_train = total_loss_testd_train,
                outputs_testd_train = outputs_testd_train_op,
                outputs_testd_train_mem = outputs_testd_train_mem,
                softmaxes_testd_train = softmaxes_testd_train_op,
                softmaxes_testd_train_mem = softmaxes_testd_train_mem,
                total_loss_testd_test = total_loss_testd_test,
                outputs_testd_test = outputs_testd_test_op,
                outputs_testd_test_mem = outputs_testd_test_mem,
                softmaxes_testd_test = softmaxes_testd_test_op,
                softmaxes_testd_test_mem = softmaxes_testd_test_mem,
                last_softmax_state_test = last_softmax_state_test,
                last_hardmax_state_test = last_hardmax_state_test,
            
                #return divided up batches
                batchesX_train = batchesX_train,
                batchesY_train = batchesY_train,
                batchesX_test = batchesX_test,
                batchesY_test = batchesY_test
        )
def run_session_RL_RNN(m, cfg, x_train, x_test, y_train, y_test):
    #pre training setting
    np.set_printoptions(precision=3, suppress=True)
    #train_fn = np_mult
    #train_fn = np_stall
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    #model training

    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())

    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #define congergance check list
    last_train_losses = []

    with tf.Session(config=config) as sess:
        # Merge all the summaries and write them out 
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] ,sess.graph, max_queue=1)
        ##enable debugger if necessary
        if (cfg[u'debug']):
            print u"Running in a debug mode"
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(u"has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        #plt.ion()
        #plt.figure()
        #plt.show() 
        #Init vars:
        _W = sess.run([m.params[u'W']])
        _W2 = sess.run([m.params[u'W2']])
        _W3 = sess.run([m.params[u'W3']])
        
        _W_mem = sess.run([m.params[u'W_mem']])
        _W2_mem = sess.run([m.params[u'W2_mem']])
        _W3_mem = sess.run([m.params[u'W3_mem']])
        
        print u"W1"
        print m.params[u'W'].eval()
        print u"W2"
        print m.params[u'W2'].eval()
        print u"W3"
        print m.params[u'W3'].eval()
        
        print u"W1_mem"
        print m.params[u'W_mem'].eval()
        print u"W2_mem"
        print m.params[u'W2_mem'].eval()
        print u"W3_mem"
        print m.params[u'W3_mem'].eval()
        
        globalstartTime = time.time()
        for epoch_idx in xrange(cfg[u'num_epochs']):
            # reset variables
            startTime = time.time()
            loss_list_train_log = [0,0]
            loss_list_train_rewards = [0,0]
            loss_list_train_math_error = [0,0]
            
            loss_list_test_rewards = [0,0]
            loss_list_test_math_error = [0,0]
            
            summary = None
            #shuffle data
            #x_train, y_train = shuffle_data(x_train, y_train)
            
            #determine if use both losses to train during current traiing
            use_both_losses = determine_loss(epoch_idx, cfg) 
            
            #set states
            _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            
                #backprop and test training set for softmax and hardmax loss
            for batch_idx in xrange(num_batches):
                            
                    start_idx = cfg[u'batch_size'] * batch_idx
                    end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]
                    
                    #print("computing rollout")
                    #rollout policites to get rewards
                    p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, True)
                    
                    #maks non mem cases for the mem backprop
                    width = np.vstack(p[u'states_mem']).shape[1]
                    mask = np.lib.pad(np.vstack(p[u'mem_masks']), ((0, 0), (0,  width - 1)), u'edge')                       
                    mem_state = np.vstack(p[u'states_mem'])[mask == True].reshape(-1, width)                        
                    
                    #width = np.vstack(p['current_exes_mem']).shape[1]
                    #mask = np.lib.pad(np.vstack(p['mem_masks']), ((0, 0), (0,  width - 1)), 'edge')                       
                    #mem_x = np.vstack(p['current_exes_mem'])[mask == True].reshape(-1, width)
                    #use same exes for the mem
                    width = np.vstack(p[u'current_exes']).shape[1]
                    mask = np.lib.pad(np.vstack(p[u'mem_masks']), ((0, 0), (0,  width - 1)), u'edge')                       
                    mem_x = np.vstack(p[u'current_exes'])[mask == True].reshape(-1, width)
                    
                    width = np.vstack(p[u'labels_mem']).shape[1]
                    mask = np.lib.pad(np.vstack(p[u'mem_masks']), ((0, 0), (0,  width - 1)), u'edge')                       
                    mem_y = np.vstack(p[u'labels_mem'])[mask == True].reshape(-1, width)
                    
                    mem_sel = np.vstack(p[u'selections_mem'])[np.vstack(p[u'mem_masks']) == True]
                    mem_rews = np.vstack(np.hstack(np.stack(p[u'discount_rewards'], axis=1)))[np.vstack(p[u'mem_masks']) == True]                   
                    no_mem_bprop = False
                    
                    if mem_state.size  == 0:
                        no_mem_bprop = True
                    if mem_x.size == 0: 
                        no_mem_bprop = True
                    if mem_y.size == 0:
                        no_mem_bprop = True
                    if mem_sel.size == 0:
                        no_mem_bprop = True
                    if mem_rews.size == 0: 
                        no_mem_bprop = True
                    
                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data       

                    if epoch_idx % cfg[u'test_cycle'] != 0 :                       
                      
                       
                        #if no error, dont backprop
                        #if p['math_error'].sum() > 0.0000000001:
                        if True:
                            if no_mem_bprop:

                                    #summary,\
                                    _total_loss_train,\
                                    _train_step        = sess.run([
                                                                      #merged,
                                                                      m.total_loss_train,
                                                                      m.train_step],
                                    feed_dict={
                                        m.init_state:np.vstack(p[u'states']),

                                        m.batchX_placeholder:np.vstack(p[u'current_exes']),

                                        m.batchY_placeholder:np.vstack(p[u'labels']),

                                        m.selections_placeholder:np.vstack(p[u'selections']),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p[u'discount_rewards'], axis=1))),
                                        
                                        m.training: True

                                    })
                            else:
                                    #summary,\
                                    _total_loss_train,\
                                    _train_step,\
                                    _trai_step_mem        = sess.run([
                                                                      #merged,
                                                                      m.total_loss_train,
                                                                      m.train_step,
                                                                      m.mem.train_step],
                                    feed_dict={
                                        m.init_state:np.vstack(p[u'states']),
                                        m.mem.init_state: np.vstack(mem_state),

                                        m.batchX_placeholder:np.vstack(p[u'current_exes']),
                                        m.mem.batchX_placeholder: np.vstack(mem_x),

                                        m.batchY_placeholder:np.vstack(p[u'labels']),
                                        m.mem.batchY_placeholder: np.vstack(mem_y),

                                        m.selections_placeholder:np.vstack(p[u'selections']),
                                        m.mem.selections_placeholder: np.vstack(mem_sel),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p[u'discount_rewards'], axis=1))),
                                        m.mem.rewards_placeholder: np.vstack(mem_rews),
                                        
                                        m.training: True

                                })                        


                    else :


                        #if p['math_error'].sum() > 0.0000000001:
                        if True:

                            if no_mem_bprop:

                                    summary,\
                                    _total_loss_train,\
                                    _train_step        = sess.run([
                                                                      merged,
                                                                      m.total_loss_train,
                                                                      m.train_step],
                                    feed_dict={
                                        m.init_state:np.vstack(p[u'states']),

                                        m.batchX_placeholder:np.vstack(p[u'current_exes']),

                                        m.batchY_placeholder:np.vstack(p[u'labels']),

                                        m.selections_placeholder:np.vstack(p[u'selections']),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p[u'discount_rewards'], axis=1))),
                                        
                                        m.training: True

                                    })
                            else:
                                    summary,\
                                    _total_loss_train,\
                                    _train_step,\
                                    _trai_step_mem        = sess.run([
                                                                      merged,
                                                                      m.total_loss_train,
                                                                      m.train_step,
                                                                      m.mem.train_step],
                                    feed_dict={
                                        m.init_state:np.vstack(p[u'states']),
                                        m.mem.init_state: np.vstack(mem_state),

                                        m.batchX_placeholder:np.vstack(p[u'current_exes']),
                                        m.mem.batchX_placeholder: np.vstack(mem_x),

                                        m.batchY_placeholder:np.vstack(p[u'labels']),
                                        m.mem.batchY_placeholder: np.vstack(mem_y),

                                        m.selections_placeholder:np.vstack(p[u'selections']),
                                        m.mem.selections_placeholder: np.vstack(mem_sel),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p[u'discount_rewards'], axis=1))),
                                        m.mem.rewards_placeholder: np.vstack(mem_rews),
                                        
                                        m.training: True

                                })
                        
                    if cfg[u'num_samples'] < 16:
                            print list(izip(np.hstack(p[u'selections']).tolist(),np.around(p[u'discount_rewards'], decimals=3).tolist()))
                            #print(list(zip( np.hstack(p['selections']).tolist(), np.hstack(p['mem_masks']).tolist() )))
                            #print(list(zip(np.hstack(p['selections_mem']).tolist(),np.around(p['discount_rewards'], decimals=3).tolist())))
                            if no_mem_bprop:
                                print u"no_mem_sel_ops"
                            else:
                                print list(izip(np.hstack(mem_sel).tolist(),np.around(mem_rews, decimals=3).tolist()))
                            print u'""'
                    #if p['math_error'].sum() > 0.0000000001:
                    loss_list_train_log.append(_total_loss_train)
                    loss_list_train_rewards.append( np.vstack(p[u'rewards']).sum() )
                    loss_list_train_math_error.append(p[u'math_error'].sum())
                        

            ##save loss for the convergance chassing 
            reduced_loss_train_log = reduce(lambda x, y: x+y, loss_list_train_log)
            last_train_losses.append(reduced_loss_train_log)
            
            ##every 'test_cycle' epochs test the testing set for sotmax/harmax loss
            if epoch_idx % cfg[u'test_cycle'] == 0 :
                
                #if sharing state, share state between training and testing data, lese
                #else completely reset the state
                if cfg[u'share_state'] is False:
                    _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
                    _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
 
            
                for batch_idx in xrange(num_test_batches):            
                            
                        start_idx = cfg[u'batch_size'] * batch_idx
                        end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]

                      
                        #print("computing rollout")
                        #rollout policites to get rewards
                        p = m.policy_rollout(sess, _current_state_test, _current_state_test_mem, batchX, batchY, cfg, False)
                        
                        loss_list_test_rewards.append( np.vstack(p[u'rewards']).sum() )
                        loss_list_test_math_error.append(p[u'math_error'].sum())
                
                #do an extra check for test cycle of the train data without backprop
                loss_list_train_rewards = [0,0]
                loss_list_train_math_error = [0,0]
                for batch_idx in xrange(num_batches):
                            
                    start_idx = cfg[u'batch_size'] * batch_idx
                    end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]
                    
                    #print("computing rollout")
                    #rollout policites to get rewards
                    u"""
                    print("train without backprop")
                    print("_current_state_train")
                    print(_current_state_train)
                    print("_current_state_train_mem")
                    print(_current_state_train_mem)
                    """
                    p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)
                    
                    loss_list_train_rewards.append( np.vstack(p[u'rewards']).sum() )
                    loss_list_train_math_error.append(p[u'math_error'].sum())
                    
                    if cfg[u'num_samples'] < 16:
                        print u"no backprop train data test"
                        print list(izip(np.hstack(p[u'selections']).tolist(),np.around(p[u'discount_rewards'], decimals=3).tolist()))
                        print list(izip( np.hstack(p[u'selections']).tolist(), np.hstack(p[u'mem_masks']).tolist() ))
                        print list(izip(np.hstack(p[u'selections_mem']).tolist(),np.around(p[u'discount_rewards'], decimals=3).tolist()))
                

            reduced_loss_train_log = reduced_loss_train_log
            reduced_loss_train_rewards = reduce(lambda x, y: x+y, loss_list_train_rewards)
            reduced_loss_train_math_error = reduce(lambda x, y: x+y, loss_list_train_math_error)

            reduced_loss_test_rewards = reduce(lambda x, y: x+y, loss_list_test_rewards)
            reduced_loss_test_soft = reduce(lambda x, y: x+y, loss_list_test_math_error)
                
            if epoch_idx % cfg[u'test_cycle'] == 0 :
                #save model            
                saver.save(sess, u'/home/rb7e15/2.7v/summaries/' + cfg[u'dst'] + u'/model/',global_step=epoch_idx)
                #write variables/loss summaries after all training/testing done
                #train_writer.add_summary(summary, epoch_idx)
                #write_no_tf_summary(train_writer, u"Log_train_loss",      reduced_loss_train_log, epoch_idx)               
                #write_no_tf_summary(train_writer, u"Rewards_train",      reduced_loss_train_rewards, epoch_idx)               
                #write_no_tf_summary(train_writer, u"Math_train_error",      reduced_loss_train_math_error, epoch_idx)               
                #write_no_tf_summary(train_writer, u"Rewards_test",      reduced_loss_test_rewards, epoch_idx)               
                #write_no_tf_summary(train_writer, u"Math_test_error",      reduced_loss_test_soft, epoch_idx)               
                
            print u""
            #harmax test

            print u"Epoch",epoch_idx
            print u"Log_train_loss\t", reduced_loss_train_log
            print u"Rewards_train\t", reduced_loss_train_rewards
            print u"Math_train_er\t", reduced_loss_train_math_error
            print u"Rewards_test\t", reduced_loss_test_rewards
            print u"Math_test_er\t", reduced_loss_test_soft

            print u"Epoch time: ", ((time.time() - startTime) % 60), u" Global Time: ",  get_time_hhmmss(time.time() - globalstartTime)
            print u"func: ", cfg[u'train_fn'].__name__, u"max_ops: ", cfg[u'max_output_ops'], u"sim_seed", cfg[u'seed'], u"tf seed", tf_ops.get_default_graph().seed

            if epoch_idx % cfg[u'convergance_check_epochs'] == 0 and epoch_idx >= cfg[u'convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print u"#################################"
                    print u"Model has converged, breaking ..."
                    print u"#################################"
                    break
                else:
                    print u"Reseting the loss conv array"
                    last_train_losses = []
            #also break on math error, as theres noice on gradients and model will not nesecerrilily converge
            if cfg[u'hardmax_break']:
                if (epoch_idx % cfg[u'test_cycle'] == 0) and (reduced_loss_train_math_error < 0.0000000001 or reduced_loss_test_soft < 0.0000000001):
                                                ## check thousand random samples
                        print u"@@checking random thousand samples"

                        #gen new seed
                        test_seed = round(random.random()*100000)
                        num_tests = 1000
                        print u"test_seed", test_seed, u"num_tests", num_tests
                        x_sample, y_sample = samples_generator(cfg[u'train_fn'], (num_tests, cfg[u'num_features']) , cfg[u'samples_value_rng'], test_seed)
                        match_count = 0
                        _current_state_train = np.zeros((1, cfg[u'state_size']))
                        _current_state_train_mem  = np.zeros((1, cfg[u'state_size']))
                        for i in xrange(num_tests):
                            batchX = np.zeros((cfg[u'batch_size']-1, cfg[u'num_features']))
                            batchX = np.concatenate(([x_sample[i]], batchX), axis=0)
                            p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, [x_sample[i]], [y_sample[i]], cfg, False)
                            match = np.allclose(y_sample[i], p[u'output'][0])
                            print u"i", i, u"match", match
                            print u"input", list(x_sample[i])
                            print u"expect", list(y_sample[i])
                            print u"actual", list(p[u'output'][0].tolist())
                            if match:
                                match_count = match_count + 1
                        print
                        print match_count, u"out of", num_tests,u"matched"          
                        
                        ##break
                        print u"#################################"
                        print u"Model reached hardmax, breaking ..."
                        print u"#################################"
                        break
    if cfg[u'relaunch']:
            cmd = gen_cmd_from_name(cfg[u"name"], cfg)
       	    fname = gen_job_file(cmd, cfg)
            cmd_qsub = get_qsub_com(cmd, fname)
            print u"ReLnch: " + cmd
            subprocess.Popen(cmd_qsub, shell=True, stderr=subprocess.STDOUT)

def restore_selection_RL_RNN(m, cfg, x_train, x_test, y_train, y_test, path):
    u"""
        #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())
    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #def batches num
    num_batches = x_train.shape[0]//cfg['batch_size']
    num_test_batches = x_test.shape[0]//cfg['batch_size']
    print("num batches train:", num_batches)
    print("num batches test:", num_test_batches)
    with tf.Session(config=config) as sess:
        ##enable debugger if necessary
        if (cfg['debug']):
            print("Running in a debug mode")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))
    """
    #create a saver to save the trained model
    saver=tf.train.Saver(var_list=tf.trainable_variables())
    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #def batches num
    num_batches = x_train.shape[0]//cfg[u'batch_size']
    num_test_batches = x_test.shape[0]//cfg[u'batch_size']
    print u"num batches train:", num_batches
    print u"num batches test:", num_test_batches
    with tf.Session(config=config) as sess:

        #init the var
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))
        # reset variables

        train_rewards = []
        train_math_error = []
        train_discount_rewards = []
        train_selections = []
        train_selections_mem = []
        train_outputs = []
        train_outputs_mem = []
        train_states = []
        train_states_mem = []
        train_current_exes = []
        train_current_exes_mem = []

        test_rewards = []
        test_math_error = []
        test_discount_rewards = []
        test_selections = []
        test_selections_mem = []
        test_outputs = []
        test_outputs_mem = []
        test_states = []
        test_states_mem = []
        test_current_exes = []
        test_current_exes_mem = []       

        
        #set states
        _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
        
        u'''
                return  dict(
                    discount_rewards = discount_rewards,
                    rewards = reeval_rewards,
                    math_error = math_error,
                    selections = selections,
                    selections_mem = selections_mem,
                    labels = labels,
                    labels_mem = labels_mem,
                    mem_masks = mem_masks,
                    log_probs = log_probs,
                    log_probs_mem = log_probs_mem,
                    states = states,
                    states_mem = states_mem,
                    current_exes = current_exes,
                    current_exes_mem = current_exes_mem,
                    outputs = outputs,
                    outputs_mem = outputs_mem
                )
        '''       
        
        
        #backprop and test training set for softmax and hardmax loss
        for batch_idx in xrange(num_batches):
                        
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]

                
                
                #print("computing rollout")
                #rollout policites to get rewards
                p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)           

                train_rewards.append( np.vstack(p[u'rewards']).sum() )
                train_math_error.append(p[u'math_error'].sum())
                train_discount_rewards.append(p[u'discount_rewards'])
                train_selections.append(p[u'selections'])
                train_selections_mem.append(p[u'selections_mem'])
                train_outputs.append(p[u'outputs'])
                train_outputs_mem.append(p[u'outputs_mem'])
                train_states.append(p[u'states'])
                train_states_mem.append(p[u'states_mem'])
                train_current_exes.append(p[u'current_exes'])
                train_current_exes_mem.append(p[u'current_exes_mem'])


            
            #if sharing state, share state between training and testing data, lese
            #else completely reset the state
        if cfg[u'share_state'] is False:
            _current_state_train = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_train_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
            _current_state_test_mem = np.zeros((cfg[u'batch_size'], cfg[u'state_size']))
 
        
        for batch_idx in xrange(num_test_batches):            
                        
            start_idx = cfg[u'batch_size'] * batch_idx
            end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

            batchX = x_test[start_idx:end_idx]
            batchY = y_test[start_idx:end_idx]

            
            #print("computing rollout")
            #rollout policites to get rewards
            p = m.policy_rollout(sess, _current_state_test, _current_state_test_mem, batchX, batchY, cfg, False)
            
            test_rewards.append( np.vstack(p[u'rewards']).sum() )
            test_math_error.append(p[u'math_error'].sum())
            test_discount_rewards.append(p[u'discount_rewards'])
            test_selections.append(p[u'selections'])
            test_selections_mem.append(p[u'selections_mem'])
            test_outputs.append(p[u'outputs'])
            test_outputs_mem.append(p[u'outputs_mem'])
            test_states.append(p[u'states'])
            test_states_mem.append(p[u'states_mem'])
            test_current_exes.append(p[u'current_exes'])
            test_current_exes_mem.append(p[u'current_exes_mem'])


        batchesX_train = []
        batchesY_train = []
        
        #FOR THE TRAINING DATA
        for batch_idx in xrange(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                batchesX_train.append(batchX)
                batchesY_train.append(batchY)
                
                
        batchesX_test = []
        batchesY_test = []
        
        #FOR THE TESTING DATA
        for batch_idx in xrange(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg[u'rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg[u'batch_size'] * batch_idx
                end_idx   = cfg[u'batch_size'] * batch_idx + cfg[u'batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                batchesX_test.append(batchX) 
                batchesY_test.append(batchY) 

        return dict(
                #outputs for the train data
                train_rewards = train_rewards,
                train_math_error = train_math_error,
                train_discount_rewards = train_discount_rewards,
                train_selections = train_selections,
                train_selections_mem = train_selections_mem,
                train_outputs = train_outputs,
                train_outputs_mem = train_outputs_mem,
                train_states = train_states,
                train_states_mem = train_states_mem,
                train_current_exes = train_current_exes,
                train_current_exes_mem = train_current_exes_mem,
            
                #outputs for the testing data
                test_rewards = test_rewards,
                test_math_error = test_math_error,
                test_discount_rewards = test_discount_rewards,
                test_selections = test_selections,
                test_selections_mem = test_selections_mem,
                test_outputs = test_outputs,
                test_outputs_mem = test_outputs_mem,
                test_states = test_states,
                test_states_mem = test_states_mem,
                test_current_exes = test_current_exes,
                test_current_exes_mem = test_current_exes_mem,
            
                #return divided up batches
                batchesX_train = batchesX_train,
                batchesY_train = batchesY_train,
                batchesX_test = batchesX_test,
                batchesY_test = batchesY_test
        )
