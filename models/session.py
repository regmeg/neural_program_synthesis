import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops as tf_ops
import pprint
import random
import time
import os
import sys
import datetime
from data_gen import *
from params import get_cfg
from rnn_base import RNN
from ops import Operations

def write_no_tf_summary(writer, tag, val, step):
   summary=tf.Summary()
   summary.value.add(tag=tag, simple_value = val)
   writer.add_summary(summary, step)
    

#helpder func
def get_time_hhmmss(dif):
    m, s = divmod(dif, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

def determine_loss(epoch, cfg):
    period = cfg["loss_swap_per"]
    mod = epoch % (period*2)
    if mod < period: return True
    else :           return False

def run_session_2RNNS(m, cfg, x_train, x_test, y_train, y_test):
    #pre training setting
    np.set_printoptions(precision=3, suppress=True)
    #train_fn = np_mult
    #train_fn = np_stall
    num_batches = x_train.shape[0]//cfg['batch_size']
    num_test_batches = x_test.shape[0]//cfg['batch_size']
    print("num batches train:", num_batches)
    print("num batches test:", num_test_batches)
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
        train_writer = tf.summary.FileWriter('./summaries/' + cfg['dst'] ,sess.graph)
        ##enable debugger if necessary
        if (cfg['debug']):
            print("Running in a debug mode")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        #plt.ion()
        #plt.figure()
        #plt.show() 
        #Init vars:
        _W = sess.run([m.params['W']])
        _W2 = sess.run([m.params['W2']])
        _softmax_sat = sess.run([m.softmax_sat])
        print(m.params['W'].eval())
        print(m.params['W2'].eval())
        print(m.softmax_sat.eval())
        globalstartTime = time.time()
        for epoch_idx in range(cfg['num_epochs']):
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
            
            '''
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
            
            #set states
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            
                #backprop and test training set for softmax and hardmax loss
            for batch_idx in range(num_batches):
                
                    #if flag set, make op and mem selection rnn use exaclty the same state
                    if cfg['rnns_same_state'] is True:
                        _current_state_train_mem = _current_state_train
                        _current_state_test_mem  = _current_state_test
                            
                    start_idx = cfg['batch_size'] * batch_idx
                    end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]

                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data
                    if epoch_idx % cfg['test_cycle'] != 0 :                       
                      
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _output_train,\
                        _grads,\
                        _softmaxes_train,\
                        _math_error_train = sess.run([m.total_loss_train, 
                                                      m.train_step,
                                                      m.train["current_state"], 
                                                      m.train["current_state_mem"], 
                                                      m.train["output"], 
                                                      m.grads, 
                                                      m.train["softmaxes"],
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
                                                      m.train["current_state"],
                                                      m.train["current_state_mem"],
                                                      m.train["output"],
                                                      m.grads,
                                                      m.train["softmaxes"],
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
                                                     m.test["current_state"],
                                                     m.test["current_state_mem"],
                                                     m.test["output"],
                                                     m.test["softmaxes"],
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
            if epoch_idx % cfg['test_cycle'] == 0 :
                
                #if sharing state, share state between training and testing data, lese
                #else completely reset the state
                if cfg['share_state'] is False:
                    _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
 
            
                for batch_idx in range(num_test_batches):
            
                        #if flag set, make op and mem selection rnn use exaclty the same state
                        if cfg['rnns_same_state'] is True:
                            _current_state_train_mem = _current_state_train
                            _current_state_test_mem  = _current_state_test
                            
                        start_idx = cfg['batch_size'] * batch_idx
                        end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]

                        _total_loss_train,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _math_error_train        = sess.run([m.total_loss_train,
                                                             m.train["current_state"],
                                                             m.train["current_state_mem"],
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
                                                            m.test["current_state"],
                                                            m.test["current_state_mem"],
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
                saver.save(sess, './summaries/' + cfg['dst'] + '/model/',global_step=epoch_idx)
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
                
            if epoch_idx % cfg['test_cycle'] == 0 :
                train_writer.add_summary(summary, epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_train_loss",      reduced_loss_train_soft, epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_math_train_loss", reduced_math_error_train_soft , epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_pen_train_loss",  pen_loss_train_soft, epoch_idx)
                
                write_no_tf_summary(train_writer, "Hardmax_train_loss",      reduced_loss_train_hard, epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_math_train_loss", reduced_math_error_train_hard, epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_pen_train_loss",  pen_loss_train_hard, epoch_idx)
                
                write_no_tf_summary(train_writer, "Softmax_test_loss",      reduced_loss_test_soft, epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_math_test_loss", reduced_math_error_test_soft, epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_pen_test_loss",  pen_loss_test_soft, epoch_idx)
                
                write_no_tf_summary(train_writer, "Hardmax_test_loss",      reduced_loss_test_hard, epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_math_test_loss", reduced_math_error_test_hard, epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_pen_test_loss",  pen_loss_test_hard, epoch_idx)

            print("")
            #harmax test
            '''
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
            print("Epoch",epoch_idx, "use_both_losses", use_both_losses)
            print("Softmax train loss\t", reduced_loss_train_soft, "(m:",reduced_math_error_train_soft ,"p:",pen_loss_train_soft,")")
            print("Hardmax train loss\t", reduced_loss_train_hard, "(m:",reduced_math_error_train_hard ,"p:",pen_loss_train_hard,")")
            print("Sotfmax test loss\t", reduced_loss_test_soft, "(m:",reduced_math_error_test_soft ,"p:",pen_loss_test_soft,")")
            print("Hardmax test loss\t", reduced_loss_test_hard, "(m:",reduced_math_error_test_hard ,"p:",pen_loss_test_hard,")")
            print("Epoch time: ", ((time.time() - startTime) % 60), " Global Time: ",  get_time_hhmmss(time.time() - globalstartTime))
            print("func: ", cfg['train_fn'].__name__, "max_ops: ", cfg['max_output_ops'], "sim_seed", cfg['seed'], "tf seed", tf_ops.get_default_graph().seed)
            #print("grads[0] - W", _grads[0][0])
            #print("grads[1] - b", _grads[1][0])
            #print("grads[2] - W2", _grads[2][0])
            #print("grads[3] - b2", _grads[3][0])
            #print("W", W.eval())
            #print("w2" , W2.eval())
            #record execution timeline
            ##check convergance over last 5000 epochs
            if epoch_idx % cfg['convergance_check_epochs'] == 0 and epoch_idx >= cfg['convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print("#################################")
                    print("Model has converged, breaking ...")
                    print("#################################")
                    break
                else:
                    print("Reseting the loss conv array")
                    last_train_losses = []


            #as well check early stopping options, once hardmax train error is small enough - there is not point to check softmax, as its combinations of math error and penalties
            if (epoch_idx % cfg['test_cycle'] == 0) and ((reduced_loss_train_hard < 10) or (reduced_loss_test_hard < 10)):
                    print("#################################")
                    print("Model reached hardmax, breaking ...")
                    print("#################################")
                    break


def restore_selection_matrixes2RNNS(m, cfg, x_train, x_test, y_train, y_test, path):
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

        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))

        #FOR THE TRAINING DATA
        for batch_idx in range(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

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
                                                     m.train["outputs"],
                                                     m.train["outputs_mem"],
                                                     m.train["softmaxes"],
                                                     m.train["softmaxes_mem"],
                                                     m.train["current_state"],
                                                     m.train["current_state_mem"]],
                                                
                                                
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
                                                     m.test["outputs"],
                                                     m.test["outputs_mem"],
                                                     m.test["softmaxes"],
                                                     m.test["softmaxes_mem"],
                                                     m.test["current_state"],
                                                     m.test["current_state_mem"]],
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
        
        if cfg['share_state'] is False:
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        
        #FOR THE TESTING DATA
        for batch_idx in range(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

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
                                                     m.train["outputs"],
                                                     m.train["outputs_mem"],
                                                     m.train["softmaxes"],
                                                     m.train["softmaxes_mem"],
                                                     m.train["current_state"],
                                                     m.train["current_state_mem"]],
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
                                                     m.test["outputs"],
                                                     m.test["outputs_mem"],
                                                     m.test["softmaxes"],
                                                     m.test["softmaxes_mem"],
                                                     m.test["current_state"],
                                                     m.test["current_state_mem"]],
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
        for batch_idx in range(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                batchesX_train.append(batchX)
                batchesY_train.append(batchY)
                
                
        batchesX_test = []
        batchesY_test = []
        
        #FOR THE TESTING DATA
        for batch_idx in range(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

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

def predict_form_sess(m, cfg, x, state, state_mem, path, mode="hard"):
    #create a saver to restore saved model
    saver=tf.train.Saver(var_list=tf.trainable_variables())

    #Enable jit
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:

        #init the var
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path))      
        
        batchX = np.zeros((cfg['batch_size']-x.shape[0], cfg['num_features']))
        batchY = np.zeros((cfg['batch_size'], cfg['num_features']))
        
        batchX = np.concatenate((x, batchX), axis=0)

        if mode == "soft":
            output = sess.run([m.output_train],
                feed_dict={
                    m.init_state:state,
                    m.mem.init_state:state_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
        elif mode == "hard":
            output = sess.run([m.output_test],
                feed_dict={
                    m.init_state:state,
                    m.mem.init_state:state_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
        else: raise("Wrong mode selected for predicting variable")
            
    return output

def run_session_HistoryRNN(m, cfg, x_train, x_test, y_train, y_test):
    #pre training setting
    np.set_printoptions(precision=3, suppress=True)
    #train_fn = np_mult
    #train_fn = np_stall
    num_batches = x_train.shape[0]//cfg['batch_size']
    num_test_batches = x_test.shape[0]//cfg['batch_size']
    print("num batches train:", num_batches)
    print("num batches test:", num_test_batches)
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
        train_writer = tf.summary.FileWriter('./summaries/' + cfg['dst'] ,sess.graph)
        ##enable debugger if necessary
        if (cfg['debug']):
            print("Running in a debug mode")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #init the var
        sess.run(tf.global_variables_initializer())
        #plt.ion()
        #plt.figure()
        #plt.show() 
        #Init vars:
        _W = sess.run([m.params['W_hist']])
        _W2 = sess.run([m.params['W2_mem']])
        print(m.params['W_hist'].eval())
        print(m.params['W2_mem'].eval())
        globalstartTime = time.time()
        for epoch_idx in range(cfg['num_epochs']):
            # reset variables
            startTime = time.time()
            loss_list_train_soft = [0,0]
            loss_list_train_hard = [0,0]
            loss_list_test_soft = [0,0]
            loss_list_test_hard = [0,0]
            summary = None
            #shuffle data
            #x_train, y_train = shuffle_data(x_train, y_train)

            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            
                #backprop and test training set for softmax and hardmax loss
            for batch_idx in range(num_batches):
                            
                    start_idx = cfg['batch_size'] * batch_idx
                    end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]

                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data
                    if epoch_idx % cfg['test_cycle'] != 0 :                       
                      
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _output_train,\
                        _grads,\
                        _math_error_train = sess.run([m.total_loss_train, 
                                                      m.train_step,
                                                      m.train["current_state"], 
                                                      m.train["output"], 
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
                                                      m.train["current_state"],
                                                      m.train["output"],
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
                                                     m.test["current_state"],
                                                     m.test["output"],
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
            if epoch_idx % cfg['test_cycle'] == 0 :
                
                #if sharing state, share state between training and testing data, 
                #else completely reset the state
                if cfg['share_state'] is False:
                    _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            
                for batch_idx in range(num_test_batches):           
                           
                        start_idx = cfg['batch_size'] * batch_idx
                        end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]

                        _total_loss_train,\
                        _current_state_train = sess.run([m.total_loss_train,
                                                             m.train["current_state"]],
                            feed_dict={
                                m.init_state:_current_state_train,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY
                            })
                        loss_list_test_soft.append(_total_loss_train)

                        _total_loss_test,\
                        _current_state_test = sess.run([m.total_loss_test,
                                                        m.test["current_state"]],
                            feed_dict={
                                m.init_state:_current_state_test,
                                m.batchX_placeholder:batchX,
                                m.batchY_placeholder:batchY
                            })
                        loss_list_test_hard.append(_total_loss_test)

                #save model            
                saver.save(sess, './summaries/' + cfg['dst'] + '/model/',global_step=epoch_idx)
                #write variables/loss summaries after all training/testing done
                train_writer.add_summary(summary, epoch_idx)
                write_no_tf_summary(train_writer, "Softmax_train_loss", reduced_loss_train_soft, epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_train_loss", reduce(lambda x, y: x+y, loss_list_train_hard), epoch_idx)
                write_no_tf_summary(train_writer, "Sotfmax_test_loss", reduce(lambda x, y: x+y, loss_list_test_soft), epoch_idx)
                write_no_tf_summary(train_writer, "Hardmax_test_loss", reduce(lambda x, y: x+y, loss_list_test_hard), epoch_idx)

            print("")
            #harmax test
            '''
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
            print("Epoch",epoch_idx)
            print("Softmax train loss\t", reduced_loss_train_soft)
            print("Hardmax train loss\t", reduce(lambda x, y: x+y, loss_list_train_hard))
            print("Sotfmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_soft))
            print("Hardmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_hard))
            print("Epoch time: ", ((time.time() - startTime) % 60), " Global Time: ",  get_time_hhmmss(time.time() - globalstartTime))
            print("func: ", cfg['train_fn'].__name__, "max_ops: ", cfg['max_output_ops'], "sim_seed", cfg['seed'], "tf seed", tf_ops.get_default_graph().seed)
            #print("grads[0] - W", _grads[0][0])
            #print("grads[1] - b", _grads[1][0])
            #print("grads[2] - W2", _grads[2][0])
            #print("grads[3] - b2", _grads[3][0])
            #print("W", W.eval())
            #print("w2" , W2.eval())
            #record execution timeline
            ##check convergance over last 5000 epochs
            if epoch_idx % cfg['convergance_check_epochs'] == 0 and epoch_idx >= cfg['convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print("#################################")
                    print("Model has converged, breaking ...")
                    print("#################################")
                    break
                else:
                    print("Reseting the loss conv array")
                    last_train_losses = []
                    
def restore_selection_matrixes_HistoryRNNS(m, cfg, x_train, x_test, y_train, y_test, path):
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

        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
    

        #FOR THE TRAINING DATA
        for batch_idx in range(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]

               
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_traind_train,\
                _outputs_traind_train_op,\
                _outputs_traind_train_mem,\
                _softmaxes_traind_train_op,\
                _softmaxes_traind_train_mem,\
                _current_state_train     = sess.run([m.total_loss_train,
                                                     m.train["outputs_op"],
                                                     m.train["outputs_mem"],
                                                     m.train["softmaxes_op"],
                                                     m.train["softmaxes_mem"],
                                                     m.train["current_state"]],
                                                
                                                
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
                                                     m.test["outputs_op"],
                                                     m.test["outputs_mem"],
                                                     m.test["softmaxes_op"],
                                                     m.test["softmaxes_mem"],
                                                     m.test["current_state"]],
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
        
        if cfg['share_state'] is False:
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
        
        #FOR THE TESTING DATA
        for batch_idx in range(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                #FOR THE SOFTMAX SELECTION
                _total_loss_testd_train,\
                _outputs_testd_train_op,\
                _outputs_testd_train_mem,\
                _softmaxes_testd_train_op,\
                _softmaxes_testd_train_mem,\
                _current_state_train     = sess.run([m.total_loss_train,
                                                     m.train["outputs_op"],
                                                     m.train["outputs_mem"],
                                                     m.train["softmaxes_op"],
                                                     m.train["softmaxes_mem"],
                                                     m.train["current_state"]],
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
                                                     m.test["outputs_op"],
                                                     m.test["outputs_mem"],
                                                     m.test["softmaxes_op"],
                                                     m.test["softmaxes_mem"],
                                                     m.test["current_state"]],
                                
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
        for batch_idx in range(num_batches):
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
            
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                batchesX_train.append(batchX)
                batchesY_train.append(batchY)
                
                
        batchesX_test = []
        batchesY_test = []
        
        #FOR THE TESTING DATA
        for batch_idx in range(num_test_batches):
            
                #if flag set, make op and mem selection rnn use exaclty the same state
                if cfg['rnns_same_state'] is True:
                    _current_state_train_mem = _current_state_train
                    _current_state_test_mem  = _current_state_test
                    
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

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