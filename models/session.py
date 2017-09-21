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
import subprocess
from data_gen import *
from params import get_cfg
from rnn_base import RNN
from ops import Operations


def write_no_tf_summary(writer, tag, val, step):
   summary=tf.Summary()
   summary.value.add(tag=tag, simple_value = val)
   writer.add_summary(summary, step)

def print_ops_matrix(matrix_lst, ops_list, indeces = None):
    np.set_printoptions(precision=3, suppress=True)
    for elem in range(len(matrix_lst[0])):
        if indeces is not None and elem not in indeces: continue 
        for matrix in matrix_lst:
            index = np.argmax(matrix[elem])
            if len(matrix[elem]) < 2:
                index = matrix[elem][0]
            op_name = ops_list[index].__name__
            print(str(matrix[elem][index])+"[ "+op_name+" ]", end=" ")
        print("")

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


def gen_cmd_from_name(name_in, cfg):
    new_seed = int(round(random.random()*100000))
    tokens = name_in.split("~")
    string = "python3 ./model.py"
    name = " --name="
    for token in tokens:
        key, val = token.split("#")
        if key == 'grad_clip_val':
            val = val.split("*")
            string += " --"+str(key)+"_min="+str(val[0])
            string += " --"+str(key)+"_max="+str(val[1])
            name += str(key)+"#"+str(val[0])+"*"+str(val[1])+"~"
        elif key == 'seed': continue
        else:
            string += " --"+str(key)+"="+str(val)
            name += str(key)+"#"+str(val)+"~"
    string += " --max_output_ops="+str(cfg["max_output_ops"])
    string += " --train_fn="+str(cfg["train_fn"].__name__)
    string += " --model="+str(cfg["model"])
    name += "seed#" + str(new_seed)
    seed  = " --seed="+str(new_seed)
    return string + seed + name
    
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
        _W3 = sess.run([m.params['W3']])
        
        _W_mem = sess.run([m.params['W_mem']])
        _W2_mem = sess.run([m.params['W2_mem']])
        #_W3_mem = sess.run([m.params['W3_mem']])
        
        print("W1")
        print(m.params['W'].eval())
        print("W2")
        print(m.params['W2'].eval())
        print("W3")
        print(m.params['W3'].eval())
        
        print("W1_mem")
        print(m.params['W_mem'].eval())
        print("W2_mem")
        print(m.params['W2_mem'].eval())
        
        #print("W3_mem")
        #print(m.params['W3_mem'].eval())
        
        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        
        
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
                    
                                
                    #set states
                    _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                    _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            

                    #for non testing cylce, simply do one forward and back prop with 1 batch with train data
                    if epoch_idx % cfg['test_cycle'] != 0 :                       
                      
                        _total_loss_train,\
                        _train_step,\
                        _current_state_train,\
                        _current_state_train_mem,\
                        _output_train,\
                        _grads,\
                        _softmaxes_train,\
                        _softmaxes_train_mem,\
                        _math_error_train = sess.run([m.total_loss_train, 
                                                      m.train_step,
                                                      m.train["current_state"], 
                                                      m.train["current_state_mem"], 
                                                      m.train["output"], 
                                                      m.grads, 
                                                      m.train["softmaxes"],
                                                      m.train["softmaxes_mem"],
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
                        _softmaxes_train_mem,\
                        _math_error_train = sess.run([merged,
                                                      m.total_loss_train,
                                                      m.train_step,
                                                      m.train["current_state"],
                                                      m.train["current_state_mem"],
                                                      m.train["output"],
                                                      m.grads,
                                                      m.train["softmaxes"],
                                                      m.train["softmaxes_mem"],
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
            
                for batch_idx in range(num_test_batches):
            
                        #if flag set, make op and mem selection rnn use exaclty the same state
                        if cfg['rnns_same_state'] is True:
                            _current_state_train_mem = _current_state_train
                            _current_state_test_mem  = _current_state_test
                            
                        start_idx = cfg['batch_size'] * batch_idx
                        end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]
                        
                        
                        #set states
                        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                        _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
                        _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                        _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                        
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
            print("#################Ops####################")
            print_ops_matrix(_softmaxes_train,cfg["used_ops_obj"])
            print("#################Mem####################")
            print_ops_matrix(_softmaxes_train_mem,cfg["used_ops_obj_mem"])
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
            if cfg['hardmax_break']:
                if (epoch_idx % cfg['test_cycle'] == 0) and ((reduced_loss_train_hard < 0.01) or (reduced_loss_test_hard < 0.01)):
                        ## check thousand random samples
                        print("@@checking random thousand samples")

                        #gen new seed
                        test_seed = round(random.random()*100000)
                        num_tests = 1000
                        print("test_seed", test_seed, "num_tests", num_tests)
                        x_sample, y_sample = samples_generator(cfg['train_fn'], (num_tests, cfg['num_features']) , cfg['samples_value_rng'], test_seed)
                        match_count = 0
                        for i in range(num_tests):
                            batchX = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                            batchX = np.concatenate(([x_sample[i]], batchX), axis=0)  
                            output = sess.run([m.test['output']],
                                feed_dict={
                                        m.init_state:np.zeros((cfg['batch_size'], cfg['state_size'])),
                                        m.mem.init_state:np.zeros((cfg['batch_size'], cfg['state_size'])),
                                        m.batchX_placeholder:batchX
                                    })
                            match = np.allclose(y_sample[i], output[0][0])
                            print("i", i , "match", match)
                            print("input", list(x_sample[i]))
                            print("expect", list(y_sample[i]))
                            print("actual", list(output[0][0].tolist()))
                            if match:
                                match_count = match_count + 1
                        print()
                        print(match_count, "out of", num_tests,"matched")
                        
                        print("#################################")
                        print("Model reached hardmax, breaking ...")
                        print("#################################")                                                     

          
                        
                        ##break
                        break

    if cfg['relaunch']:
            cmd = gen_cmd_from_name(cfg["name"], cfg)
            print("ReLnch: " + cmd)
            subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)


def restore_selection_matrixes2RNNS(m, cfg, x_train, x_test, y_train, y_test, path, test_1000):
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

                _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
               
                
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
                
                _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
                
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
        
        if test_1000:
            test_1000_samples_RNN(m, sess, cfg)
        
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
                batchesY_test = batchesY_test,
                
                #resturn session
                sess = sess
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
            output = sess.run([m.train['output']],
                feed_dict={
                    m.init_state:state,
                    m.mem.init_state:state_mem,
                    m.batchX_placeholder:batchX,
                    m.batchY_placeholder:batchY
                })
        elif mode == "hard":
            output = sess.run([m.test['output']],
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
                    
def restore_selection_matrixes_HistoryRNNS(m, cfg, x_train, x_test, y_train, y_test, path, test_1000):
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
def run_session_RL_RNN(m, cfg, x_train, x_test, y_train, y_test):
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
        _W3 = sess.run([m.params['W3']])
        
        _W_mem = sess.run([m.params['W_mem']])
        _W2_mem = sess.run([m.params['W2_mem']])
        _W3_mem = sess.run([m.params['W3_mem']])
        
        print("W1")
        print(m.params['W'].eval())
        print("W2")
        print(m.params['W2'].eval())
        print("W3")
        print(m.params['W3'].eval())
        
        print("W1_mem")
        print(m.params['W_mem'].eval())
        print("W2_mem")
        print(m.params['W2_mem'].eval())
        print("W3_mem")
        print(m.params['W3_mem'].eval())
        
        globalstartTime = time.time()
        for epoch_idx in range(cfg['num_epochs']):
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
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            
                #backprop and test training set for softmax and hardmax loss
            for batch_idx in range(num_batches):
                            
                    start_idx = cfg['batch_size'] * batch_idx
                    end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]
                    
                    #print("computing rollout")
                    #rollout policites to get rewards
                    p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, True)
                    
                    #maks non mem cases for the mem backprop
                    width = np.vstack(p['states_mem']).shape[1]
                    mask = np.lib.pad(np.vstack(p['mem_masks']), ((0, 0), (0,  width - 1)), 'edge')                       
                    mem_state = np.vstack(p['states_mem'])[mask == True].reshape(-1, width)                        
                    
                    #width = np.vstack(p['current_exes_mem']).shape[1]
                    #mask = np.lib.pad(np.vstack(p['mem_masks']), ((0, 0), (0,  width - 1)), 'edge')                       
                    #mem_x = np.vstack(p['current_exes_mem'])[mask == True].reshape(-1, width)
                    #use same exes for the mem
                    width = np.vstack(p['current_exes']).shape[1]
                    mask = np.lib.pad(np.vstack(p['mem_masks']), ((0, 0), (0,  width - 1)), 'edge')                       
                    mem_x = np.vstack(p['current_exes'])[mask == True].reshape(-1, width)
                    
                    width = np.vstack(p['labels_mem']).shape[1]
                    mask = np.lib.pad(np.vstack(p['mem_masks']), ((0, 0), (0,  width - 1)), 'edge')                       
                    mem_y = np.vstack(p['labels_mem'])[mask == True].reshape(-1, width)
                    
                    mem_sel = np.vstack(p['selections_mem'])[np.vstack(p['mem_masks']) == True]
                    mem_rews = np.vstack(np.hstack(np.stack(p['discount_rewards'], axis=1)))[np.vstack(p['mem_masks']) == True]                   
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

                    if epoch_idx % cfg['test_cycle'] != 0 :                       
                      
                       
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
                                        m.init_state:np.vstack(p['states']),

                                        m.batchX_placeholder:np.vstack(p['current_exes']),

                                        m.batchY_placeholder:np.vstack(p['labels']),

                                        m.selections_placeholder:np.vstack(p['selections']),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p['discount_rewards'], axis=1))),
                                        
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
                                        m.init_state:np.vstack(p['states']),
                                        m.mem.init_state: np.vstack(mem_state),

                                        m.batchX_placeholder:np.vstack(p['current_exes']),
                                        m.mem.batchX_placeholder: np.vstack(mem_x),

                                        m.batchY_placeholder:np.vstack(p['labels']),
                                        m.mem.batchY_placeholder: np.vstack(mem_y),

                                        m.selections_placeholder:np.vstack(p['selections']),
                                        m.mem.selections_placeholder: np.vstack(mem_sel),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p['discount_rewards'], axis=1))),
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
                                        m.init_state:np.vstack(p['states']),

                                        m.batchX_placeholder:np.vstack(p['current_exes']),

                                        m.batchY_placeholder:np.vstack(p['labels']),

                                        m.selections_placeholder:np.vstack(p['selections']),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p['discount_rewards'], axis=1))),
                                        
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
                                        m.init_state:np.vstack(p['states']),
                                        m.mem.init_state: np.vstack(mem_state),

                                        m.batchX_placeholder:np.vstack(p['current_exes']),
                                        m.mem.batchX_placeholder: np.vstack(mem_x),

                                        m.batchY_placeholder:np.vstack(p['labels']),
                                        m.mem.batchY_placeholder: np.vstack(mem_y),

                                        m.selections_placeholder:np.vstack(p['selections']),
                                        m.mem.selections_placeholder: np.vstack(mem_sel),

                                        m.rewards_placeholder:np.vstack(np.hstack(np.stack(p['discount_rewards'], axis=1))),
                                        m.mem.rewards_placeholder: np.vstack(mem_rews),
                                        
                                        m.training: True

                                })
                        
                    if cfg['num_samples'] < 16:
                            print(list(zip(np.hstack(p['selections']).tolist(),np.around(p['discount_rewards'], decimals=3).tolist())))
                            #print(list(zip( np.hstack(p['selections']).tolist(), np.hstack(p['mem_masks']).tolist() )))
                            #print(list(zip(np.hstack(p['selections_mem']).tolist(),np.around(p['discount_rewards'], decimals=3).tolist())))
                            if no_mem_bprop:
                                print("no_mem_sel_ops")
                            else:
                                print(list(zip(np.hstack(mem_sel).tolist(),np.around(mem_rews, decimals=3).tolist())))
                            print('""')
                    #if p['math_error'].sum() > 0.0000000001:
                    loss_list_train_log.append(_total_loss_train)
                    loss_list_train_rewards.append( np.vstack(p['rewards']).sum() )
                    loss_list_train_math_error.append(p['math_error'].sum())
                        

            ##save loss for the convergance chassing 
            reduced_loss_train_log = reduce(lambda x, y: x+y, loss_list_train_log)
            last_train_losses.append(reduced_loss_train_log)
            
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
                            
                        start_idx = cfg['batch_size'] * batch_idx
                        end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                        batchX = x_test[start_idx:end_idx]
                        batchY = y_test[start_idx:end_idx]

                      
                        #print("computing rollout")
                        #rollout policites to get rewards
                        p = m.policy_rollout(sess, _current_state_test, _current_state_test_mem, batchX, batchY, cfg, False)
                        
                        loss_list_test_rewards.append( np.vstack(p['rewards']).sum() )
                        loss_list_test_math_error.append(p['math_error'].sum())
                
                #do an extra check for test cycle of the train data without backprop
                loss_list_train_rewards = [0,0]
                loss_list_train_math_error = [0,0]
                for batch_idx in range(num_batches):
                            
                    start_idx = cfg['batch_size'] * batch_idx
                    end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                    batchX = x_train[start_idx:end_idx]
                    batchY = y_train[start_idx:end_idx]
                    
                    #print("computing rollout")
                    #rollout policites to get rewards
                    """
                    print("train without backprop")
                    print("_current_state_train")
                    print(_current_state_train)
                    print("_current_state_train_mem")
                    print(_current_state_train_mem)
                    """
                    p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)
                    
                    loss_list_train_rewards.append( np.vstack(p['rewards']).sum() )
                    loss_list_train_math_error.append(p['math_error'].sum())
                    
                    if cfg['num_samples'] < 16:
                        print("no backprop train data test")
                        print(list(zip(np.hstack(p['selections']).tolist(),np.around(p['discount_rewards'], decimals=3).tolist())))
                        print(list(zip( np.hstack(p['selections']).tolist(), np.hstack(p['mem_masks']).tolist() )))
                        print(list(zip(np.hstack(p['selections_mem']).tolist(),np.around(p['discount_rewards'], decimals=3).tolist())))
                

            reduced_loss_train_log = reduced_loss_train_log
            reduced_loss_train_rewards = reduce(lambda x, y: x+y, loss_list_train_rewards)
            reduced_loss_train_math_error = reduce(lambda x, y: x+y, loss_list_train_math_error)

            reduced_loss_test_rewards = reduce(lambda x, y: x+y, loss_list_test_rewards)
            reduced_loss_test_soft = reduce(lambda x, y: x+y, loss_list_test_math_error)
                
            if epoch_idx % cfg['test_cycle'] == 0 :
                #save model            
                saver.save(sess, './summaries/' + cfg['dst'] + '/model/',global_step=epoch_idx)
                #write variables/loss summaries after all training/testing done
                train_writer.add_summary(summary, epoch_idx)
                write_no_tf_summary(train_writer, "Log_train_loss",      reduced_loss_train_log, epoch_idx)               
                write_no_tf_summary(train_writer, "Rewards_train",      reduced_loss_train_rewards, epoch_idx)               
                write_no_tf_summary(train_writer, "Math_train_error",      reduced_loss_train_math_error, epoch_idx)               
                write_no_tf_summary(train_writer, "Rewards_test",      reduced_loss_test_rewards, epoch_idx)               
                write_no_tf_summary(train_writer, "Math_test_error",      reduced_loss_test_soft, epoch_idx)               
                
            print("")
            #harmax test

            print("Epoch",epoch_idx)
            print("Log_train_loss\t", reduced_loss_train_log)
            print("Rewards_train\t", reduced_loss_train_rewards)
            print("Math_train_er\t", reduced_loss_train_math_error)
            print("Rewards_test\t", reduced_loss_test_rewards)
            print("Math_test_er\t", reduced_loss_test_soft)

            print("Epoch time: ", ((time.time() - startTime) % 60), " Global Time: ",  get_time_hhmmss(time.time() - globalstartTime))
            print("func: ", cfg['train_fn'].__name__, "max_ops: ", cfg['max_output_ops'], "sim_seed", cfg['seed'], "tf seed", tf_ops.get_default_graph().seed)

            if epoch_idx % cfg['convergance_check_epochs'] == 0 and epoch_idx >= cfg['convergance_check_epochs']: 
                if np.allclose(last_train_losses, last_train_losses[0], equal_nan=True, rtol=1e-05, atol=1e-02):
                    print("#################################")
                    print("Model has converged, breaking ...")
                    print("#################################")
                    break
                else:
                    print("Reseting the loss conv array")
                    last_train_losses = []
            #also break on math error, as theres noice on gradients and model will not nesecerrilily converge
            if cfg['hardmax_break']:
                if (epoch_idx % cfg['test_cycle'] == 0) and (reduced_loss_train_math_error < 0.0000000001 or reduced_loss_test_soft < 0.0000000001):
                                                ## check thousand random samples
                        print("@@checking random thousand samples")

                        #gen new seed
                        test_seed = round(random.random()*100000)
                        num_tests = 1000
                        print("test_seed", test_seed, "num_tests", num_tests)
                        x_sample, y_sample = samples_generator(cfg['train_fn'], (num_tests, cfg['num_features']) , cfg['samples_value_rng'], test_seed)
                        match_count = 0
                        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
                        _current_state_train_mem  = np.zeros((cfg['batch_size'], cfg['state_size']))
                        for i in range(num_tests):
                            batchX = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                            batchX = np.concatenate(([x_sample[i]], batchX), axis=0)
                            
                            batchY = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                            batchY = np.concatenate(([y_sample[i]], batchY), axis=0)
                            
                            p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)
                            match = np.allclose(y_sample[i], p['output'][0])
                            print("i", i , "match", match)
                            print("input", list(x_sample[i]))
                            print("expect", list(y_sample[i]))
                            print("actual", list(p['output'][0].tolist()))
                            if match:
                                match_count = match_count + 1
                        print()
                        print(match_count, "out of", num_tests,"matched")          
                        
                        ##break
                        print("#################################")
                        print("Model reached hardmax, breaking ...")
                        print("#################################")
                        break
    if cfg['relaunch']:
            cmd = gen_cmd_from_name(cfg["name"], cfg)
            print("ReLnch: " + cmd)
            subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)

def restore_selection_RL_RNN(m, cfg, x_train, x_test, y_train, y_test, path, test_1000):
    """
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
    num_batches = x_train.shape[0]//cfg['batch_size']
    num_test_batches = x_test.shape[0]//cfg['batch_size']
    print("num batches train:", num_batches)
    print("num batches test:", num_test_batches)
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
        _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
        
        '''
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
        for batch_idx in range(num_batches):
                        
                start_idx = cfg['batch_size'] * batch_idx
                end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]

                
                
                #print("computing rollout")
                #rollout policites to get rewards
                p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)           

                train_rewards.append( np.vstack(p['rewards']).sum() )
                train_math_error.append(p['math_error'].sum())
                train_discount_rewards.append(p['discount_rewards'])
                train_selections.append(p['selections'])
                train_selections_mem.append(p['selections_mem'])
                train_outputs.append(p['outputs'])
                train_outputs_mem.append(p['outputs_mem'])
                train_states.append(p['states'])
                train_states_mem.append(p['states_mem'])
                train_current_exes.append(p['current_exes'])
                train_current_exes_mem.append(p['current_exes_mem'])


            
            #if sharing state, share state between training and testing data, lese
            #else completely reset the state
        if cfg['share_state'] is False:
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_train_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test_mem = np.zeros((cfg['batch_size'], cfg['state_size']))
 
        
        for batch_idx in range(num_test_batches):            
                        
            start_idx = cfg['batch_size'] * batch_idx
            end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

            batchX = x_test[start_idx:end_idx]
            batchY = y_test[start_idx:end_idx]

            
            #print("computing rollout")
            #rollout policites to get rewards
            p = m.policy_rollout(sess, _current_state_test, _current_state_test_mem, batchX, batchY, cfg, False)
            
            test_rewards.append( np.vstack(p['rewards']).sum() )
            test_math_error.append(p['math_error'].sum())
            test_discount_rewards.append(p['discount_rewards'])
            test_selections.append(p['selections'])
            test_selections_mem.append(p['selections_mem'])
            test_outputs.append(p['outputs'])
            test_outputs_mem.append(p['outputs_mem'])
            test_states.append(p['states'])
            test_states_mem.append(p['states_mem'])
            test_current_exes.append(p['current_exes'])
            test_current_exes_mem.append(p['current_exes_mem'])
           



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

        if test_1000:
            test_1000_samples_RL(m, sess, cfg)

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
                batchesY_test = batchesY_test,
            
                )
def test_1000_samples_RL(m, sess, cfg):
            print("########################################################")
            print("########################################################")
            print("@@checking random thousand samples")

            #gen new seed
            test_seed = round(random.random()*100000)
            num_tests = 1000
            print("test_seed", test_seed, "num_tests", num_tests)
            x_sample, y_sample = samples_generator(cfg['train_fn'], (num_tests, cfg['num_features']) , cfg['samples_value_rng'], test_seed)
            match_count = 0
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_train_mem  = np.zeros((cfg['batch_size'], cfg['state_size']))
            for i in range(num_tests):
                batchX = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                batchX = np.concatenate(([x_sample[i]], batchX), axis=0)

                batchY = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                batchY = np.concatenate(([y_sample[i]], batchY), axis=0)

                p = m.policy_rollout(sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg, False)
                match = np.allclose(y_sample[i], p['output'][0])
                print("i", i , "match", match)
                print("input", list(x_sample[i]))
                print("expect", list(y_sample[i]))
                print("actual", list(p['output'][0].tolist()))
                if match:
                    match_count = match_count + 1
            print()
            print(match_count, "out of", num_tests,"matched") 
            print("#################################")
            print("Model reached hardmax, breaking ...")
            print("#################################")                                                     

def test_1000_samples_RNN(m, sess, cfg):
            print("########################################################")
            print("########################################################")
            print("@@checking random thousand samples")
            #gen new seed
            test_seed = round(random.random()*100000)
            num_tests = 1000
            print("test_seed", test_seed, "num_tests", num_tests)
            x_sample, y_sample = samples_generator(cfg['train_fn'], (num_tests, cfg['num_features']) , cfg['samples_value_rng'], test_seed)
            match_count = 0
            for i in range(num_tests):
                batchX = np.zeros((cfg['batch_size']-1, cfg['num_features']))
                batchX = np.concatenate(([x_sample[i]], batchX), axis=0)  
                output = sess.run([m.test['output']],
                    feed_dict={
                            m.init_state:np.zeros((cfg['batch_size'], cfg['state_size'])),
                            m.mem.init_state:np.zeros((cfg['batch_size'], cfg['state_size'])),
                            m.batchX_placeholder:batchX
                        })
                match = np.allclose(y_sample[i], output[0][0])
                print("i", i , "match", match)
                print("input", list(x_sample[i]))
                print("expect", list(y_sample[i]))
                print("actual", list(output[0][0].tolist()))
                if match:
                    match_count = match_count + 1
            print()
            print(match_count, "out of", num_tests,"matched")
            
            print("#################################")
            print("Model reached hardmax, breaking ...")
            print("#################################") 