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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var.name.replace(":","_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

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


cfg = get_cfg()    
ops = Operations(cfg)
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

#model
m = RNN(cfg,variable_summaries)
#pre training setting
np.set_printoptions(precision=3, suppress=True)
#train_fn = np_mult
#train_fn = np_stall
x,y = samples_generator(cfg['train_fn'], (cfg['num_samples'], cfg['num_features']) , cfg['samples_value_rng'], cfg['seed'])
x_train, x_test, y_train, y_test = split_train_test (x, y , cfg['test_ratio'])
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
    _W = sess.run([m.W])
    _W2 = sess.run([m.W2])
    print(W.eval())
    print(W2.eval())
    globalstartTime = time.time()
    for epoch_idx in range(cfg['num_epochs']):
        startTime = time.time()
        loss_list_train_soft = [0,0]
        loss_list_train_hard = [0,0]
        loss_list_test_soft = [0,0]
        loss_list_test_hard = [0,0]
        summary = None
        
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
                    _total_loss_train, _train_step, _current_state_train, _output_train, _grads, _softmaxes_train, _math_error_train = sess.run([m.total_loss_train, m.train_step, m.current_state_train, m.output_train, m.grads, m.softmaxes_train, m.math_error_train],
                        feed_dict={
                            init_state:_current_state_train,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
                        })
                    loss_list_train_soft.append(_total_loss_train)
                
                else :
                #for testing cylce, do one forward and back prop with 1 batch with training data, plus produce summary and hardmax result
                    summary, _total_loss_train, _train_step, _current_state_train, _output_train, _grads, _softmaxes_train, _math_error_train = sess.run([merged, m.total_loss_train, m.train_step, m.current_state_train, m.output_train, m.grads, m.softmaxes_train, m.math_error_train],
                    feed_dict={
                        init_state:_current_state_train,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                    loss_list_train_soft.append(_total_loss_train)
                
                    _total_loss_test, _current_state_test, _output_test, _softmaxes_test, _math_error_test = sess.run([m.total_loss_test, m.current_state_test, m.output_test, m.softmaxes_test, m.math_error_test],
                        feed_dict={
                            init_state:_current_state_test,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
                        })
                    loss_list_train_hard.append(_total_loss_test)
        ##save loss for the convergance chessing        
        reduced_loss_train_soft = reduce(lambda x, y: x+y, loss_list_train_soft)
        last_train_losses.append(reduced_loss_train_soft)
        ##every 'test_cycle' epochs test the testing set for sotmax/harmax loss
        if epoch_idx % cfg['test_cycle'] == 0 :
            _current_state_train = np.zeros((cfg['batch_size'], cfg['state_size']))
            _current_state_test = np.zeros((cfg['batch_size'], cfg['state_size']))
            for batch_idx in range(num_test_batches):
                    start_idx = cfg['batch_size'] * batch_idx
                    end_idx   = cfg['batch_size'] * batch_idx + cfg['batch_size']

                    batchX = x_test[start_idx:end_idx]
                    batchY = y_test[start_idx:end_idx]

                    _total_loss_train, _current_state_train = sess.run([m.total_loss_train, m.current_state_train],
                        feed_dict={
                            init_state:_current_state_train,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
                        })
                    loss_list_test_soft.append(_total_loss_train)

                    _total_loss_test, _current_state_test = sess.run([m.total_loss_test, m.current_state_test],
                        feed_dict={
                            init_state:_current_state_test,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
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