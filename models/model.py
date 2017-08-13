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
from ops import Operations

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
ops = Operations(cfg['batch_size'])
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

#model constants
dummy_matrix = tf.zeros([cfg['batch_size'], cfg['num_features']], dtype=cfg['datatype'], name="dummy_constant")

#model placeholders
batchX_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], None], name="batchX")
batchY_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], None], name="batchY")

init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state")


#set random seed
tf.set_random_seed(cfg['seed'])

#model parameters
W = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
b = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")
variable_summaries(W)
variable_summaries(b)

W2 = tf.Variable(tf.truncated_normal([cfg['state_size'], ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
b2 = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")
variable_summaries(W2)
variable_summaries(b2)

    #forward pass
def run_forward_pass(mode="train"):
    current_state = init_state

    output = batchX_placeholder

    outputs = []

    softmaxes = []
    
    #printtf = tf.Print(output, [output], message="Strated cycle")
    #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
    
    for timestep in range(cfg['max_output_ops']):
        print("timestep " + str(timestep))
        current_input = output



        input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state")  # Increasing number of columns
        next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="tanh_next_state")  # Broadcasted addition
        #next_state = tf.nn.relu(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="relu_next-state")  # Broadcasted addition
        current_state = next_state

        #calculate softmax and produce the mask of operations
        logits = tf.add(tf.matmul(next_state, W2, name="state_mul_W2"), b2, name="add_bias2") #Broadcasted addition
        softmax = tf.nn.softmax(logits, name="get_softmax")
        
        #in test change to hardmax
        if mode is "test":
            argmax  = tf.argmax(softmax, 1, )
            softmax  = tf.one_hot(argmax, ops.num_of_ops, dtype=cfg['datatype'])
        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
        
        #######################
        #perform op selection #
        #######################
        
        #perform all ops in the current timestep intput and save output results together with the op name
        op_res = []
        for op in ops.ops:
            name = op.__name__
            op_outp = op(current_input)
            op_res.append((name, op_outp))
        
        #slice softmax results for each operation
        ops_softmax = []
        for i, op in enumerate(ops.ops):
            name = "slice_"+op.__name__+"_softmax_val"
            softmax_slice = tf.slice(softmax, [0,i], [cfg['batch_size'],1], name=name)
            ops_softmax.append(softmax_slice)

        #calculate the result matrix width produced by an op
        ops_width = []
        for res in op_res:
            name=res[0]+"_op_shape"
            op_widht = tf.shape(res[1], name=name)[1]
            ops_width.append(op_widht)
        
        #apply softmax on each operation so that operation selection is performed
        ops_final = []
        for i,res in enumerate(op_res):
            name = "mult_"+res[0]+"_softmax"
            op_selection =  tf.multiply(res[1], ops_softmax[i], name=name)
            ops_final.append(op_selection)
        
        #slice the missing from the dummy zero matrix and concat it with the op produced output, so that all ops have same output
        ops_matrixes = []
        for i,res in enumerate(op_res):
            slice_name = "slice_"+res[0]
            zeros_slice = tf.slice(dummy_matrix, [0,0], [cfg['batch_size'], cfg['num_features'] - ops_width[i]], name=slice_name)
            concat_name = "concat_"+res[0]+"_op_dummy_zeros"
            op_slize_concat = tf.concat([ops_final[i], zeros_slice], 1, name=concat_name)
            ops_matrixes.append(op_slize_concat)
         
        #add results from all operation with applied softmax together
        output = tf.add_n(ops_matrixes)
        
        #save the sequance of softmaxes and outputs
        outputs.append(output)
        softmaxes.append(softmax)
    #printtf = tf.Print(output, [output], message="Finished cycle")
    #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
    return output, current_state, softmax, outputs, softmaxes

#cost function
def calc_loss(output):
    #reduced_output = tf.reshape( tf.reduce_sum(output, axis = 1, name="red_output"), [batch_size, -1], name="resh_red_output")
    math_error = tf.multiply(tf.constant(0.5, dtype=cfg['datatype']), tf.square(tf.subtract(output , batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")
    
    total_loss = tf.reduce_sum(math_error, name="red_total_loss")
    return total_loss, math_error

output_train, current_state_train, softmax_train, outputs_train, softmaxes_train = run_forward_pass(mode = "train")
total_loss_train, math_error_train = calc_loss(output_train)

output_test, current_state_test, softmax_test, outputs_test, softmaxes_test = run_forward_pass(mode = "test")
total_loss_test, math_error_test = calc_loss(output_test)

grads_raw = tf.gradients(total_loss_train, [W,b,W2,b2], name="comp_gradients")

#clip gradients by value and add summaries
if cfg['norm']:
    print("norming the grads")
    grads, norms = tf.clip_by_global_norm(grads_raw, cfg['grad_norm'])
    variable_summaries(norms)
else:
    grads = grads_raw

for grad in grads: variable_summaries(grad)


train_step = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['epsilon'] ,name="AdamOpt").apply_gradients(zip(grads, [W,b,W2,b2]), name="min_loss")
print("grads are")
print(grads)

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
    _W = sess.run([W])
    _W2 = sess.run([W2])
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
                    _total_loss_train, _train_step, _current_state_train, _output_train, _grads, _softmaxes_train, _math_error_train = sess.run([total_loss_train, train_step, current_state_train, output_train, grads, softmaxes_train, math_error_train],
                        feed_dict={
                            init_state:_current_state_train,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
                        })
                    loss_list_train_soft.append(_total_loss_train)
                
                else :
                #for testing cylce, do one forward and back prop with 1 batch with training data, plus produce summary and hardmax result
                    summary, _total_loss_train, _train_step, _current_state_train, _output_train, _grads, _softmaxes_train, _math_error_train = sess.run([merged, total_loss_train, train_step, current_state_train, output_train, grads, softmaxes_train, math_error_train],
                    feed_dict={
                        init_state:_current_state_train,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                    loss_list_train_soft.append(_total_loss_train)
                
                    _total_loss_test, _current_state_test, _output_test, _softmaxes_test, _math_error_test = sess.run([total_loss_test, current_state_test, output_test, softmaxes_test, math_error_test],
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

                    _total_loss_train, _current_state_train = sess.run([total_loss_train, current_state_train],
                        feed_dict={
                            init_state:_current_state_train,
                            batchX_placeholder:batchX,
                            batchY_placeholder:batchY
                        })
                    loss_list_test_soft.append(_total_loss_train)

                    _total_loss_test, _current_state_test = sess.run([total_loss_test, current_state_test],
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