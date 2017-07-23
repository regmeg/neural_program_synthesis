import numpy as np
import tensorflow as tf
from functools import reduce
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from numpy.random import RandomState
import random
import time
import threading 
from tensorflow.python.client import timeline

#model flags
tf.flags.DEFINE_boolean("debug", False, "weather run in a dubg mode")
tf.flags.DEFINE_integer("seed", round(random.random()*100000), "the global simulation seed for np and tf")
tf.flags.DEFINE_string("name", "predef_sim_name" , "name of the simulation")

datatype = tf.float64
FLAGS = tf.flags.FLAGS


#set random seed
tf.set_random_seed(FLAGS.seed)

#configuraion constants
total_num_epochs = 10000000
iters_per_epoch = 1
num_epochs = total_num_epochs // iters_per_epoch
state_size = 50
num_of_operations = 3
max_output_ops = 5
num_features = 3
num_samples = 1500
batch_size  = 100
param_init = 0.1
learning_rate = 0.005

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var.name.replace(":","_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def split_train_test (x, y , test_ratio):
    
    if y.shape != x.shape:
        raise Exception('Model expects x and y shapes to be the same')
    
    test_len  = int(x.shape[0]*test_ratio)
    train_len = x.shape[0] - test_len

    x_train = x[0:train_len][:]
    x_test  = x[-test_len:][:]
    y_train = y[0:train_len][:]
    y_test  = y[-test_len:][:]
    
    print(train_len)
    print(test_len)

    train_shape = (train_len, x.shape[1])
    test_shape = (test_len, x.shape[1])
    
    if y_train.shape != train_shape or x_train.shape != train_shape or x_test.shape != test_shape or y_test.shape != test_shape:
        raise Exception('One of the conversion test/train shapes gone wrong')
    
    return  x_train, x_test, y_train, y_test

#helpder func
def get_time_hhmmss(dif):
    m, s = divmod(dif, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


#sample gen functions
def np_add(vec):
    return reduce((lambda x, y: x + y),vec)

def np_mult(vec):
    return reduce((lambda x, y: x * y),vec)

def np_stall(vec):
    return vec

def samples_generator(fn, shape, rng, seed):
    '''
    Generate random samples for the model:
    @fn - function to be applied on the input features to get the ouput
    @shape - shape of the features matrix (num_samples, num_features)
    @rng - range of the input features to be generated within (a,b)
    Outputs a tuple of input and output features matrix
    '''
    prng = RandomState(seed)
    x = (rng[1] - rng[0]) * prng.random_sample(shape) + rng[0]
    y = np.apply_along_axis(fn, 1, x).reshape((shape[0],-1))
    z = np.zeros((shape[0],shape[1] - y.shape[1]))
    y = np.concatenate((y, z), axis=1)
    
    return x,y
    

#model operations
def tf_multiply(inpt):
    return tf.reshape( tf.reduce_prod(inpt, axis = 1, name = "tf_mult"), [batch_size, -1], name = "tf_mult_reshape")

def tf_add(inpt):
    return  tf.reshape( tf.reduce_sum(inpt, axis = 1, name = "tf_add"), [batch_size, -1], name = "tf_add_reshape")

def tf_stall(a):
    return a

#model constants
dummy_matrix = tf.zeros([batch_size, num_features], dtype=datatype, name="dummy_constant")

#model placeholders
batchX_placeholder = tf.placeholder(datatype, [batch_size, None], name="batchX")
batchY_placeholder = tf.placeholder(datatype, [batch_size, None], name="batchY")

init_state = tf.placeholder(datatype, [batch_size, state_size], name="init_state")

#model parameters
W = tf.Variable(tf.truncated_normal([state_size+num_features, state_size], -1*param_init, param_init, dtype=datatype), dtype=datatype, name="W")
b = tf.Variable(np.zeros((state_size)), dtype=datatype, name="b")
variable_summaries(W)
variable_summaries(b)

W2 = tf.Variable(tf.truncated_normal([state_size, num_of_operations], -1*param_init, param_init, dtype=datatype),dtype=datatype, name="W2")
b2 = tf.Variable(np.zeros((num_of_operations)), dtype=datatype, name="b2")
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
    
    for timestep in range(max_output_ops):
        print("timestep " + str(timestep))
        current_input = output



        input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state")  # Increasing number of columns
        next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="tanh_next_state")  # Broadcasted addition
        #next_state = tf.nn.relu(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="relu_next-state")  # Broadcasted addition
        current_state = next_state

        #calculate softmax and produce the mask of operations
        logits = tf.add(tf.matmul(next_state, W2, name="state_mul_W2"), b2, name="add_bias2") #Broadcasted addition
        softmax = tf.nn.softmax(logits, name="get_softmax")
        #argmax = tf.argmax(softmax, 1)
        '''
        print(logits)
        print(softmax)
        print(argmax)
        '''
        #perform ops
        add   = tf_add(current_input)
        mult  = tf_multiply(current_input)
        stall = tf_stall(current_input)
        #add = tf.reshape( tf.reduce_prod(current_input, axis = 1), [batch_size, -1])
        #mult = tf.reshape( tf.reduce_sum(current_input, axis = 1), [batch_size, -1])
        #stall = current_input
        #values = tf.concat([add, mult, stall], 1)
        #values = tf.concat([add, mult, stall], 1, name="concact_op_values")
        #values = tf.cast(values,dtype=datatype)
        #get softmaxes for operations
        #add_softmax = tf.slice(softmax, [0,0], [batch_size,1])
        #mult_softmax = tf.slice(softmax, [0,1], [batch_size,1])
        #stall_softmax = tf.slice(softmax, [0,2], [batch_size,1])
        #produce output matrix
        #onehot  = tf.one_hot(argmax_dum, num_of_operations)
        #stall_width = tf.shape(stall)[1]
        #stall_select = tf.slice(onehot, [0,2], [batch_size,1])
        #mask_arr = [onehot]
        #for i in range(num_features-1):
        #    mask_arr.append(stall_select)
        #mask = tf.concat(mask_arr, 1)
        #argmax = tf.reshape( softmax, [batch_size, -1])
        #mask = onehot
        #mask = tf.cast(mask, dtype=datatype)
        #mask = tf.cast(mask, tf.bool)
        #apply mask
        #output = tf.boolean_mask(values,mask)
        #in test change to hardmax
        if mode is "test":
            argmax  = tf.argmax(softmax, 1, )
            softmax  = tf.one_hot(argmax, num_of_operations, dtype=datatype)
        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
        add_softmax   = tf.slice(softmax, [0,0], [batch_size,1], name="slice_add_softmax_val")
        mult_softmax  = tf.slice(softmax, [0,1], [batch_size,1], name="slice_mult_softmax_val")
        stall_softmax = tf.slice(softmax, [0,2], [batch_size,1], name="stall_mult_softmax_val")

        add_width   = tf.shape(add, name="add_op_shape")[1]
        mult_width  = tf.shape(mult, name="mult_op_shape")[1]
        stall_width = tf.shape(stall, name="stall_op_shape")[1]


        add_final   = tf.multiply(add, add_softmax, name="mult_add_softmax")
        mult_final  = tf.multiply(mult,mult_softmax, name="mult_mult_softmax")
        stall_final = tf.multiply(stall, stall_softmax, name="mult_stall_softmax")

        ##conact add and mult results with zeros matrix
        add_final = tf.concat([add_final, tf.slice(dummy_matrix, [0,0], [batch_size, num_features - add_width], name="slice_dum_add")], 1, name="concat_add_op_dummy_zeros") 
        mult_final = tf.concat([mult_final, tf.slice(dummy_matrix, [0,0], [batch_size, num_features - mult_width], name="slice_dum_mult")], 1, name="concat_mult_op_dummy_zeros") 


        output = tf.add(add_final, mult_final, name="add_final_op_mult_add")
        output =  tf.add(output, stall_final, name="add_final_op_stall")
        outputs.append(output)
        softmaxes.append(softmax)
    #printtf = tf.Print(output, [output], message="Finished cycle")
    #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
    return output, current_state, softmax, outputs, softmaxes

#cost function
def calc_loss(output):
    #reduced_output = tf.reshape( tf.reduce_sum(output, axis = 1, name="red_output"), [batch_size, -1], name="resh_red_output")
    math_error = tf.multiply(tf.constant(0.5, dtype=datatype), tf.square(tf.subtract(output , batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")
    
    total_loss = tf.reduce_sum(math_error, name="red_total_loss")
    return total_loss, math_error

output_train, current_state_train, softmax_train, outputs_train, softmaxes_train = run_forward_pass(mode = "train")
total_loss_train, math_error_train = calc_loss(output_train)

output_test, current_state_test, softmax_test, outputs_test, softmaxes_test = run_forward_pass(mode = "test")
total_loss_test, math_error_test = calc_loss(output_test)

grads_raw = tf.gradients(output_train, [W,b,W2,b2], name="comp_gradients")

#clip gradients by value
grads, norms = tf.clip_by_global_norm(grads_raw, 10e2)


#ad summaries
for grad in grads: variable_summaries(grad)
variable_summaries(norms)

train_step = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6 ,name="AdamOpt").apply_gradients(zip(grads, [W,b,W2,b2]), name="min_loss")
print("grads are")
print(grads)

#pre training setting
np.set_printoptions(precision=3, suppress=True)
train_fn = np_add
#train_fn = np_mult
#train_fn = np_stall
x,y = samples_generator(train_fn, (num_samples, num_features) , (-100, 100), FLAGS.seed)
x_train, x_test, y_train, y_test = split_train_test (x, y , 0.33333333333)
num_batches = x_train.shape[0]//batch_size
num_test_batches = x_test.shape[0]//batch_size
#model training

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)


#Enable jit
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(config=config) as sess:
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./summaries/' + FLAGS.name ,sess.graph)
    ##enable debugger if necessary
    if (FLAGS.debug):
        print("Running in a debug mode")
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())
    #plt.ion()
    #plt.figure()
    #plt.show() 
    globalstartTime = time.time()
    for epoch_idx in range(num_epochs):
        startTime = time.time()
        loss_list_train_soft = [0,0]
        loss_list_train_hard = [0,0]
        loss_list_test_soft = [0,0]
        loss_list_test_hard = [0,0]
        
        for _iter in range(iters_per_epoch):
            _current_state_train = np.zeros((batch_size, state_size))
            _current_state_test = np.zeros((batch_size, state_size))

            for batch_idx in range(num_batches):
                start_idx = batch_size * batch_idx
                end_idx   = batch_size * batch_idx + batch_size

                batchX = x_train[start_idx:end_idx]
                batchY = y_train[start_idx:end_idx]
                
                summary, _total_loss_train, _train_step, _current_state_train, _output_train, _grads, _softmaxes_train, _math_error_train = sess.run([merged, total_loss_train, train_step, current_state_train, output_train, grads, softmaxes_train, math_error_train],
                    feed_dict={
                        init_state:_current_state_train,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                loss_list_train_soft.append(_total_loss_train)

                summary, _total_loss_test, _current_state_test, _output_test, _softmaxes_test, _math_error_test = sess.run([merged, total_loss_test, current_state_test, output_test, softmaxes_test, math_error_test],
                    feed_dict={
                        init_state:_current_state_test,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                loss_list_train_hard.append(_total_loss_test)
                    
            _current_state_train = np.zeros((batch_size, state_size))
            _current_state_test = np.zeros((batch_size, state_size))
            for batch_idx in range(num_test_batches):
                start_idx = batch_size * batch_idx
                end_idx   = batch_size * batch_idx + batch_size

                batchX = x_test[start_idx:end_idx]
                batchY = y_test[start_idx:end_idx]
                
                summary, _total_loss_train, _current_state_train = sess.run([merged, total_loss_train, current_state_train],
                    feed_dict={
                        init_state:_current_state_train,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                loss_list_test_soft.append(_total_loss_train)

                summary, _total_loss_test, _current_state_test = sess.run([merged, total_loss_test, current_state_test],
                    feed_dict={
                        init_state:_current_state_test,
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
                loss_list_test_hard.append(_total_loss_test)
            train_writer.add_summary(summary, epoch_idx)

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
        print("Softmax train loss\t", reduce(lambda x, y: x+y, loss_list_train_soft))
        print("Hardmax train loss\t", reduce(lambda x, y: x+y, loss_list_train_hard))
        print("Sotfmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_soft))
        print("Hardmax test loss\t", reduce(lambda x, y: x+y, loss_list_test_hard))
        print("Epoch time: ", ((time.time() - startTime) % 60), " Global Time: ",  get_time_hhmmss(time.time() - globalstartTime))
        print("func: ", train_fn.__name__, "max_ops: ", max_output_ops, "sim_seed", FLAGS.seed)
        #print("grads[0] - W", _grads[0][0])
        #print("grads[1] - b", _grads[1][0])
        #print("grads[2] - W2", _grads[2][0])
        #print("grads[3] - b2", _grads[3][0])
        #print("W", W.eval())
        #print("w2" , W2.eval())
        #record execution timeline
