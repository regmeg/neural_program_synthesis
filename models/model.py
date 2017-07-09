import numpy as np
import tensorflow as tf
from functools import reduce
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

#model flags
tf.flags.DEFINE_boolean("debug", False, "weather run in a dubg mode")
datatype = tf.float64
FLAGS = tf.flags.FLAGS

#configuraion constants
num_epochs = 1000000
state_size = 4
num_of_operations = 3
max_output_ops = 1
num_batches = 100
num_features = 3
num_samples = 10
max_num_features = 16
max_num_outputs = 16
batch_size = num_samples
param_init = 0.1
#sample gen functions
def np_add(vec):
    return reduce((lambda x, y: x + y),vec)

def np_mult(vec):
    return reduce((lambda x, y: x * y),vec)

def np_stall(vec):
    return vec

def samples_generator(fn, shape, rng, seed=None):
    '''
    Generate random samples for the model:
    @fn - function to be applied on the input features to get the ouput
    @shape - shape of the features matrix (num_samples, num_features)
    @rng - range of the input features to be generated within (a,b)
    @seed  - generation seed
    Outputs a tuple of input and output features matrix
    '''
    x = (rng[1] - rng[0]) * np.random.random_sample(shape) + rng[0]
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

W2 = tf.Variable(tf.truncated_normal([state_size, num_of_operations], -1*param_init, param_init, dtype=datatype),dtype=datatype, name="W2")
b2 = tf.Variable(np.zeros((num_of_operations)), dtype=datatype, name="b2")

    #forward pass
def run_forward_pass(mode="train"):
    current_state = init_state

    output = batchX_placeholder

    outputs = []

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
    return output, current_state, softmax

#cost function
def calc_loss(output):
    #reduced_output = tf.reshape( tf.reduce_sum(output, axis = 1, name="red_output"), [batch_size, -1], name="resh_red_output")
    math_error = tf.multiply(tf.constant(0.5, dtype=datatype), tf.square(tf.subtract(output , batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")
    
    total_loss = tf.reduce_sum(math_error, name="red_total_loss")
    return total_loss, math_error

output, current_state, softmax = run_forward_pass(mode = "train")
total_loss, math_error = calc_loss(output)

output_test, current_state_test, softmax_test = run_forward_pass(mode = "test")
total_loss_test, math_error_test = calc_loss(output_test)

grads = tf.gradients(total_loss, [W,b,W2,b2], name="comp_gradients")
train_step = tf.train.AdamOptimizer(0.1, epsilon=1e-6 ,name="AdamOpt").apply_gradients(zip(grads, [W,b,W2,b2]), name="min_loss")
print("grads are")
print(grads)

#numpy printpoints
np.set_printoptions(precision=3)

#model training
with tf.Session() as sess:
    
    ##enable debugger if necessary
    if (FLAGS.debug):
        print("Running in a debug mode")
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())
    #plt.ion()
    #plt.figure()
    #plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))
        x,y = samples_generator(np_stall, (num_samples, num_features) , (-100, 100))
        #print("\r New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):

            _total_loss, _train_step, _current_state, _output, _grads, _softmax, _math_error = sess.run(
                    [total_loss, train_step, current_state, output, grads, softmax, math_error],
                    feed_dict={
                        init_state:_current_state,
                        batchX_placeholder:x,
                        batchY_placeholder:y
                    })
            loss_list.append(_total_loss)
            
            
        #harmax test
        _total_loss_test, _softmax_test, _output_test, _math_error_test = sess.run([total_loss_test, softmax_test, output_test, math_error_test], feed_dict={init_state:_current_state, batchX_placeholder:x, batchY_placeholder:y})
        print(" ")

        print("output_train\t\t\t\t\toutput_test")
        print(np.column_stack((_output, _output_test)))
        print("x\t\t\\t\t\t\t\y")
        print(np.column_stack((x, y)))
        print("softmax_train\t\t\t\t\softmax_test")
        print(np.column_stack((_softmax, _softmax_test)))
        print("mat_error_train\t\t\t\t\math_error_test")
        print(np.column_stack((_math_error, _math_error_test)))
        print("Epoch",epoch_idx, " Loss\t", _total_loss , "    ")
        print("Harmax test\t", _total_loss_test)
        #print("grads[0] - W", _grads[0][0])
        #print("grads[1] - b", _grads[1][0])
        #print("grads[2] - W2", _grads[2][0])
        #print("grads[3] - b2", _grads[3][0])
        #print("W", W.eval())
        #print("w2" , W2.eval())