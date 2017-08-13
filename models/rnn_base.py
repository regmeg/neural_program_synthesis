import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RNN(NNbase):
    
    def __init__(self, cfg, ops):

        #init parent
        super(RNN, self).__init__(cfg, ops)
        
        #placeholder for the initial state of the model
        self.init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state")

        #set random seed
        tf.set_random_seed(cfg['seed'])
    
        #model parameters
        self.params["W"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
        self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

        self.params["W2"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
        self.params["b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")

        #write model param summaries
        for param, tensor in self.params.items(): self.variable_summaries(tensor)
                                                   
        #create graphs for forward pass to soft and hard selection
        self.output_train, self.current_state_train, self.softmax_train, self.outputs_train, self.softmaxes_train =                     self.run_forward_pass(cfg, mode = "train")
        
        self.total_loss_train, self.math_error_train = self.calc_loss(cfg, self.output_train)

        self.output_test, self.current_state_test, self.softmax_test, self.outputs_test, self.softmaxes_test =                           self.run_forward_pass(cfg, mode = "test")

        self.total_loss_test, self.math_error_test = self.calc_loss(cfg, self.output_test)
    
        #calc grads and hereby the backprop step
        self.grads, self.train_step  = self.calc_backprop(cfg)

    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state

        output = self.batchX_placeholder

        outputs = []

        softmaxes = []

        #printtf = tf.Print(output, [output], message="Strated cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")

        for timestep in range(cfg['max_output_ops']):
            print("timestep " + str(timestep))
            current_input = output

            input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state")  # Increasing number of columns
            next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, self.params["W"], name="input-state_mult_W"), self.params["b"], name="add_bias"), name="tanh_next_state")  # Broadcasted addition
            #next_state = tf.nn.relu(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="relu_next-state")  # Broadcasted addition
            current_state = next_state

            #calculate softmax and produce the mask of operations
            logits = tf.add(tf.matmul(next_state, self.params["W2"], name="state_mul_W2"), self.params["b2"], name="add_bias2") #Broadcasted addition
            softmax = tf.nn.softmax(logits, name="get_softmax")

            #in test change to hardmax
            if mode is "test":
                argmax  = tf.argmax(softmax, 1, )
                softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])
            #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)

            output = self.select_op(current_input, softmax, cfg)

            #save the sequance of softmaxes and outputs
            outputs.append(output)
            softmaxes.append(softmax)
        #printtf = tf.Print(output, [output], message="Finished cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
        return output, current_state, softmax, outputs, softmaxes