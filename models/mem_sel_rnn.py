import tensorflow as tf
import numpy as np
from nn_base import NNbase

class MemRNN(NNbase):
    
    def __init__(self, cfg, ops):

        #init parent
        super(MemRNN, self).__init__(cfg, ops)
        
        with tf.name_scope("RNN_mem"):
            #placeholder for the initial state of the model
            self.init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state_mem")

            #create a ROM cell for the raw inputs and applied op on them
            self.current_input = self.batchX_placeholder
            
            with tf.name_scope("Init_cell_vals"):
                self.mem_cell = [op(self.current_input, self.dummy_matrix) for op in ops.ops]

            #set random seed
            tf.set_random_seed(cfg['seed'])
            with tf.name_scope("Params"):
                #model parameters
                self.params["W_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W_mem")
                self.params["b_mem"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b_mem")

                self.params["W2_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2_mem")
                self.params["b2_mem"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2_mem")

    #forward pass
    def run_forward_pass(self,current_input, cfg, mode="train"):
        with tf.name_scope("Forward_pass_"+mode):
            current_state = self.init_state
            with tf.name_scope("Comp_softmax"):
                input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state_mem")  # Increasing number of columns
                next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, self.params["W_mem"], name="input-state_mult_W"), self.params["b_mem"], name="add_bias_mem"), name="tanh_next_state_mem")  # Broadcasted addition
                current_state = next_state

                #calculate softmax and produce the mask of operations
                logits = tf.add(tf.matmul(next_state, self.params["W2_mem"], name="state_mul_W2_mem"), self.params["b2_mem"], name="add_bias2_mem") #Broadcasted addition
                softmax = tf.nn.softmax(logits, name="get_softmax_mem")

                #in test change to hardmax
                if mode is "test":
                    argmax  = tf.argmax(softmax, 1, )
                    softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])
                #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
            with tf.name_scope("Comp_mem"):
                 output = self.select_mem(self.mem_cell, softmax, cfg)

        return output, current_state, softmax