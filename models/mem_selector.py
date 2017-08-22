import tensorflow as tf
import numpy as np
from nn_base import NNbase

class MemSel(NNbase):
    
    def __init__(self, cfg, ops):

        #init parent
        super(MemSel, self).__init__(cfg, ops)
        
        with tf.name_scope("MemSel"):

            #create a ROM cell for the raw inputs and applied op on them
            self.current_input = self.batchX_placeholder
            
            with tf.name_scope("Init_cell_vals"):
                self.mem_cell = [op(self.current_input, self.dummy_matrix) for op in ops.ops]

            #set random seed
            tf.set_random_seed(cfg['seed'])
            with tf.name_scope("Params"):
                #model parameters
                self.params["W_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W_mem")
                self.params["W2_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2_mem")


    #forward pass
    def select(self, batchX, state, cfg, mode="train"):
        with tf.name_scope("Mem_sel"+mode):
            with tf.name_scope("Comp_softmax"):
                input_and_state_concatenated = tf.concat([batchX, state], 1, name="concat_input_state_mem")  # Increasing number of columns
                next_state = tf.tanh(tf.matmul(input_and_state_concatenated, self.params["W_mem"], name="input-state_mult_W"), name="tanh_next_state_mem")  # Broadcasted addition

                #calculate softmax and produce the mask of operations
                logits = tf.matmul(next_state, self.params["W2_mem"], name="state_mul_W2_mem")
                softmax = tf.nn.softmax(logits, name="get_softmax_mem")

                #in test change to hardmax
                if mode is "test":
                    argmax  = tf.argmax(softmax, 1, )
                    softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])
                #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
            with tf.name_scope("Comp_mem"):
                 output = self.select_mem(self.mem_cell, softmax, cfg)

        return output, softmax