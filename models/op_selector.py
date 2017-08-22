import tensorflow as tf
import numpy as np
from nn_base import NNbase

class OpSel(NNbase):
    
    def __init__(self, cfg, ops):

        #init parent
        super(OpSel, self).__init__(cfg, ops)
        
        #placeholder for the initial state of the model
        with tf.name_scope("OpSel"):
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):
                self.params["W_op"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W_op")
                self.params["W2_op"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2_op")


    #forward pass
    def select(self, batchX, state, current_input, output_mem ,cfg, mode="train"):
        with tf.name_scope("Op_Sel_"+mode):
                    with tf.name_scope("Comp_softmax"):
                        input_and_state_concatenated = tf.concat([batchX, state], 1, name="concat_input_state")  # Increasing number of columns
                        next_state = tf.tanh(tf.matmul(input_and_state_concatenated, self.params["W_op"], name="input-state_mult_W"), name="tanh_next_state_op")

                        #calculate softmax and produce the mask of operations
                        logits = tf.matmul(next_state, self.params["W2_op"], name="state_mul_W2") #Broadcasted addition
                        softmax = tf.nn.softmax(logits, name="get_softmax")

                        #in test change to hardmax
                        if mode is "test":
                            argmax  = tf.argmax(softmax, 1, )
                            softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])

                    with tf.name_scope("Comp_output"):
                        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
                        output = self.select_op(current_input, output_mem, softmax, cfg)
        
            #build the response
        return output, softmax