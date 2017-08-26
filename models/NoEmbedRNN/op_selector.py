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
                self.params["b_op"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b_op")
                
                self.params["W2_op"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2_op")


    #forward pass
    def select(self, batchX, state, current_input, output_mem ,cfg, mode="train"):
        with tf.name_scope("Op_Sel_"+mode):
                    with tf.name_scope("Comp_softmax"):
                        input_and_state_concatenated = tf.concat([batchX, state], 1, name="concat_input_state")  # Increasing number of columns
                        _mul1 = tf.matmul(input_and_state_concatenated, self.params["W_op"], name="input-state_mult_W")
                        _add1 = tf.add(_mul1, self.params["b_op"], name="add_bias")
                        if   cfg["state_fn"] == "tanh":
                            next_state = tf.tanh(_add1, name="tanh_next_state")
                        elif cfg["state_fn"] == "relu":
                            next_state = tf.nn.relu(_add1, name="relu_next_state")
                        
                        #apply dropout
                        state_dropped = tf.layers.dropout(next_state, cfg['drop_rate'], training = (mode is 'train'))
                        
                        #calculate softmax and produce the mask of operations
                        logits = tf.matmul(state_dropped, self.params["W2_op"], name="state_mul_W2") #Broadcasted addition
                        logits_scaled = tf.multiply(logits, self.softmax_sat, name="sat_softmax")
                        softmax = tf.nn.softmax(logits_scaled, name="get_softmax")

                        #in test change to hardmax
                        if mode is "test":
                            argmax  = tf.argmax(softmax, 1, )
                            softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])

                    with tf.name_scope("Comp_output"):
                        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
                        output = self.select_op(current_input, output_mem, softmax, cfg)
        
            #build the response
        return output, softmax