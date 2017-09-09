from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from nn_base import NNbase

class MemRNN(NNbase):
    
    def __init__(self, cfg, ops):

        #init parent
        super(MemRNN, self).__init__(cfg, ops)
        
        with tf.name_scope(u"RNN_mem"):
            #placeholder for the initial state of the model
            self.init_state = tf.placeholder(cfg[u'datatype'], [None, cfg[u'state_size']], name=u"init_state_mem")

            #create a ROM cell for the raw inputs and applied op on them
            self.current_input = self.batchX_placeholder
            
            with tf.name_scope(u"Init_cell_vals"):
                self.mem_cell = [op(self.current_input, self.dummy_matrix) for op in ops.ops_mem]

            #set random seed
            tf.set_random_seed(cfg[u'seed'])
            with tf.name_scope(u"Params"):
                #model parameters
                self.params[u"W_mem"] = tf.get_variable(u"W_mem", shape=[ cfg[u'state_size']+cfg[u'num_features'], cfg[u'state_size'] ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params[u"b_mem"] = tf.Variable(np.zeros((cfg[u'state_size'])), dtype=cfg[u'datatype'], name=u"b_mem")

                self.params[u"W2_mem"] = tf.get_variable(u"W2_mem", shape=[ cfg[u'state_size'], ops.num_of_ops_mem ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params[u"b2_mem"] = tf.Variable(np.zeros((ops.num_of_ops_mem)), dtype=cfg[u'datatype'], name=u"b2_mem")
                
                u"""
                self.params["W3_mem"] = tf.get_variable("W3_mem", shape=[ ops.num_of_ops_mem, cfg['num_features'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b3_mem"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3_mem")
                """
                u'''
                self.params["W_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W_mem")
                self.params["b_mem"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b_mem")

                self.params["W2_mem"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops_mem], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2_mem")
                self.params["b2_mem"] = tf.Variable(np.zeros((ops.num_of_ops_mem)), dtype=cfg['datatype'], name="b2_mem")

                self.params["W3_mem"] = tf.Variable(tf.truncated_normal([self.ops.num_of_ops_mem, cfg['num_features']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W3_mem")
                self.params["b3_mem"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3_mem")
                '''
    #forward pass
    def run_forward_pass(self,curent_x, current_state, step, cfg, mode=u"train"):
        with tf.name_scope(u"Forward_pass_"+mode):
            with tf.name_scope(u"Comp_softmax"):
                input_and_state_concatenated = tf.concat([curent_x, current_state], 1, name=u"concat_input_state_mem")  # Increasing number of columns
                _mul1 = tf.matmul(input_and_state_concatenated, self.params[u"W_mem"], name=u"input-state_mult_W")
                _add1 = tf.add(_mul1, self.params[u"b_mem"], name=u"add_bias")
                #_add1 =_mul1
                if   cfg[u"state_fn"] == u"tanh":
                    next_state = tf.tanh(_add1, name=u"tanh_next_state")
                elif cfg[u"state_fn"] == u"relu":
                    next_state = tf.nn.softplus(_add1, name=u"relu_next_state")
                    #next_state = tf.nn.relu(_add1) - 0.1*tf.nn.relu(-_add1)

                current_state = next_state
                
                #apply dropout
                u'''
                self.dropout_cntr =  1 + self.dropout_cntr
                droupout_seed = cfg['seed'] + self.dropout_cntr
                state_dropped = tf.layers.dropout(next_state, cfg['drop_rate'], seed=droupout_seed, training = (mode is 'train'))
                '''
                state_dropped = tf.layers.dropout(next_state, cfg[u'drop_rate'], training = (mode is u'train'))
                
                #calculate softmax and produce the mask of operations
                logits = tf.add(tf.matmul(state_dropped, self.params[u"W2_mem"], name=u"state_mul_W2_mem"), self.params[u"b2_mem"], name=u"add_bias2_mem") #Broadcasted addition
                #logits = tf.matmul(state_dropped, self.params["W2_mem"], name="state_mul_W2_mem")
                logits_scaled = tf.multiply(logits, self.softmax_sat, name=u"sat_softmax")
                softmax = tf.nn.softmax(logits_scaled, name=u"get_softmax")
                #softmax = self.custom_softmax(logits_scaled, cfg)

                #in test change to hardmax
                if mode is u"test":
                    argmax  = tf.argmax(softmax, 1, )
                    softmax  = tf.one_hot(argmax, self.ops.num_of_ops_mem, dtype=cfg[u'datatype'])
                #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
            with tf.name_scope(u"Comp_mem"):
                 output = self.select_mem(self.mem_cell, softmax, cfg)
            
            u"""
            with tf.name_scope("Comp_next_x"):
                next_x = tf.add(tf.matmul(logits, self.params["W3_mem"], name="state_mul_W3_mem"), self.params["b3_mem"], name="add_bias3_mem")
                current_x = next_x
            """ 
        #return output, current_state, softmax, current_x
        return output, current_state, softmax