from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RLRNNMEM(NNbase):
    
    def __init__(self, cfg, ops_env):

        #init parent
        super(RLRNNMEM, self).__init__(cfg, ops_env)
        
        #placeholder for the initial state of the model
        with tf.name_scope(u"RNN_mem"):
            self.init_state = tf.placeholder(cfg[u'datatype'], [None, cfg[u'state_size']], name=u"init_state")
            self.batchX_placeholder = tf.placeholder(cfg[u'datatype'], [None, cfg[u'num_features']], name=u"batchXMem")
            self.batchY_placeholder = tf.placeholder(cfg[u'datatype'], [None, ops_env.num_of_ops_mem], name=u"batchYMem")
            self.selections_placeholder = tf.placeholder(cfg[u'datatype'], name=u"selections_placeholderMem")
            self.rewards_placeholder = tf.placeholder(cfg[u'datatype'], name=u"rewards_placeholderMem")
                        
            #set ops_env
            self.ops_env = ops_env
            self.num_of_ops = ops_env.num_of_ops_mem
            #set random seed
            tf.set_random_seed(cfg[u'seed'])

            #model parameters
            with tf.name_scope(u"Params"):

                self.params[u"W_mem"] = tf.get_variable(u"W_mem", shape=[ cfg[u'state_size']+cfg[u'num_features'], cfg[u'state_size'] ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None,  dtype=cfg['datatype']))
                self.params[u"b_mem"] = tf.Variable(np.zeros((cfg[u'state_size'])), dtype=cfg[u'datatype'], name=u"b_mem")

                self.params[u"W2_mem"] = tf.get_variable(u"W2_mem", shape=[ cfg[u'state_size'], ops_env.num_of_ops_mem ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None,  dtype=cfg['datatype']))
                self.params[u"b2_mem"] = tf.Variable(np.zeros((ops_env.num_of_ops_mem)), dtype=cfg[u'datatype'], name=u"b2_mem")
                
                self.params[u"W3_mem"] = tf.get_variable(u"W3_mem", shape=[ cfg[u'num_features'], cfg[u'num_features'] ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None,  dtype=cfg['datatype']))
                self.params[u"b3_mem"] = tf.Variable(np.zeros((cfg[u'num_features'])), dtype=cfg[u'datatype'], name=u"b3_mem")

                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = u"train")
            self.selected = self.perform_selection_RL(self.train[u'log_probs'], cfg)
            self.total_loss_train = self.calc_RL_loss(self.train[u'log_probs'], cfg)
            
            #calc grads and hereby the backprop step
            self.grads, self.train_step, self.norms = self.calc_backprop_RL(self.total_loss_train, cfg)
            
            u'''
            with tf.name_scope("Summaries_grads"):
                for grad, var in self.grads: self.variable_summaries(grad, name=var.name.replace(":","_")+"_grad")
            '''
                
    #forward pass
    def run_forward_pass(self, cfg, mode=u"train"):
        current_state = self.init_state
        
        current_x = self.batchX_placeholder                             
                        
        #define policy network
        with tf.name_scope(u"Forward_pass_"+mode):
           
                    
                    with tf.name_scope(u"Comp_next_x"):
                        next_x = tf.add(tf.matmul(current_x, self.params[u"W3_mem"], name=u"state_mul_W3"), self.params[u"b3_mem"], name=u"add_bias3")
                        current_x = next_x
                    
                    with tf.name_scope(u"Comp_softmax"):
                        input_and_state_concatenated = tf.concat([current_x, current_state], 1, name=u"concat_input_state")  # Increasing number of columns
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
                        state_dropped = tf.layers.dropout(next_state, cfg[u'drop_rate'], training = self.training)
                        
                        #calculate softmax and produce the mask of operations
                        #logits = tf.matmul(state_dropped, self.params["W2"], name="state_mul_W2")
                        logits = tf.add(tf.matmul(state_dropped, self.params[u"W2_mem"], name=u"state_mul_W2"), self.params[u"b2_mem"], name=u"add_bias2") #Broadcasted addition
                        softmax = tf.nn.softmax(logits, name=u"get_softmax")
                        # log probabilities - might be untsable, use softmax instead
                        log_probs = tf.log(softmax + 1e-10)
                        #log_probs = softmax


            #build the response dict
        return dict(
                    #outputs main, op seleciton RNN
                    current_state = current_state,
                    log_probs = log_probs,
                    logits = logits,
                    current_x = current_x,
                   )
    
