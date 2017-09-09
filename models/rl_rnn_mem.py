import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RLRNNMEM(NNbase):
    
    def __init__(self, cfg, ops_env):

        #init parent
        super(RLRNNMEM, self).__init__(cfg, ops_env)
        
        #placeholder for the initial state of the model
        with tf.name_scope("RNN_mem"):
            self.init_state = tf.placeholder(cfg['datatype'], [None, cfg['state_size']], name="init_state")
            self.batchX_placeholder = tf.placeholder(cfg['datatype'], [None, cfg['num_features']], name="batchXMem")
            self.batchY_placeholder = tf.placeholder(cfg['datatype'], [None, ops_env.num_of_ops_mem], name="batchYMem")
            self.selections_placeholder = tf.placeholder(cfg['datatype'], name="selections_placeholderMem")
            self.rewards_placeholder = tf.placeholder(cfg['datatype'], name="rewards_placeholderMem")
                        
            #set ops_env
            self.ops_env = ops_env
            self.num_of_ops = ops_env.num_of_ops_mem
            #set random seed
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):

                self.params["W_mem"] = tf.get_variable("W_mem", shape=[ cfg['state_size']+cfg['num_features'], cfg['state_size'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b_mem"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b_mem")

                self.params["W2_mem"] = tf.get_variable("W2_mem", shape=[ cfg['state_size'], ops_env.num_of_ops_mem ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b2_mem"] = tf.Variable(np.zeros((ops_env.num_of_ops_mem)), dtype=cfg['datatype'], name="b2_mem")
                
                self.params["W3_mem"] = tf.get_variable("W3_mem", shape=[ cfg['num_features'], cfg['num_features'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b3_mem"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3_mem")

                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = "train")
            self.selected = self.perform_selection_RL(self.train['log_probs'], cfg)
            self.total_loss_train = self.calc_RL_loss(self.train['log_probs'], cfg)
            
            #calc grads and hereby the backprop step
            self.grads, self.train_step, self.norms = self.calc_backprop_RL(self.total_loss_train, cfg)
            
            '''
            with tf.name_scope("Summaries_grads"):
                for grad, var in self.grads: self.variable_summaries(grad, name=var.name.replace(":","_")+"_grad")
            '''
                
    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state
        
        current_x = self.batchX_placeholder                             
                        
        #define policy network
        with tf.name_scope("Forward_pass_"+mode):
           
                    
                    with tf.name_scope("Comp_next_x"):
                        next_x = tf.add(tf.matmul(current_x, self.params["W3_mem"], name="state_mul_W3"), self.params["b3_mem"], name="add_bias3")
                        current_x = next_x
                    
                    with tf.name_scope("Comp_softmax"):
                        input_and_state_concatenated = tf.concat([current_x, current_state], 1, name="concat_input_state")  # Increasing number of columns
                        _mul1 = tf.matmul(input_and_state_concatenated, self.params["W_mem"], name="input-state_mult_W")
                        _add1 = tf.add(_mul1, self.params["b_mem"], name="add_bias")
                        #_add1 =_mul1
                        if   cfg["state_fn"] == "tanh":
                            next_state = tf.tanh(_add1, name="tanh_next_state")
                        elif cfg["state_fn"] == "relu":
                            next_state = tf.nn.softplus(_add1, name="relu_next_state")
                            #next_state = tf.nn.relu(_add1) - 0.1*tf.nn.relu(-_add1)
                        current_state = next_state
                        
                        #apply dropout
                        '''
                        self.dropout_cntr =  1 + self.dropout_cntr
                        droupout_seed = cfg['seed'] + self.dropout_cntr
                        state_dropped = tf.layers.dropout(next_state, cfg['drop_rate'], seed=droupout_seed, training = (mode is 'train'))
                        '''
                        state_dropped = tf.layers.dropout(next_state, cfg['drop_rate'], training = self.training)
                        
                        #calculate softmax and produce the mask of operations
                        #logits = tf.matmul(state_dropped, self.params["W2"], name="state_mul_W2")
                        logits = tf.add(tf.matmul(state_dropped, self.params["W2_mem"], name="state_mul_W2"), self.params["b2_mem"], name="add_bias2") #Broadcasted addition
                        softmax = tf.nn.softmax(logits, name="get_softmax")
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
    
