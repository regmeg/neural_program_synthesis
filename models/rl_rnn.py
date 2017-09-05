import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RLRNN(NNbase):
    
    def __init__(self, cfg, ops_env, mem):

        #init parent
        super(RLRNN, self).__init__(cfg, ops_env)
        
        #placeholder for the initial state of the model
        with tf.name_scope("RNN_op"):
            self.init_state = tf.placeholder(cfg['datatype'], [None, cfg['state_size']], name="init_state")
            self.batchY_placeholder = tf.placeholder(cfg['datatype'], [None, ops_env.num_of_ops], name="batchY")
            self.selections_placeholder = tf.placeholder(cfg['datatype'], name="selections_placeholder")
            self.rewards_placeholder = tf.placeholder(cfg['datatype'], name="rewards_placeholder")
            
            #set ops_env
            self.ops_env = ops_env
            self.num_of_ops = ops_env.num_of_ops
            #set the mem
            self.mem = mem
            #set random seed
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):

                self.params["W"] = tf.get_variable("W", shape=[ cfg['state_size']+cfg['num_features'], cfg['state_size'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.get_variable("W2", shape=[ cfg['state_size'], ops_env.num_of_ops ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b2"] = tf.Variable(np.zeros((ops_env.num_of_ops)), dtype=cfg['datatype'], name="b2")
                
                self.params["W3"] = tf.get_variable("W3", shape=[ cfg['num_features'], cfg['num_features'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b3"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3")

                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = "train")
            self.selected = self.perform_selection_RL(self.train['log_probs'], cfg)
            self.total_loss_train = self.calc_RL_loss(self.train['log_probs'], cfg)
            
            #calc grads and hereby the backprop step
            self.grads, self.train_step, self.norms = self.calc_backprop_RL(self.total_loss_train, cfg)
            #self.train_step  = self.calc_backprop_RL(self.total_loss_train + self.mem.total_loss_train, cfg)
            

        #write model param and grad summaries outside of all scopes
        with tf.name_scope("Summaries_params"):
            for param, tensor in self.params.items(): self.variable_summaries(tensor)               
               
        with tf.name_scope("Summaries_grads"):
            for grad, var in self.grads: 
                print("writing grad", var.name.replace(":","_")+"_grad")
                self.variable_summaries(grad, name=var.name.replace(":","_")+"_grad")
        
        if cfg['norm']:
            with tf.name_scope("Summaries_norms"):
                self.variable_summaries(self.norms)
        
    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state
        
        current_x = self.batchX_placeholder                             
                        
        #define policy network
        with tf.name_scope("Forward_pass_"+mode):
                    
                    with tf.name_scope("Comp_next_x"):
                        next_x = tf.add(tf.matmul(current_x, self.params["W3"], name="state_mul_W3"), self.params["b3"], name="add_bias3")
                        current_x = next_x
                
                
                    with tf.name_scope("Comp_softmax"):
                
                        input_and_state_concatenated = tf.concat([current_x, current_state], 1, name="concat_input_state")  # Increasing number of columns
                        _mul1 = tf.matmul(input_and_state_concatenated, self.params["W"], name="input-state_mult_W")
                        _add1 = tf.add(_mul1, self.params["b"], name="add_bias")
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
                        state_dropped = tf.layers.dropout(next_state, cfg['drop_rate'], training = (mode is 'train'))
                        
                        #calculate softmax and produce the mask of operations
                        #logits = tf.matmul(state_dropped, self.params["W2"], name="state_mul_W2")
                        logits = tf.add(tf.matmul(state_dropped, self.params["W2"], name="state_mul_W2"), self.params["b2"], name="add_bias2") #Broadcasted addition
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
    
        #perform policy rollout - select up to five ops max
    def policy_rollout(self, sess, _current_state_train, _current_state_train_mem, batchX, batchY, cfg):
        
        _current_x = batchX
        _current_x_mem = batchX
        output = batchX
        
        #produce two types of arrays - one for op another for mem selection
        rewards = []
        selections = []
        selections_mem = []
        labels = []
        labels_mem = []
        mem_masks = []
        log_probs = []
        log_probs_mem = []
        states = []
        states_mem = []
        current_exes = []
        current_exes_mem = []
        outputs = []
        outputs_mem = []
 
        for timestep in range(cfg['max_output_ops']):
            #print("timestep", timestep)
            
                        
            #track states produced by the policy RNN
            states.append(_current_state_train)
            states_mem.append(_current_state_train_mem)
            
            #track actual inputs/outputs from the RNN net
            current_exes.append(_current_x)
            current_exes_mem.append(_current_x_mem)
          
            _current_state_train,\
            _current_state_train_mem,\
            _current_x,\
            _current_x_mem,\
            _labels,\
            _labels_mem,\
            _selection,\
            _selection_mem,\
            _log_probs,\
            _log_probs_mem = sess.run([  self.train["current_state"],
                                         self.mem.train["current_state"],
                                         self.train["current_x"],
                                         self.mem.train["current_x"],
                                         self.selected["labels"],
                                         self.mem.selected["labels"],
                                         self.selected["selection"],
                                         self.mem.selected["selection"],
                                         self.train["log_probs"],
                                         self.mem.train["log_probs"]],
                            feed_dict={
                                self.init_state:_current_state_train,
                                self.mem.init_state:_current_state_train_mem,
                                self.batchX_placeholder: _current_x,
                                #self.mem.batchX_placeholder: _current_x_mem
                                #feed the same exes to the mem network
                                self.mem.batchX_placeholder: _current_x
                            })

            #print(np.hstack([_selection, _logits, _log_probs]))
            #def apply_op(self, selections, selections_mem, prev_sel, prev_sel_mem, inptX, batchX, batchY):
            output, output_mem, error, math_error, mem_mask = self.ops_env.apply_op(_selection, _selection_mem, selections[-1:], selections_mem[-1:], output, batchX, batchY)
            reward = cfg['max_reward'] - error
            
            #track rewards
            rewards.append(reward)
            
            #trakc op selection indeces
            selections.append(_selection)
            selections_mem.append(_selection_mem)
            
            #trakc op selection labels
            labels.append(_labels)
            labels_mem.append(_labels_mem)
            
            #track mem mask
            mem_masks.append(mem_mask)
            
            #track probs
            log_probs.append(_log_probs)
            log_probs_mem.append(_log_probs_mem)

            #track outputs from the ops
            outputs.append(output)
            outputs_mem.append(output_mem)

            
            if math_error.sum() < 1:
                #print("erro sum", math_error.sum())
                #print("breaking")
                break

        #print("finished_loops")
        #print("rewards before discounting")
        #print(rewards_ord)

        reeval_rewards = np.apply_along_axis( self.reeval_rewards, 1, np.hstack(rewards)).reshape((cfg['batch_size'],-1))
        discount_rewards = np.apply_along_axis( self.discount_rewards, 1, reeval_rewards).reshape((cfg['batch_size'],-1))
        #return np.float64(discount_rewards), np.float64(selections), np.float64(states), np.float64(current_exes)
        return  dict(
                    discount_rewards = discount_rewards,
                    rewards = reeval_rewards,
                    math_error = math_error,
                    selections = selections,
                    selections_mem = selections_mem,
                    labels = labels,
                    labels_mem = labels_mem,
                    mem_masks = mem_masks,
                    log_probs = log_probs,
                    log_probs_mem = log_probs_mem,
                    states = states,
                    states_mem = states_mem,
                    current_exes = current_exes,
                    current_exes_mem = current_exes_mem,
                    outputs = outputs,
                    outputs_mem = outputs_mem
                )