import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RLRNN(NNbase):
    
    def __init__(self, cfg, ops_env):

        #init parent
        super(RLRNN, self).__init__(cfg, ops_env)
        
        #placeholder for the initial state of the model
        with tf.name_scope("RNN_op"):
            self.init_state = tf.placeholder(cfg['datatype'], [None, cfg['state_size']], name="init_state")
            self.selections_placeholder = tf.placeholder(cfg['datatype'], name="selections_placeholder")
            self.rewards_placeholder = tf.placeholder(cfg['datatype'], name="rewards_placeholder")
            
            #set ops_env
            self.ops_env = ops_env
            #set random seed
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):

                self.params["W"] = tf.get_variable("W", shape=[ cfg['state_size']+cfg['num_features'], cfg['state_size'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.get_variable("W2", shape=[ cfg['state_size'], ops_env.num_of_ops ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b2"] = tf.Variable(np.zeros((ops_env.num_of_ops)), dtype=cfg['datatype'], name="b2")
                
                self.params["W3"] = tf.get_variable("W3", shape=[ ops_env.num_of_ops, cfg['num_features'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b3"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3")

                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = "train")
            self.selection = self.perform_selection(self.train['log_probs'], cfg)
            self.total_loss_train = self.calc_RL_loss(self.train['log_probs'], cfg)
            
            #calc grads and hereby the backprop step
            #self.grads, self.train_step, self.norms  = self.calc_backprop(cfg)
            self.train_step  = self.calc_backprop(self.total_loss_train, cfg)
        

        #write model param and grad summaries outside of all scopes
        with tf.name_scope("Summaries_params"):
            for param, tensor in self.params.items(): self.variable_summaries(tensor)               
        '''       
        with tf.name_scope("Summaries_grads"):
            param_names = [tensor.name.replace(":","_") for param, tensor in self.params.items()]
            for i, grad in enumerate(self.grads): self.variable_summaries(grad, name=param_names[i]+"_grad")
        
        if cfg['norm']:
            with tf.name_scope("Summaries_norms"):
                self.variable_summaries(self.norms)
        '''
    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state
        
        current_x = self.batchX_placeholder                             
                        
        #define policy network
        with tf.name_scope("Forward_pass_"+mode):
           
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
                        #log_probs = tf.log(logits)

                    with tf.name_scope("Comp_next_x"):
                        next_x = tf.add(tf.matmul(logits, self.params["W3"], name="state_mul_W3"), self.params["b3"], name="add_bias3")
                        current_x = next_x

            #build the response dict
        return dict(
                    #outputs main, op seleciton RNN
                    current_state = current_state,
                    log_probs = log_probs,
                    logits = logits,
                    current_x = current_x,
                   )
    
    #perform selection from the distribution
    def perform_selection(self, logits, cfg):
            with tf.name_scope("perform_selection"):
                selection = tf.multinomial(logits, 1, name="draw_from_logits")
                reshape = tf.reshape(selection , [cfg['batch_size'], -1], name = "reshape")
                return reshape
    
    #perform policy rollout - select up to five ops max
    def policy_rollout(self, sess, _current_state_train, batchX, batchY, cfg):
        
        _current_x = batchX
        output = batchX
        
        #produce two types of arrays - ones step based and others - batch based
        rewards = []
        selections = []
        states = []
        current_exes = []
        outputs = []        
 
        for timestep in range(cfg['max_output_ops']):
            #print("timestep", timestep)
            
                        
            #track states produced by the policy RNN
            states.append(_current_state_train)
            
            #track actual inputs/outputs from the RNN net
            current_exes.append(_current_x)
            
            
            
            _current_state_train,\
            _current_x,\
            _logits,\
            _log_probs,\
            _selection  = sess.run([self.train["current_state"],
                                      self.train["current_x"],
                                      self.train["logits"],
                                      self.train["log_probs"],
                                      self.selection],
                            feed_dict={
                                self.init_state:_current_state_train,
                                self.batchX_placeholder: _current_x
                            })
            
            #print(np.hstack([_selection, _logits, _log_probs]))

            output, error, math_error = self.ops_env.apply_op(_selection, output, batchY)
            reward = cfg['max_reward'] - error
            
            #track rewards
            rewards.append(reward)
            
            #trakc op selection indeces
            selections.append(_selection)           

            #track outputs from the ops
            outputs.append(output)

            
            if abs(error.sum()) < 1:
                #print("erro sum", abs(error.sum()))
                #print("breaking")
                break
        
        #print("finished_loops")
        #print("rewards before discounting")
        #print(rewards_ord)
        _discount_rewards = np.apply_along_axis( self.discount_rewards, 1, np.hstack(rewards)).reshape((cfg['batch_size'],-1))
        #return np.float64(discount_rewards), np.float64(selections), np.float64(states), np.float64(current_exes)
        return  _discount_rewards,\
                rewards,\
                selections,\
                states,\
                current_exes,\
                outputs,\
                math_error
    
    #calculate loss function based on selected policies and the achieved rewards
    def calc_RL_loss(self, log_prob, cfg):
        indices = tf.cast(tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1], cfg['datatype']) + self.selections_placeholder
        op_prob = tf.gather(tf.reshape(log_prob, [-1]), tf.cast(indices, tf.int64))

        #comp loss
        loss = -tf.reduce_sum(tf.multiply(op_prob, self.rewards_placeholder))
        
        return loss
    
    #dicount rewards for the first selection and make the later selection more important - which is contrary to https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5#file-pg-pong-py-L130
    def discount_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        #print("discounting")
        #print(rewards)
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in range(0, len(rewards)):
            running_add = running_add * 0.9 + rewards[t] #for all pos/negative rewards mean is always going to be bigger than the first reward, hence it will become positive when centered
            discounted_r[t] = running_add
        #normalise rewards
        #print("discounted_r")
        #print(discounted_r)
        #print("np.mean(discounted_r)")
        #print(np.mean(discounted_r))
        #print("np.std(discounted_r)")
        #print(np.std(discounted_r))
        #print("np.linalg.norm(discounted_r, 1)")
        #print(np.linalg.norm(discounted_r, 1))
        #print("np.linalg.norm(discounted_r, 2)")
        #print(np.linalg.norm(discounted_r, 2))
        #dont scale but norm, as scaling might result into inversion of signs
        #discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-10)
        discounted_r = discounted_r/ np.linalg.norm(discounted_r, 2)
        #print("normalised r")
        #print(discounted_r)
        return discounted_r

    def calc_backprop(self, loss, cfg):
        optimizer = tf.train.RMSPropOptimizer(cfg['learning_rate'])
        train_step = optimizer.minimize(loss)
        return train_step