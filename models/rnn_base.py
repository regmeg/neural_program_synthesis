import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RNN(NNbase):
    
    def __init__(self, cfg, ops, mem):

        #init parent
        super(RNN, self).__init__(cfg, ops)
        
        #placeholder for the initial state of the model
        with tf.name_scope("RNN_op"):
            self.init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state")

            #set mem
            self.mem = mem
            #set random seed
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):
                
                self.params["W"] = tf.get_variable("W", shape=[ cfg['state_size']+cfg['num_features'], cfg['state_size'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.get_variable("W2", shape=[ cfg['state_size'], ops.num_of_ops ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")
                
                self.params["W3"] = tf.get_variable("W3", shape=[ ops.num_of_ops, cfg['num_features'] ], dtype=cfg['datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params["b3"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3")
                
                
                '''
                self.params["W"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
                self.params["b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")
                
                self.params["W3"] = tf.Variable(tf.truncated_normal([self.ops.num_of_ops, cfg['num_features']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W3")
                self.params["b3"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3")
                '''
                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = "train")
            self.total_loss_train, self.math_error_train = self.calc_loss(cfg, self.train["output"], mode = "train")

            self.test = self.run_forward_pass(cfg, mode = "test")
            self.total_loss_test, self.math_error_test = self.calc_loss(cfg, self.test["output"], mode = "test")

            #calc grads and hereby the backprop step
            self.grads, self.train_step, self.norms  = self.calc_backprop(cfg)

        #write model param and grad summaries outside of all scopes
        with tf.name_scope("Summaries_params"):
            for param, tensor in self.params.items(): self.variable_summaries(tensor)               
        
        with tf.name_scope("Summaries_grads"):
            param_names = [tensor.name.replace(":","_") for param, tensor in self.params.items()]
            for i, grad in enumerate(self.grads): self.variable_summaries(grad, name=param_names[i]+"_grad")
        
        if cfg['norm']:
            with tf.name_scope("Summaries_norms"):
                self.variable_summaries(self.norms)

    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state

        output = self.batchX_placeholder
        current_x = self.batchX_placeholder
        current_x_mem = self.batchX_placeholder
        
        outputs = []
        outputs_mem = []
        
        softmaxes = []
        softmaxes_mem = []
        
        current_xes = []
        current_xes_mem = []
        
        #printtf = tf.Print(output, [output], message="Strated cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
        with tf.name_scope("Forward_pass_"+mode):
            for timestep in range(cfg['max_output_ops']):
                print("timestep " + str(timestep))
                with tf.name_scope("Step_"+str(timestep)):
                    current_input = output
            
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
                        logits_scaled = tf.multiply(logits, self.softmax_sat, name="sat_softmax")
                        softmax = tf.nn.softmax(logits_scaled, name="get_softmax")
                        #softmax = self.custom_softmax(logits_scaled, cfg)
                        #in test change to hardmax
                        if mode is "test":
                            argmax  = tf.argmax(softmax, 1, )
                            softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])

                    with tf.name_scope("Comp_mem"):
                        #run the forward pass from the mem module, hence select mem cell
                        output_mem, current_state_mem, softmax_mem, current_x_mem = self.mem.run_forward_pass(current_input, current_x_mem, cfg, mode)
                        outputs_mem.append(output_mem)
                        softmaxes_mem.append(softmax_mem)
                        current_xes_mem.append(current_x_mem)
                        
                    with tf.name_scope("Comp_output"):
                        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
                        output = self.select_op(current_input, output_mem, softmax, cfg)

                        #save the sequance of softmaxes and outputs
                        outputs.append(output)
                        softmaxes.append(softmax)
                    
                    with tf.name_scope("Comp_next_x"):
                        next_x = tf.add(tf.matmul(logits, self.params["W3"], name="state_mul_W3"), self.params["b3"], name="add_bias3")
                        current_x = next_x
                        current_xes.append(current_x)

            #build the response dict
        return dict(
                    #outputs main, op seleciton RNN
                    output = output,
                    current_state = current_state,
                    softmax = softmax,
                    outputs = outputs,
                    softmaxes = softmaxes,
                    current_x = current_x,
                    current_xes = current_xes,
                    #Outputs mem rnn
                    output_mem = output_mem,
                    current_state_mem = current_state_mem,
                    softmax_mem = softmax_mem,
                    outputs_mem = outputs_mem,
                    softmaxes_mem = softmaxes_mem,
                    current_x_mem = current_x_mem,
                    current_xes_mem = current_xes_mem
                   )


