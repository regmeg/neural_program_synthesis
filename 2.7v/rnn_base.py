from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from nn_base import NNbase

class RNN(NNbase):
    
    def __init__(self, cfg, ops, mem):

        #init parent
        super(RNN, self).__init__(cfg, ops)
        
        #placeholder for the initial state of the model
        with tf.name_scope(u"RNN_op"):
            self.init_state = tf.placeholder(cfg[u'datatype'], [None, cfg[u'state_size']], name=u"init_state")

            #set mem
            self.mem = mem
            #set random seed
            tf.set_random_seed(cfg[u'seed'])

            #model parameters
            with tf.name_scope(u"Params"):
                
                self.params[u"W"] = tf.get_variable(u"W", shape=[ cfg[u'state_size']+cfg[u'num_features'], cfg[u'state_size'] ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params[u"b"] = tf.Variable(np.zeros((cfg[u'state_size'])), dtype=cfg[u'datatype'], name=u"b")

                self.params[u"W2"] = tf.get_variable(u"W2", shape=[ cfg[u'state_size'], ops.num_of_ops ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params[u"b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg[u'datatype'], name=u"b2")
                
                self.params[u"W3"] = tf.get_variable(u"W3", shape=[ ops.num_of_ops, cfg[u'num_features'] ], dtype=cfg[u'datatype'], initializer=tf.contrib.layers.xavier_initializer())
                self.params[u"b3"] = tf.Variable(np.zeros((cfg[u'num_features'])), dtype=cfg[u'datatype'], name=u"b3")
                
                
                u'''
                self.params["W"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
                self.params["b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")
                
                self.params["W3"] = tf.Variable(tf.truncated_normal([self.ops.num_of_ops, cfg['num_features']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W3")
                self.params["b3"] = tf.Variable(np.zeros((cfg['num_features'])), dtype=cfg['datatype'], name="b3")
                '''
                
            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = u"train")
            self.total_loss_train, self.math_error_train = self.calc_loss(cfg, self.train[u"output"], mode = u"train")

            self.test = self.run_forward_pass(cfg, mode = u"test")
            self.total_loss_test, self.math_error_test = self.calc_loss(cfg, self.test[u"output"], mode = u"test")

            #calc grads and hereby the backprop step
            self.grads, self.train_step, self.norms  = self.calc_backprop(cfg)

        #write model param and grad summaries outside of all scopes
        with tf.name_scope(u"Summaries_params"):
            for param, tensor in self.params.items(): self.variable_summaries(tensor)               
               
        with tf.name_scope(u"Summaries_grads"):
            for grad, var in self.grads: 
                print u"writing grad", var.name.replace(u":",u"_")+u"_grad"
                self.variable_summaries(grad, name=var.name.replace(u":",u"_")+u"_grad")
        
        if cfg[u'norm']:
            with tf.name_scope(u"Summaries_norms"):
                self.variable_summaries(self.norms)

    #forward pass
    def run_forward_pass(self, cfg, mode=u"train"):
        current_state = self.init_state
        current_state_mem = self.init_state

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
        with tf.name_scope(u"Forward_pass_"+mode):
            for timestep in xrange(cfg[u'max_output_ops']):
                print u"timestep " + unicode(timestep)
                with tf.name_scope(u"Step_"+unicode(timestep)):
                    current_input = output
            
                    with tf.name_scope(u"Comp_softmax"):
                        input_and_state_concatenated = tf.concat([current_x, current_state], 1, name=u"concat_input_state")  # Increasing number of columns
                        _mul1 = tf.matmul(input_and_state_concatenated, self.params[u"W"], name=u"input-state_mult_W")
                        _add1 = tf.add(_mul1, self.params[u"b"], name=u"add_bias")
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
                        #logits = tf.matmul(state_dropped, self.params["W2"], name="state_mul_W2")
                        logits = tf.add(tf.matmul(state_dropped, self.params[u"W2"], name=u"state_mul_W2"), self.params[u"b2"], name=u"add_bias2") #Broadcasted addition
                        logits_scaled = tf.multiply(logits, self.softmax_sat, name=u"sat_softmax")
                        softmax = tf.nn.softmax(logits_scaled, name=u"get_softmax")
                        #softmax = self.custom_softmax(logits_scaled, cfg)
                        #in test change to hardmax
                        if mode is u"test":
                            argmax  = tf.argmax(softmax, 1, )
                            softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg[u'datatype'])

                    with tf.name_scope(u"Comp_mem"):
                        #run the forward pass from the mem module, hence select mem cell
                        #output_mem, current_state_mem, softmax_mem, current_x_mem = self.mem.run_forward_pass(current_input, current_x_mem, cfg, mode)
                        output_mem, current_state_mem, softmax_mem = self.mem.run_forward_pass(current_x, current_state_mem , timestep, cfg, mode)
                        outputs_mem.append(output_mem)
                        softmaxes_mem.append(softmax_mem)
                        current_xes_mem.append(current_x_mem)
                        
                    with tf.name_scope(u"Comp_output"):
                        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
                        output = self.select_op(current_input, output_mem, softmax, cfg)

                        #save the sequance of softmaxes and outputs
                        outputs.append(output)
                        softmaxes.append(softmax)
                    
                    with tf.name_scope(u"Comp_next_x"):
                        next_x = tf.add(tf.matmul(logits, self.params[u"W3"], name=u"state_mul_W3"), self.params[u"b3"], name=u"add_bias3")
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
                    #current_x_mem = current_x_mem,
                    #current_xes_mem = current_xes_mem
                   )


