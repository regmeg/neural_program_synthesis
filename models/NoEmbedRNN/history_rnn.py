'''
This is a history RNN which replicates the neural programmer in more detail
'''

import tensorflow as tf
import numpy as np
from nn_base import NNbase

class HistoryRNN(NNbase):
    
    def __init__(self, cfg, ops, mem_sel, op_sel):

        #init parent
        super(HistoryRNN, self).__init__(cfg, ops)
        
        #placeholder for the initial state of the model
        with tf.name_scope("RNN_hist"):
            self.init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state")
            
            #set op sel
            self.op = op_sel
            #set mem
            self.mem = mem_sel            
            #set random seed
            tf.set_random_seed(cfg['seed'])

            #model parameters
            with tf.name_scope("Params"):
                self.params["W_hist"] = tf.Variable(tf.truncated_normal([3*cfg['state_size'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W_hist")
                self.params["b_hist"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b_hist")

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
        
        states_h = []        
        outputs_op = []
        outputs_mem = []        
        softmaxes_op = []
        softmaxes_mem = []

        with tf.name_scope("Forward_pass_"+mode):
            for timestep in range(cfg['max_output_ops']):
                print("timestep " + str(timestep))
                with tf.name_scope("Step_"+str(timestep)):
                    with tf.name_scope("Comp_selections"):
                            mem_sel, softmax_mem = self.mem.select(self.batchX_placeholder, current_state, cfg, mode)
                            outputs_mem.append(mem_sel)
                            softmaxes_mem.append(softmax_mem)
                            op_sel, softmax_op = self.op.select(self.batchX_placeholder, current_state, output, mem_sel, cfg, mode)
                            outputs_op.append(op_sel)
                            softmaxes_op.append(softmax_op)           
                    with tf.name_scope("Conc_op_mem_softmax"):
                            op_sel_t = tf.transpose(self.params["W2_op"], name='transpose_op_W2')
                            mem_sel_t = tf.transpose(self.params["W2_mem"], name='transpose_mem_W2')
                            op_sel_m = tf.matmul(softmax_op, op_sel_t)
                            mem_sel_m = tf.matmul(softmax_mem, mem_sel_t)
                            c_op_mem = tf.concat([op_sel_m, mem_sel_m], 1, name="concat_op_sel_m_mem_sel_m")
                    
                    with tf.name_scope("Comp_state"):
                        input_and_state_concatenated = tf.concat([c_op_mem, current_state], 1, name="concat_input_state")
                        
                        _mul1 = tf.matmul(input_and_state_concatenated, self.params["W_hist"], name="input-state_mult_W")
                        _add1 = tf.add(_mul1, self.params["b_hist"], name="add_bias")
                        if   cfg["state_fn"] == "tanh":
                            next_state = tf.tanh(_add1, name="tanh_next_state")
                        elif cfg["state_fn"] == "relu":
                            next_state = tf.nn.relu(_add1, name="relu_next_state")
  
                        states_h.append(next_state)
                        current_state = next_state
                        output = op_sel
        
            #build the response dict
        return dict(states_h = states_h,
                    outputs_op = outputs_op,
                    outputs_mem = outputs_mem,
                    softmaxes_op = softmaxes_op,
                    softmaxes_mem = softmaxes_mem,
                    mem_sel = mem_sel,
                    softmax_mem = softmax_mem,
                    op_sel = op_sel,
                    softmax_op = softmax_op,
                    current_state = current_state,
                    output = output
                   )