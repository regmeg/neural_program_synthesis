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
                self.params["W"] = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
                self.params["b"] = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")

                self.params["W2"] = tf.Variable(tf.truncated_normal([cfg['state_size'], self.ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
                self.params["b2"] = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")

            #create graphs for forward pass to soft and hard selection
            self.train = self.run_forward_pass(cfg, mode = "train")
            self.total_loss_train, self.math_error_train = self.calc_loss(cfg, self.train["output"])

            self.test = self.run_forward_pass(cfg, mode = "test")
            self.total_loss_test, self.math_error_test = self.calc_loss(cfg, self.test["output"])

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

        outputs = []
        outputs_mem = []
        
        softmaxes = []
        softmaxes_mem = []
        
        #printtf = tf.Print(output, [output], message="Strated cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
        with tf.name_scope("Forward_pass_"+mode):
            for timestep in range(cfg['max_output_ops']):
                print("timestep " + str(timestep))
                with tf.name_scope("Step_"+str(timestep)):
                    current_input = output

                    with tf.name_scope("Comp_softmax"):
                        input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state")  # Increasing number of columns
                        next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, self.params["W"], name="input-state_mult_W"), self.params["b"], name="add_bias"), name="tanh_next_state")  # Broadcasted addition
                        #next_state = tf.nn.relu(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="relu_next-state")  # Broadcasted addition
                        current_state = next_state

                        #calculate softmax and produce the mask of operations
                        logits = tf.add(tf.matmul(next_state, self.params["W2"], name="state_mul_W2"), self.params["b2"], name="add_bias2") #Broadcasted addition
                        softmax = tf.nn.softmax(logits, name="get_softmax")

                        #in test change to hardmax
                        if mode is "test":
                            argmax  = tf.argmax(softmax, 1, )
                            softmax  = tf.one_hot(argmax, self.ops.num_of_ops, dtype=cfg['datatype'])

                    with tf.name_scope("Comp_mem"):
                        #run the forward pass from the mem module, hence select mem cell
                        output_mem, current_state_mem, softmax_mem = self.mem.run_forward_pass(current_input, cfg, mode)
                        outputs_mem.append(output_mem)
                        softmaxes_mem.append(softmax_mem)

                    with tf.name_scope("Comp_output"):
                        #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)
                        output = self.select_op(current_input, output_mem, softmax, cfg)

                        #save the sequance of softmaxes and outputs
                        outputs.append(output)
                        softmaxes.append(softmax)
        
            #build the response dict
        return dict(output = output,
                    current_state = current_state,
                    softmax = softmax,
                    outputs = outputs,
                    softmaxes = softmaxes,
                    output_mem = output_mem,
                    current_state_mem = current_state_mem,
                    softmax_mem = softmax_mem,
                    outputs_mem = outputs_mem,
                    softmaxes_mem = softmaxes_mem
                   )