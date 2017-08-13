import tensorflow as tf
from ops import Operations

class RNN:
    
    def __init__(self, cfg, variable_summaries):   
        #model constants
        self.dummy_matrix = tf.zeros([cfg['batch_size'], cfg['num_features']], dtype=cfg['datatype'], name="dummy_constant")

        #model placeholders
        self.batchX_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], None], name="batchX")
        self.batchY_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], None], name="batchY")

        self.init_state = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['state_size']], name="init_state")


        #set random seed
        tf.set_random_seed(cfg['seed'])

        #model parameters
        self.W = tf.Variable(tf.truncated_normal([cfg['state_size']+cfg['num_features'], cfg['state_size']], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']), dtype=cfg['datatype'], name="W")
        self.b = tf.Variable(np.zeros((cfg['state_size'])), dtype=cfg['datatype'], name="b")
        variable_summaries(W)
        variable_summaries(b)

        self.W2 = tf.Variable(tf.truncated_normal([cfg['state_size'], ops.num_of_ops], -1*cfg['param_init'], cfg['param_init'], dtype=cfg['datatype']),dtype=cfg['datatype'], name="W2")
        self.b2 = tf.Variable(np.zeros((ops.num_of_ops)), dtype=cfg['datatype'], name="b2")
        variable_summaries(W2)
        variable_summaries(b2)
        
        #create graphs for forward pass to soft and hard selection
        self.output_train, self.current_state_train, self.softmax_train, self.outputs_train, self.softmaxes_train =                     self.run_forward_pass(cfg, mode = "train")
        
        self.total_loss_train, self.math_error_train = calc_loss(cfg, self.output_train)

        self.output_test, self.current_state_test, self.softmax_test, self.outputs_test, self.softmaxes_test =                         self.run_forward_pass(cfg, mode = "test")

        self.total_loss_test, self.math_error_test = calc_loss(cfg, self.output_test)
    
        #calc grads and hereby the backprop step
        self.train_step  = calc_backprop(cfg)
    #forward pass
    def run_forward_pass(self, cfg, mode="train"):
        current_state = self.init_state

        output = self.batchX_placeholder

        outputs = []

        softmaxes = []

        #printtf = tf.Print(output, [output], message="Strated cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")

        for timestep in range(cfg['max_output_ops']):
            print("timestep " + str(timestep))
            current_input = output



            input_and_state_concatenated = tf.concat([current_input, current_state], 1, name="concat_input_state")  # Increasing number of columns
            next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, self.W, name="input-state_mult_W"), self.b, name="add_bias"), name="tanh_next_state")  # Broadcasted addition
            #next_state = tf.nn.relu(tf.add(tf.matmul(input_and_state_concatenated, W, name="input-state_mult_W"), b, name="add_bias"), name="relu_next-state")  # Broadcasted addition
            current_state = next_state

            #calculate softmax and produce the mask of operations
            logits = tf.add(tf.matmul(next_state, self.W2, name="state_mul_W2"), self.b2, name="add_bias2") #Broadcasted addition
            softmax = tf.nn.softmax(logits, name="get_softmax")

            #in test change to hardmax
            if mode is "test":
                argmax  = tf.argmax(softmax, 1, )
                softmax  = tf.one_hot(argmax, ops.num_of_ops, dtype=cfg['datatype'])
            #in the train mask = saturated softmax for all ops. in test change it to onehot(hardmax)

            #######################
            #perform op selection #
            #######################

            #perform all ops in the current timestep intput and save output results together with the op name

            op_res = []
            for op in ops.ops:
                name = op.__name__
                op_outp = op(current_input)
                op_res.append((name, op_outp))

            #slice softmax results for each operation
            ops_softmax = []
            for i, op in enumerate(ops.ops):
                name = "slice_"+op.__name__+"_softmax_val"
                softmax_slice = tf.slice(softmax, [0,i], [cfg['batch_size'],1], name=name)
                ops_softmax.append(softmax_slice)


            #apply softmax on each operation so that operation selection is performed
            ops_final = []
            for i,res in enumerate(op_res):
                name = "mult_"+res[0]+"_softmax"
                op_selection =  tf.multiply(res[1], ops_softmax[i], name=name)
                ops_final.append(op_selection)


            #add results from all operation with applied softmax together
            output = tf.add_n(ops_final)

            #save the sequance of softmaxes and outputs
            outputs.append(output)
            softmaxes.append(softmax)
        #printtf = tf.Print(output, [output], message="Finished cycle")
        #output = tf.reshape( printtf, [batch_size, -1], name = "dummu_rehap")
        return output, current_state, softmax, outputs, softmaxes

    #cost function
    def calc_loss(self,cfg, output):
        #reduced_output = tf.reshape( tf.reduce_sum(output, axis = 1, name="red_output"), [batch_size, -1], name="resh_red_output")
        math_error = tf.multiply(tf.constant(0.5, dtype=cfg['datatype']), tf.square(tf.subtract(output , self.batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")

        total_loss = tf.reduce_sum(math_error, name="red_total_loss")
        return total_loss, math_error

    def calc_backprop(self, cfg):
        grads_raw = tf.gradients(self.total_loss_train, [self.W,self.b,self.W2,self.b2], name="comp_gradients")

        #clip gradients by value and add summaries
        if cfg['norm']:
            print("norming the grads")
            grads, norms = tf.clip_by_global_norm(grads_raw, cfg['grad_norm'])
            variable_summaries(norms)
        else:
            grads = grads_raw

        for grad in grads: variable_summaries(grad)


        train_step = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['epsilon'] ,name="AdamOpt").apply_gradients(zip(grads, [self.W,self.b,self.W2,self.b2]), name="min_loss")
        print("grads are")
        print(grads)

        return train_step