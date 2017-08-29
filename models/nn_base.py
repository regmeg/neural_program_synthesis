import tensorflow as tf
import numpy as np
from collections import OrderedDict
from params import get_cfg
from tensorflow.python.ops import random_ops

class NNbase(object):
    #define model params as a static dict,so that when grad are computed, all params are used from child op and mem selections rrns    
    params = OrderedDict()
    model_vars = OrderedDict()
    
    #share the inputs and ouputs between the RNNs as well
    with tf.name_scope("Batches"):
        batchX_placeholder = None
        batchY_placeholder = None
        use_both_losses = None
    
    def __init__(self, cfg, ops):   
        self.ops = ops
        self.dropout_cntr = 0
        #set batches
        if NNbase.batchX_placeholder is None:
            NNbase.batchX_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['num_features']], name="batchX")
        if NNbase.batchY_placeholder is None:
            NNbase.batchY_placeholder = tf.placeholder(cfg['datatype'], [cfg['batch_size'], cfg['num_features']], name="batchY")
        if NNbase.use_both_losses is None:
            NNbase.use_both_losses = tf.placeholder(dtype=tf.bool, name='use_both_penalty_and_math_loss')
        if "global_step" not in NNbase.model_vars:
            self.model_vars["global_step"] = tf.Variable(0, name='global_step', trainable=False, dtype=cfg['datatype'])

        #model constants
        with tf.name_scope("Constants"):
            self.dummy_matrix = tf.zeros([cfg['batch_size'], cfg['num_features']], dtype=cfg['datatype'], name="dummy_constant")            
            self.softmax_sat = tf.constant(np.full((cfg['batch_size'], 1), cfg['softmax_sat']), dtype=cfg['datatype'], name="softmax_sat") 

    def select_op(self,current_input,mem_selection, softmax, cfg):
            #######################
            #perform op selection #
            #######################

            #perform all ops in the current timestep intput and save output results together with the op name
        with tf.name_scope("Select_op"):
            op_res = []
            for op in self.ops.ops:
                name = op.__name__
                op_outp = op(current_input, mem_selection)
                op_res.append((name, op_outp))

            #slice softmax results for each operation
            ops_softmax = []
            for i, op in enumerate(self.ops.ops):
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
            return output
        
    def select_mem(self, mem_cell, softmax, cfg):
            #######################
            #perform mem selection #
            #######################
        with tf.name_scope("Select_mem"):
            #slice softmax results for each mem cell
            mem_softmax = []
            for i, op in enumerate(self.ops.ops):
                name = "slice_"+op.__name__+"_softmax_val_mem"
                softmax_slice = tf.slice(softmax, [0,i], [cfg['batch_size'],1], name=name)
                mem_softmax.append(softmax_slice)

            #apply softmax on each mem cell so that operation seletion is performed
            mems_final = []
            for i,mem in enumerate(mem_cell):
                name = "mult_"+op.__name__+"_softmax_mem"
                mem_selection =  tf.multiply(mem, mem_softmax[i], name=name)
                mems_final.append(mem_selection)

            #add results from all operation with applied softmax together
            output = tf.add_n(mems_final)
            return output

    #cost function
    def calc_loss(self,cfg, output, mode="train"):
        
        with tf.name_scope("Loss_comp_"+mode):        
            #calc math error
            with tf.name_scope("math_error"):
                math_error = tf.multiply(tf.constant(cfg['loss_weight'], dtype=cfg['datatype']), tf.square(tf.subtract(output , self.batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")
            #calc sofmax penalties
            if mode == "train" and cfg["pen_sofmax"]:
                sofmax_op_pen = [self.skewed_sig_dev(smax, num_ops = self.ops.num_of_ops) for smax in self.train['softmaxes']]
                sofmax_mem_pen = [self.skewed_sig_dev(smax, num_ops = self.ops.num_of_ops) for smax in self.train['softmaxes_mem']]
                sofmax_penalty = tf.reduce_sum(sofmax_op_pen, name="red_sofmax_op_pen") + tf.reduce_sum(sofmax_mem_pen, name="red_sofmax_mem_pen")
            else :
                sofmax_penalty = 0
            #calc total error
            with tf.name_scope("Total_loss_comp"):
                max_error_tot = tf.reduce_sum(math_error, name="red_math_loss")
                #make it inversly propotionate to the math error, if math error is big, penelise it less
                #it depends on  error for a batch not for the whole error - hence it should kickoff at a smaller threshhold -
                #num_samples/batch_size
                #num_batches = (cfg["num_samples"]/cfg["batch_size"])*cfg["test_ratio"]
                #sofmax_pen_r = tf.cast((1 - tf.sigmoid( (num_batches*max_error_tot/50) - 10 ) ) * cfg["smax_pen_r"], cfg['datatype'])
                '''
                sofmax_pen_r = tf.divide( tf.cast(20*cfg["smax_pen_r"], cfg['datatype']) ,
                                          tf.sqrt(max_error_tot + tf.cast(cfg['epsilon'], cfg['datatype'])) ,
                               name = "cal_smax_pen")
                '''
                
                '''
                total_loss = tf.cond(self.use_both_losses, 
                                     lambda : tf.cast(max_error_tot + sofmax_pen_r*sofmax_penalty, cfg['datatype']),
                                     lambda : tf.cast(sofmax_penalty, cfg['datatype'])
                                    )
                '''
                sofmax_pen_r = cfg["smax_pen_r"]
                total_loss = max_error_tot + (sofmax_pen_r*sofmax_penalty)
        return total_loss, max_error_tot

    def calc_backprop(self, cfg):
        print(list(self.params.values()))
        with tf.name_scope("Grads"):
            grads = tf.gradients(self.total_loss_train, list(self.params.values()), name="comp_gradients")

           
            if cfg['add_noise']:
                noisy_gradients = []
                for grad in grads:
                    denom = tf.pow( (1+self.model_vars["global_step"]), tf.cast(0.55, cfg['datatype']))
                    variance =  tf.cast(1/denom, cfg['datatype'])
                    gradient_shape = grad.get_shape()
                    noise = random_ops.truncated_normal(gradient_shape, stddev=tf.sqrt(variance), dtype=cfg['datatype'])
                    noisy_gradients.append(grad + noise)
                grads = noisy_gradients
            
            if cfg['augument_grad']:
                augumented_grads = []
                param_names = [tensor.name.replace(":","_") for param, tensor in self.params.items()]
                for i, grad in enumerate(grads): 
                        if "W3" or "b3" in param_names[i]: 
                            augumented_grads.append(grad)
                        else:
                            aug_ratio = tf.log1p(self.math_error_train)* tf.cast(cfg['softmax_sat'], cfg['datatype'])
                            augumented_grads.append(aug_ratio*grad)
                grads = augumented_grads
                
                        #clip gradients by norm and add summaries
            if cfg['norm']:
                print("norming the grads")
                grads, norms = tf.clip_by_global_norm(grads, cfg['grad_norm'])
            elif cfg['clip']:
                print("clipping the grads")
                grads = [tf.clip_by_value(grad, cfg['grad_clip_val_min'], cfg['grad_clip_val_max']) for grad in grads]
                norms = []
            else:
                grads = grads
                norms = []
                
        with tf.name_scope("Train_step"):
            train_step = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['epsilon'] ,name="AdamOpt").apply_gradients(zip(grads, list(self.params.values())), global_step=self.model_vars["global_step"], name="min_loss")
            print("grads are")
            print(grads)
            print("norm is ")
            print(norms)
            return grads, train_step, norms
        
    #funciton for calculating sofmax loss - idea there is to penelise the most if all sofmaxes are the same, that if they are all 1/nump of ops, and have the least penelty for sofmaxes that are 1 or 0
    def skewed_sig_dev(self, x, num_ops = 3, scale = 10):
        with tf.name_scope("softmax_pen"):
            worst_case = 1 / num_ops
            shifted_x = x - worst_case + 0.02
            scaled_x = scale*shifted_x
            nom = scale*tf.exp(-scaled_x)+scale*num_ops*tf.exp(-num_ops*scaled_x)
            denom = tf.square(tf.exp(-scaled_x) + 1)*tf.square(tf.exp(-(num_ops-1)*scaled_x) + 1)
            res = nom/denom
            return worst_case*((res)*scale**3)
    '''
    def custom_softmax(self, x, cfg, base = 180):
        maxx = tf.reduce_max(x, axis=1, keep_dims=True)
        powx = tf.pow(tf.cast(base, cfg['datatype']), x-maxx)
        reduced  = tf.reduce_sum(powx, axis=1, keep_dims=True)
        maxg = powx/ reduced
        return maxg

    '''
    '''
    def custom_softmax(self, x, cfg, base = 1):
        maxx = tf.reduce_max(x, axis=1, keep_dims=True)
        maxg = x/ (maxx + 0.1)
        #powx = tf.pow(maxg, tf.cast(base, cfg['datatype']))
        powx = maxg
        #reduced  = tf.reduce_sum(powx, axis=1, keep_dims=True)
        return powx
    '''
    def custom_softmax(self, x, cfg, base = 10):
        maxx = tf.reduce_max(x, axis=1, keep_dims=True)
        maxg = x - maxx
        #maxg =tf.nn.relu(maxg)
        #powx = maxg
        #for i in range(base):
        #    powx = tf.sqrt(powx)
        powx = maxg
        return powx
    
    def variable_summaries(self, var, name=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""        
        if name is None: name = var.name.replace(":","_")
        with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)