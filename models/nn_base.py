import tensorflow as tf
import numpy as np
from collections import OrderedDict
from params import get_cfg

class NNbase(object):
    #define model params as a static dict,so that when grad are computed, all params are used from child op and mem selections rrns    
    params = OrderedDict()
    
    #share the inputs and ouputs between the RNNs as well
    with tf.name_scope("Batches"):
        batchX_placeholder = tf.placeholder(get_cfg()['datatype'], [get_cfg()['batch_size'], get_cfg()['num_features']], name="batchX")
        batchY_placeholder = tf.placeholder(get_cfg()['datatype'], [get_cfg()['batch_size'], get_cfg()['num_features']], name="batchY")
    
    def __init__(self, cfg, ops):   
        self.ops = ops
        #model constants
        with tf.name_scope("Constants"):
            self.dummy_matrix = tf.zeros([cfg['batch_size'], cfg['num_features']], dtype=cfg['datatype'], name="dummy_constant")


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
        #reduced_output = tf.reshape( tf.reduce_sum(output, axis = 1, name="red_output"), [batch_size, -1], name="resh_red_output")
        with tf.name_scope("Loss_comp_"+mode):        
            math_error = tf.multiply(tf.constant(0.5, dtype=cfg['datatype']), tf.square(tf.subtract(output , self.batchY_placeholder, name="sub_otput_batchY"), name="squar_error"), name="mult_with_0.5")

            total_loss = tf.reduce_sum(math_error, name="red_total_loss")
        return total_loss, math_error

    def calc_backprop(self, cfg):
        print(list(self.params.values()))
        with tf.name_scope("Grads"):
            grads_raw = tf.gradients(self.total_loss_train, list(self.params.values()), name="comp_gradients")

            #clip gradients by value and add summaries
            if cfg['norm']:
                print("norming the grads")
                grads, norms = tf.clip_by_global_norm(grads_raw, cfg['grad_norm'])
            else:
                grads = grads_raw
                norms = []
                
        with tf.name_scope("Train_step"):
            train_step = tf.train.AdamOptimizer(cfg['learning_rate'], cfg['epsilon'] ,name="AdamOpt").apply_gradients(zip(grads, list(self.params.values())), name="min_loss")
            print("grads are")
            print(grads)
            return grads, train_step, norms
    
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