import tensorflow as tf
import numpy as np

class Operations:

    def __init__(self, cfg):
        self.cfg = cfg
        self.ops = [self.tf_inpt_len, self.tf_divide, self.tf_input_mem_concat, self.tf_add]
        #self.ops = [self.tf_add, self.tf_multiply, self.tf_stall, self.tf_input_mem_concat]
        self.num_of_ops = len(self.ops)

    #model operations
    #for each reduce based operation, result is reshaped and repadded to fit the model working size num_featuresxbatch_size
    def tf_add(self, inpt, mem_sel=None):
        with tf.name_scope("tf_add"):
            result = tf.reduce_sum(inpt, axis = 1, name = "add")
            reshape = tf.reshape(result, [self.cfg['batch_size'], -1], name = "reshape")
            pad_res = tf.pad(reshape, [[0,0],[0,self.cfg['num_features'] - 1]], "CONSTANT", name="pad")
            return  pad_res
    
    def tf_multiply(self ,inpt, mem_sel=None):
        with tf.name_scope("tf_multiply"):
            result = tf.reduce_prod(inpt, axis = 1, name = "mult")
            reshape = tf.reshape(result , [self.cfg['batch_size'], -1], name = "reshape")
            pad_res = tf.pad(reshape, [[0,0],[0,self.cfg['num_features'] - 1]], "CONSTANT", name="pad")
            return pad_res

    #stall operation is simply simulated as returning the input back
    def tf_stall(self, inpt, mem_sel=None):
        with tf.name_scope("tf_stall"):
            reshape = tf.reshape(inpt , [self.cfg['batch_size'], self.cfg['num_features']], name = "reshape")
            return  reshape
    
    #get input lenght, asssing all values to ones and then reduce
    def tf_inpt_len(self,inpt, mem_sel=None):
            with tf.name_scope("tf_inpt_len"):
                masked_zeros = tf.where( tf.equal(inpt,tf.zeros_like(inpt, dtype=self.cfg['datatype'])), tf.ones_like(inpt, dtype=self.cfg['datatype']), inpt, name="clean_zeros")
                inpt_ones = tf.divide(masked_zeros, masked_zeros, name="produce_ones")
                result = tf.reduce_sum(inpt_ones, axis = 1, name = "len_reduce")
                reshape = tf.reshape(result, [self.cfg['batch_size'], -1], name = "reshape")
                pad_res = tf.pad(reshape, [[0,0],[0,self.cfg['num_features'] - 1]], "CONSTANT", name="pad")
                return  pad_res
    
    #divide selected delected numbers in the row, convetion is:
    # only the first elem can be 0, other zeros are replaced with ones, so that no infs are produced
    #the division is achieved by keeping the first elem as it is and then producing repriocals for all the rest, hence the reductions produces division
    def tf_divide(self, inpt, mem_sel=None):
        with tf.name_scope("tf_divide"):
            repriocal = tf.reciprocal(inpt, name="reciprocal")
            reg_slice = tf.slice(inpt, [0,0], [self.cfg['batch_size'],1], name="regular_slice")
            repr_slice = tf.slice(repriocal, [0,1], [self.cfg['batch_size'], self.cfg['num_features']-1], name="repriocal_slice")
            intp  = tf.concat([reg_slice, repr_slice],1, name="reg_repr_concat")
            masked_ones = tf.where(tf.is_inf(intp), tf.ones_like(inpt, dtype=self.cfg['datatype']), intp, name="clean_inf")
            return self.tf_multiply(masked_ones)

        
    #get value from saved store 
    def tf_input_mem_concat(self, inpt, mem_sel=None):
        with tf.name_scope("tf_input_mem_concat"):
            #inpt_slice = tf.slice(inpt, [0,0], [self.cfg['batch_size'],1], name="tf_inp_mem_cnt_slice1")
            mem_slice = tf.slice(mem_sel, [0,0], [self.cfg['batch_size'],1], name="mem_slice")
            pad_res = tf.pad(mem_slice, [[0,0],[1,self.cfg['num_features'] - 2]], "CONSTANT", name="pad")
            return  tf.add(inpt, pad_res, name="add")
    
    ######helper functions######
    def not_zero(self, inpt, mem_sel=None):
        with tf.name_scope("not_zero"):
            greater = tf.greater(inpt,tf.zeros_like(inpt, dtype=self.cfg['datatype']))
            less = tf.less(inpt, tf.zeros_like(inpt, dtype=self.cfg['datatype']))
            not_zero = tf.logical_or(greater, less)
            return not_zero

            
    def add_dummy(self, inpt, mem_sel=None):
        with tf.name_scope("Select_mem"):
             return tf.add(inpt, mem_sel)


    