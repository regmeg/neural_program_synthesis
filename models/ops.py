import tensorflow as tf
import numpy as np

class Operations:

    def __init__(self, cfg, mem):
        self.batch_size = cfg['batch_size']
        self.num_features = cfg['num_features']
        self.ops = [self.tf_add, self.tf_multiply, self.tf_stall]
        self.num_of_ops = len(self.ops)
        self.mem = mem 

    #model operations
    #for each reduce based operation, result is reshaped and repadded to fit the model working size num_featuresxbatch_size
    def tf_add(wself, inpt):
        result = tf.reduce_sum(inpt, axis = 1, name = "tf_add")
        reshape = tf.reshape(result, [self.batch_size, -1], name = "tf_add_reshape")
        pad_res = tf.pad(reshape, [[0,0],[0,self.num_features - 1]], "CONSTANT", name="tf_add_pad")
        return  pad_res
    
    def tf_multiply(self ,inpt):
        result = tf.reduce_prod(inpt, axis = 1, name = "tf_mult")
        reshape = tf.reshape(result , [self.batch_size, -1], name = "tf_mult_reshape")
        pad_res = tf.pad(reshape, [[0,0],[0,self.num_features - 1]], "CONSTANT", name="tf_mult_pad")
        return pad_res

    #stall operation is simply simulated as returning the input back
    def tf_stall(self, inpt):
        return  inpt
    
    #get input lenght, asssing all values to ones and then reduce
    def tf_inpt_len(self,inpt):
        inpt = tf.assign(inpt, np.ones(inpt.get_shape()), name="tf_add_assign_ones")
        result = tf.reduce_sum(inpt, axis = 1, name = "tf_add")
        reshape = tf.reshape(result, [self.batch_size, -1], name = "tf_add_reshape")
        pad_res = tf.pad(reshape, [[0,0],[0,self.num_features - 1]], "CONSTANT", name="tf_add_pad")
        return  pad_res
    
    #divide selected delected numbers in the row, convetion is:
    # only the first elem can be 0, other zeros are replaced with ones, so that no infs are produced
    #the division is achieved by keeping the first elem as it is and then producing repriocals for all the rest, hence the reductions produces division
    def tf_divide(inpt):
        repriocal = tf.reciprocal(inpt, name="tf_div_rerp")
        reg_slice = tf.slice(inpt, [0,0], [self.batch_size,1], name="tf_div_reg_slice")
        repr_slice = tf.slice(repriocal, [0,1], [self.batch_size, self.num_features-1], name="tf_div_repr_slice")
        intp  = tf.concat([reg_slice, repr_slice],1, name="tf_div_reg_repr")
        masked_ones = tf.where(tf.is_inf(intp), tf.ones_like(inpt, dtype=cfg['datatype']), intp, name="tf_div_clean_inf")
        return self.tf_multiply(masked_ones)

    #save produced input in temp mem
    def tf_save_inpt(self,inpt):
        return  inpt
        
    #get value from saved store 
    def tf_saved_input_concat(self,inpt):
        return  inpt
    
    ######helper functions######
    def not_zero(inpt):
        greater = tf.greater(inpt,tf.zeros_like(inpt, dtype=cfg['datatype']))
        less = tf.less(inpt, tf.zeros_like(inpt, dtype=cfg['datatype']))
        not_zero = tf.logical_or(greater, less)
        return not_zero

        
    