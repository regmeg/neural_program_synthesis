import tensorflow as tf

class Operations:

    def __init__(self, cfg):
        self.batch_size = cfg['batch_size']
        self.num_features = cfg['num_features']
        self.ops = [self.tf_add, self.tf_multiply, self.tf_stall]
        self.num_of_ops = len(self.ops)

    #model operations
    #for each reduce based operation, result is reshaped and repadded to fit the model working size num_featuresxbatch_size
    def tf_add(self, inpt):
        result = tf.reduce_sum(inpt, axis = 1, name = "tf_add")
        reshape = tf.reshape(result, [self.batch_size, -1], name = "tf_add_reshape")
        pad_res = tf.pad(reshape, [[0,0],[0,self.num_features - 1]], "CONSTANT", name="tf_add_pad")
        return  
    
    def tf_multiply(self ,inpt):
        result = tf.reduce_prod(inpt, axis = 1, name = "tf_mult")
        reshape = tf.reshape(result , [self.batch_size, -1], name = "tf_mult_reshape")
        pad_res = tf.pad(reshape, [[0,0],[0,self.num_features - 1]], "CONSTANT", name="tf_mult_pad")
        return

    #stall operation is simply simulated as returning the input back
    def tf_stall(self, inpt):
        return  inpt