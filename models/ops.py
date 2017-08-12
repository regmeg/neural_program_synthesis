import tensorflow as tf

class Operations:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.ops = [self.tf_multiply, self.tf_add, tf_stall]
        self.num_of_ops = len(self.ops)

    #model operations
    def tf_multiply(self ,inpt):
        return tf.reshape( tf.reduce_prod(inpt, axis = 1, name = "tf_mult"), [self.batch_size, -1], name = "tf_mult_reshape")

    def tf_add(self, inpt):
        return  tf.reshape( tf.reduce_sum(inpt, axis = 1, name = "tf_add"), [self.batch_size, -1], name = "tf_add_reshape")

    def tf_stall(self, a):
        return a