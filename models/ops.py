import tensorflow as tf

#model operations
def tf_multiply(inpt, batch_size):
    return tf.reshape( tf.reduce_prod(inpt, axis = 1, name = "tf_mult"), [batch_size, -1], name = "tf_mult_reshape")

def tf_add(inpt, batch_size):
    return  tf.reshape( tf.reduce_sum(inpt, axis = 1, name = "tf_add"), [batch_size, -1], name = "tf_add_reshape")

def tf_stall(a):
    return a