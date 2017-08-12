import numpy as np

#sample gen functions
def np_add(vec):
    return reduce((lambda x, y: x + y),vec)

def np_mult(vec):
    return reduce((lambda x, y: x * y),vec)

def np_stall(vec):
    return vec

def samples_generator(fn, shape, rng, seed):
    '''
    Generate random samples for the model:
    @fn - function to be applied on the input features to get the ouput
    @shape - shape of the features matrix (num_samples, num_features)
    @rng - range of the input features to be generated within (a,b)
    Outputs a tuple of input and output features matrix
    '''
    prng = RandomState(seed)
    x = (rng[1] - rng[0]) * prng.random_sample(shape) + rng[0]
    y = np.apply_along_axis(fn, 1, x).reshape((shape[0],-1))
    z = np.zeros((shape[0],shape[1] - y.shape[1]))
    y = np.concatenate((y, z), axis=1)
    
    return x,y

def get_syn_fn(fn_name):
    if   fn_name == "np_add":   return np_add
    elif fn_name == "np_mult":  return np_mult
    elif fn_name == "np_stall": return np_stall
    else: raise Exception('Function passed by the flag to be synthesised has not been defined')
        
def split_train_test(x, y , test_ratio):
    
    if y.shape != x.shape:
        raise Exception('Model expects x and y shapes to be the same')
    
    test_len  = int(x.shape[0]*test_ratio)
    train_len = x.shape[0] - test_len

    x_train = x[0:train_len][:]
    x_test  = x[-test_len:][:]
    y_train = y[0:train_len][:]
    y_test  = y[-test_len:][:]
    
    train_shape = (train_len, x.shape[1])
    test_shape = (test_len, x.shape[1])
    
    if test_ratio == 0:
        x_test = np.zeros(test_shape)
        y_test = np.zeros(test_shape)

    if y_train.shape != train_shape or x_train.shape != train_shape or x_test.shape != test_shape or y_test.shape != test_shape:
        raise Exception('One of the conversion test/train shapes gone wrong')
    
    return  x_train, x_test, y_train, y_test
