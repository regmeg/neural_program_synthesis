import numpy as np
from functools import reduce
from numpy.random import RandomState
from collections import OrderedDict

#sample gen functions
def np_add(vec):
    return reduce((lambda x, y: x + y),vec)

def np_mult(vec):
    return reduce((lambda x, y: x * y),vec)


def np_stall(vec):
    return vec

def np_avg_val(vec):
    return np.average(vec)

def np_center(vec):
    return np.subtract(vec, np.average(vec))

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
        
def split_train_test(x, y , test_ratio):
    
    assert len(x) == len(y), 'Model expects x and y shapes to be the same'
    
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

def shuffle_data(x,y):
    assert len(x) == len(y), 'Model expects x and y shapes to be the same'
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]

def wrap_for_env(vec, val):
    res = np.array([val])
    return np.lib.pad(res, ( 0, vec.shape[0] - 1 ), 'constant', constant_values=(0))
    

def np_add_env(vec):
    return wrap_for_env(vec, np_add(vec))

def np_mult_env(vec):
    return wrap_for_env(vec, np_mult(vec))

def np_stall_env(vec):
    return vec

class OpsEnv(object):
        
    def __init__(self, cfg):   
        self.cfg = cfg
        self.ops = [np_add_env, np_mult_env ,np_stall_env]
        self.num_of_ops = len(self.ops)
    '''
    def apply_op(self, selections, inptX, batchY):
            output = []
            for i, sel in enumerate(selections):
                print(i)
                fn = self.ops[int(sel)]
                output.append(fn(inptX[i]))
            output = np.vstack(output)
            error = batchY - output
            error = np.vstack(0.5*np.square(error.sum(axis=1)))
            return output, error
    '''
    
    def apply_op(self, selections, inptX, batchY):
            output = []
            for i, sel in enumerate(selections):
                fn = self.ops[int(sel)]
                #print("i",i,"sel",sel,"fn", fn.__name__)
                inp = inptX[i]
                out = fn(inp)
                #print("inp", inp, "out", out)
                output.append(out)
            output = np.vstack(output)
            #print("output")
            #print(output)
            error = batchY - output
            #print("error")
            #print(error)
            #print("error_ab")
            error_ab = abs(error)
            #print(error_ab)
            #print("error_flat")
            error_flat = np.vstack(error_ab.sum(axis=1))
            error_flat[error_flat>0] = 2*self.cfg['max_reward']
            #print(error_flat)
            
            #print("erro_")
            #print(error_sum)
            return output, error_flat