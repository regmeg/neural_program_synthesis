'''
This module contains all the operations implemented in numpy and data generation methods and as well as the program execution envfiroment for the reinforcment learning model
'''
import numpy as np
from functools import reduce
from numpy.random import RandomState
from collections import OrderedDict
import random

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

def np_get_len(vec):
    return vec.size

def np_div(vec1, vec2):
    if vec2[0] == 0: return 0
    return vec1[0] / vec2[0]

#val is going to come from the inp, vec is comming from the mem
def np_sub(val, vec):
    return np.subtract(vec,val[0])

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
    

def np_add_env(vec, mem=None):
    return wrap_for_env(vec, np_add(vec))
np_add_env.req_mem = False

def np_mult_env(vec, mem=None):
    return wrap_for_env(vec, np_mult(vec))
np_mult_env.req_mem = False

def np_stall_env(vec, mem=None):
    return vec
np_stall_env.req_mem = False

def np_get_size_env(vec, mem=None):
    return wrap_for_env(vec, np_get_len(vec))
np_get_size_env.req_mem = False

def np_div_env(vec, mem):
    return wrap_for_env(vec, np_div(vec, mem))
np_div_env.req_mem = True

def np_sub_env(vec, mem):
    return np_sub(vec, mem)
np_sub_env.req_mem = True

def np_corrupt(vec):
    return list(map(lambda x: x+random.randint(0, 255)/random.randint(1,10) , vec))

class OpsEnv(object):
        
    def __init__(self, cfg):   
        self.cfg = cfg
        self.ops =      [np_add_env, np_mult_env, np_get_size_env, np_div_env, np_sub_env] # dont use stall, it is pointless for dynamic RNN
        self.ops_mem =  [np_add_env, np_mult_env, np_get_size_env, np_stall_env] #dont use div, it does require another mem acccess
        self.num_of_ops = len(self.ops)
        self.num_of_ops_mem = len(self.ops_mem)
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
    #self.ops_env.apply_op(_selection, _selection_mem, output, batchX, batchY)
    def apply_op(self, selections, selections_mem, prev_sel, prev_sel_mem, inptX, batchX, batchY):
            """
            print("selections", selections)
            print("prev_sel", prev_sel)
            
            print("selections_mem", selections_mem)
            print("prev_sel_mem", prev_sel_mem)
            """
            output = []
            output_mem = []
            mem_mask = []
            for i, sel in enumerate(selections):
                #print("i", i, "sel", sel)
                mem_fn = self.ops_mem[int(selections_mem[i])]
                #mem_fn = self.ops[2]
                fn = self.ops[int(sel)]
                #print("i",i,"sel",sel,"fn", fn.__name__)
                batch = batchX[i]
                inp = inptX[i]
                mem = mem_fn(batch, np.zeros_like(batch))
                out = fn(inp, mem)
                try:
                    corrupt = False
                    #if same operation is selected, which produces the same output, corrupt the result
                    """
                    print("prev_sel[i]", prev_sel[0][i])
                    print("inp", inp)
                    print("out", out)
                    """
                    pr_sel = int(prev_sel[0][i])                    
                    if int(sel) == pr_sel:
                        #if requires memory, check if memory selections are the same
                        if fn.req_mem:
                            pr_sel_mem = int(prev_sel_mem[0][i])
                            if int(selections_mem[i]) == pr_sel_mem:
                                if (out == inp).all():
                                    corrupt = True
                        else:
                            if (out == inp).all():
                                corrupt = True
                    
                    if corrupt: 
                        out = np_corrupt(out)
                    """
                    print("corrupt", corrupt, "out", out)
                    """
                except IndexError:
                    pass
                #print("inp", inp, "out", out)
                output.append(out)
                output_mem.append(mem)
                mem_mask.append(fn.req_mem is True)
                    
            output = np.vstack(output)
            output_mem = np.vstack(output_mem)
            mem_mask = np.vstack(mem_mask)
            #print("output")req_mem
            #print(output)
            error = batchY - output
            #print("error")
            #print(error)
            #print("error_ab")
            error_ab = abs(error)
            #print(error_ab)
            #print("error_flat")
            error_flat = np.vstack(error_ab.sum(axis=1))
            math_error = np.vstack(error_ab.sum(axis=1))
            error_flat[error_flat>0] = 2*self.cfg['max_reward']
            #print(error_flat)
            #print("math_error")
            #print(math_error)
            #print("erro_")
            #print(error_sum)
            return output, output_mem, error_flat, math_error, mem_mask