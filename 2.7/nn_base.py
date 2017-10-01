from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from params import get_cfg
from tensorflow.python.ops import random_ops
from itertools import izip
from itertools import imap

class NNbase(object):
    #define model params as a static dict,so that when grad are computed, all params are used from child op and mem selections rrns    
    params = OrderedDict()
    model_vars = OrderedDict()
    
    #share the inputs and ouputs between the RNNs as well
    with tf.name_scope(u"Batches"):
        batchX_placeholder = None
        batchY_placeholder = None
        use_both_losses = None
        training = None
    
    def __init__(self, cfg, ops):   
        self.ops = ops
        self.dropout_cntr = 0
        #set batches
        if NNbase.batchX_placeholder is None:
            NNbase.batchX_placeholder = tf.placeholder(cfg[u'datatype'], [None, cfg[u'num_features']], name=u"batchX")
        if NNbase.batchY_placeholder is None:
            NNbase.batchY_placeholder = tf.placeholder(cfg[u'datatype'], [None, cfg[u'num_features']], name=u"batchY")
        if NNbase.use_both_losses is None:
            NNbase.use_both_losses = tf.placeholder(dtype=tf.bool, name=u'use_both_penalty_and_math_loss')
        if NNbase.training is None:            
            NNbase.training = tf.placeholder(dtype=tf.bool, name=u'training_placeholder')
        if u"global_step" not in NNbase.model_vars:
            self.model_vars[u"global_step"] = tf.Variable(0, name=u'global_step', trainable=False, dtype=cfg[u'datatype'])

        #model constants
        with tf.name_scope(u"Constants"):
            self.dummy_matrix = tf.zeros([cfg[u'batch_size'], cfg[u'num_features']], dtype=cfg[u'datatype'], name=u"dummy_constant")            
            self.softmax_sat = tf.constant(np.full((cfg[u'batch_size'], 1), cfg[u'softmax_sat']), dtype=cfg[u'datatype'], name=u"softmax_sat") 

    def select_op(self,current_input,mem_selection, softmax, cfg):
            #######################
            #perform op selection #
            #######################

            #perform all ops in the current timestep intput and save output results together with the op name
        with tf.name_scope(u"Select_op"):
            op_res = []
            for op in self.ops.ops:
                name = op.__name__
                op_outp = op(current_input, mem_selection)
                op_res.append((name, op_outp))

            #slice softmax results for each operation
            ops_softmax = []
            for i, op in enumerate(self.ops.ops):
                name = u"slice_"+op.__name__+u"_softmax_val"
                softmax_slice = tf.slice(softmax, [0,i], [cfg[u'batch_size'],1], name=name)
                ops_softmax.append(softmax_slice)


            #apply softmax on each operation so that operation selection is performed
            ops_final = []
            for i,res in enumerate(op_res):
                name = u"mult_"+res[0]+u"_softmax"
                op_selection =  tf.multiply(res[1], ops_softmax[i], name=name)
                ops_final.append(op_selection)


            #add results from all operation with applied softmax together
            output = tf.add_n(ops_final)
            return output
        
    def select_mem(self, mem_cell, softmax, cfg):
            #######################
            #perform mem selection #
            #######################
        with tf.name_scope(u"Select_mem"):
            #slice softmax results for each mem cell
            mem_softmax = []
            for i, op in enumerate(self.ops.ops):
                name = u"slice_"+op.__name__+u"_softmax_val_mem"
                softmax_slice = tf.slice(softmax, [0,i], [cfg[u'batch_size'],1], name=name)
                mem_softmax.append(softmax_slice)

            #apply softmax on each mem cell so that operation seletion is performed
            mems_final = []
            for i,mem in enumerate(mem_cell):
                name = u"mult_"+op.__name__+u"_softmax_mem"
                mem_selection =  tf.multiply(mem, mem_softmax[i], name=name)
                mems_final.append(mem_selection)

            #add results from all operation with applied softmax together
            output = tf.add_n(mems_final)
            return output

    #cost function
    def calc_loss(self,cfg, output, mode=u"train"):
        
        with tf.name_scope(u"Loss_comp_"+mode):        
            #calc math error
            with tf.name_scope(u"math_error"):
                math_error = tf.multiply(tf.constant(cfg[u'loss_weight'], dtype=cfg[u'datatype']), tf.square(tf.subtract(output , self.batchY_placeholder, name=u"sub_otput_batchY"), name=u"squar_error"), name=u"mult_with_0.5")
            #calc sofmax penalties
            if mode == u"train" and cfg[u"pen_sofmax"]:
                sofmax_op_pen = [self.skewed_sig_dev(smax, num_ops = self.ops.num_of_ops) for smax in self.train[u'softmaxes']]
                sofmax_mem_pen = [self.skewed_sig_dev(smax, num_ops = self.ops.num_of_ops) for smax in self.train[u'softmaxes_mem']]
                sofmax_penalty = tf.reduce_sum(sofmax_op_pen, name=u"red_sofmax_op_pen") + tf.reduce_sum(sofmax_mem_pen, name=u"red_sofmax_mem_pen")
            else :
                sofmax_penalty = 0
            #calc total error
            with tf.name_scope(u"Total_loss_comp"):
                max_error_tot = tf.reduce_sum(math_error, name=u"red_math_loss")
                #make it inversly propotionate to the math error, if math error is big, penelise it less
                #it depends on  error for a batch not for the whole error - hence it should kickoff at a smaller threshhold -
                #num_samples/batch_size
                #num_batches = (cfg["num_samples"]/cfg["batch_size"])*cfg["test_ratio"]
                #sofmax_pen_r = tf.cast((1 - tf.sigmoid( (num_batches*max_error_tot/50) - 10 ) ) * cfg["smax_pen_r"], cfg['datatype'])
                u'''
                sofmax_pen_r = tf.divide( tf.cast(20*cfg["smax_pen_r"], cfg['datatype']) ,
                                          tf.sqrt(max_error_tot + tf.cast(cfg['epsilon'], cfg['datatype'])) ,
                               name = "cal_smax_pen")
                '''
                
                u'''
                total_loss = tf.cond(self.use_both_losses, 
                                     lambda : tf.cast(max_error_tot + sofmax_pen_r*sofmax_penalty, cfg['datatype']),
                                     lambda : tf.cast(sofmax_penalty, cfg['datatype'])
                                    )
                '''
                sofmax_pen_r = cfg[u"smax_pen_r"]
                total_loss = max_error_tot + (sofmax_pen_r*sofmax_penalty)
        return total_loss, max_error_tot

    def calc_backprop(self, cfg):
        print list(self.params.values())
        optimizer = tf.train.AdamOptimizer(cfg[u'learning_rate'], cfg[u'epsilon'] ,name=u"AdamOpt")
        
        with tf.name_scope(u"Grads"):            
            grads = optimizer.compute_gradients(self.total_loss_train, var_list=list(self.params.values()) )

    
                               
            if cfg[u'add_noise']:
                noisy_gradients = []
                for grad, var in grads: 
                    denom = tf.pow( (1+self.model_vars[u"global_step"]), tf.cast(0.55, cfg[u'datatype']))
                    variance =  tf.cast(1/denom, cfg[u'datatype'])
                    #variance =  tf.cast(1, cfg['datatype'])
                    gradient_shape = grad.get_shape()
                    noise = random_ops.truncated_normal(gradient_shape, stddev=tf.sqrt(variance), dtype=cfg[u'datatype'])
                    noisy_gradients.append((grad + noise, var))
                grads = noisy_gradients   
            
            if cfg[u'augument_grad']:
                augumented_grads = []
                #param_names = [tensor.name.replace(":","_") for param, tensor in self.params.items()]
                for grad, var in grads: 
                            u"""
                            if ("W" or "b") and "3" in var.name.replace(":","_"):
                                print("#not augmenting grad for")
                                print(var.name.replace(":","_"))
                                augumented_grads.append((grad, var))
                            else:
                            """
                            print u"#augmenting grad for"
                            print var.name.replace(u":",u"_")                            
                            #aug_ratio = tf.log1p(self.math_error_train)* tf.cast(cfg['softmax_sat'], cfg['datatype'])
                            aug_ratio = tf.cast(cfg[u'softmax_sat'], cfg[u'datatype'])
                            grad_aug = tf.cast(10*aug_ratio*grad, cfg[u'datatype'])
                            augumented_grads.append((grad_aug, var))
            
                        
                grads = augumented_grads
         
                
            #clip gradients by norm and add summaries
            if cfg[u'norm']:
                print u"norming the grads"
                gradients, variables = izip(*grads)
                grads_normed, norms = tf.clip_by_global_norm(gradients, cfg[u'grad_norm'])
                grads = list(izip(grads_normed, variables))
            elif cfg[u'clip']:
                print u"clipping the grads"
                gradients, variables = izip(*grads)
                grads_clipped = [tf.clip_by_value(grad, cfg[u'grad_clip_val_min'], cfg[u'grad_clip_val_max']) for grad in gradients]
                grads = list(izip(grads_clipped, variables))
                norms = []
            else:
                grads = grads
                norms = []
                
        with tf.name_scope(u"Train_step"):
            #train_step = optimizer.apply_gradients(zip(grads, list(self.params.values())), global_step=self.model_vars["global_step"], name="min_loss")
            #train_step = train_step = tf.train.RMSPropOptimizer(cfg['learning_rate'], name="RMSPropOpt").apply_gradients(zip(grads, list(self.params.values())), global_step=self.model_vars["global_step"], name="min_loss")
            train_step = optimizer.apply_gradients(grads, global_step=self.model_vars[u"global_step"], name=u"min_loss")
            
            print u"grads are"
            print grads
            print u"norm is "
            print norms
            return grads, train_step, norms
        
    #funciton for calculating sofmax loss - idea there is to penelise the most if all sofmaxes are the same, that if they are all 1/nump of ops, and have the least penelty for sofmaxes that are 1 or 0
    def skewed_sig_dev(self, x, num_ops = 3, scale = 10):
        with tf.name_scope(u"softmax_pen"):
            worst_case = 1 / num_ops
            shifted_x = x - worst_case + 0.02
            scaled_x = scale*shifted_x
            nom = scale*tf.exp(-scaled_x)+scale*num_ops*tf.exp(-num_ops*scaled_x)
            denom = tf.square(tf.exp(-scaled_x) + 1)*tf.square(tf.exp(-(num_ops-1)*scaled_x) + 1)
            res = nom/denom
            return worst_case*((res)*scale**3)
    u'''
    def custom_softmax(self, x, cfg, base = 180):
        maxx = tf.reduce_max(x, axis=1, keep_dims=True)
        powx = tf.pow(tf.cast(base, cfg['datatype']), x-maxx)
        reduced  = tf.reduce_sum(powx, axis=1, keep_dims=True)
        maxg = powx/ reduced
        return maxg

    '''
    u'''
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
        u"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""        
        if name is None: name = var.name.replace(u":",u"_")
        with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar(u'mean', mean)
                tf.summary.scalar(u'stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
                tf.summary.scalar(u'max', tf.reduce_max(var))
                tf.summary.scalar(u'min', tf.reduce_min(var))
                tf.summary.histogram(u'histogram', var)
                
    ##################
    ####RL methods
    ##################
        #perform selection from the distribution for RL model
    def perform_selection_RL(self, logits, cfg):
            with tf.name_scope(u"perform_selection"):
                #selection = tf.multinomial(logits, 1, name="draw_from_logits")
                selection  = tf.argmax(logits, 1, )
                labels = tf.one_hot(selection, self.num_of_ops, dtype=cfg[u'datatype'])
                reshape_s = tf.reshape(selection , [cfg[u'batch_size'], -1], name = u"reshape_s")
                reshape_l = tf.reshape(labels , [cfg[u'batch_size'], -1], name = u"reshape_l")
                return dict(selection = reshape_s,
                            labels = reshape_l)                 
    
    
    
    #perform policy rollout - select up to five ops max
    def policy_rollout_no_mem(self, sess, _current_state_train, batchX, batchY, cfg, training):
        
        _current_x = batchX
        output = batchX
        
        #produce two types of arrays - ones step based and others - batch based
        rewards = []
        selections = []
        states = []
        current_exes = []
        outputs = []        
 
        for timestep in xrange(cfg[u'max_output_ops']):
            #print("timestep", timestep)
            
                        
            #track states produced by the policy RNN
            states.append(_current_state_train)
            
            #track actual inputs/outputs from the RNN net
            current_exes.append(_current_x)
            
            
            
            _current_state_train,\
            _current_x,\
            _logits,\
            _log_probs,\
            _selection  = sess.run([self.train[u"current_state"],
                                      self.train[u"current_x"],
                                      self.train[u"logits"],
                                      self.train[u"log_probs"],
                                      self.selection],
                            feed_dict={
                                self.init_state:_current_state_train,
                                self.batchX_placeholder: _current_x,
                                self.training: training
                            })
            
            #print(np.hstack([_selection, _logits, _log_probs]))
            
            output, error, math_error = self.ops_env.apply_op(_selection, output, batchY)
            reward = cfg[u'max_reward'] - error
            
            #track rewards
            rewards.append(reward)
            
            #trakc op selection indeces
            selections.append(_selection)           

            #track outputs from the ops
            outputs.append(output)

            
            if math_error.sum() < 1:
                #print("erro sum", math_error.sum())
                #print("breaking")
                break
        
        #print("finished_loops")
        #print("rewards before discounting")
        #print(rewards_ord)

        reeval_rewards = np.apply_along_axis( self.reeval_rewards, 1, np.hstack(rewards)).reshape((cfg[u'batch_size'],-1))
        discount_rewards = np.apply_along_axis( self.discount_rewards, 1, reeval_rewards).reshape((cfg[u'batch_size'],-1))
        #return np.float64(discount_rewards), np.float64(selections), np.float64(states), np.float64(current_exes)
        return  discount_rewards,\
                reeval_rewards,\
                selections,\
                states,\
                current_exes,\
                outputs,\
                math_error

    
    #dicount rewards for later selections https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5#file-pg-pong-py-L130
    def discount_rewards(self, rewards):
        u""" take 1D float array of rewards and compute discounted reward """
    
        #discount the rewards
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        
        #if rewards did not swith, penelise the frst selection first
        #ng = range(0, len(checked_rewards)) if rewards == checked_rewards else reversed(range(0, len(checked_rewards)))
        for t in reversed(xrange(0, len(rewards))):          
            running_add = running_add * 0.09 + rewards[t] #for all pos/negative rewards mean is always going to be bigger than the first reward, hence it will become positive when centered
            discounted_r[t] = running_add
        #normalise rewards
        
        #dont scale but norm, as scaling might result into inversion of signs
        #discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-10)
        normalised_r = discounted_r/ np.linalg.norm(discounted_r, 2)
        #normalised_r = discounted_r
        u'''
        print("discounting")
        print(rewards)
        print("discounted_r")
        print(discounted_r)
        print("np.mean(discounted_r)")
        print(np.mean(discounted_r))
        print("np.std(discounted_r)")
        print(np.std(discounted_r))
        print("np.linalg.norm(discounted_r, 1)")
        print(np.linalg.norm(discounted_r, 1))
        print("np.linalg.norm(discounted_r, 2)")
        print(np.linalg.norm(discounted_r, 2))
        print("normalised_r")
        print(normalised_r)
        '''
        
        return normalised_r

    def reeval_rewards(self,rewards):
        u'''
            Reevaluate cases - idea is: penelise only sequances which are fully negative, otherwise find the postive, make rewards prior positive which led to the pos reward and then make rest zeros, as they are irrelevant after that.
                1.all neg: leave as it is
                [-3000.0, -3000.0, -3000.0, -3000.0, -3000.0]
                [-3000.0, -3000.0, -3000.0, -3000.0, -3000.0]
                2. all pos: Leave first, make rest zeros
                [3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
                [3000.0,    0.0,    0.0,    0.0,    0.0]
                3. Middle switch from neg to pos, make all pos, leave rest zero
                [-3000.0, -3000.0, -3000.0, 3000.0, 3000.0]
                [ 3000.0, 3000.0,  3000.0, 3000.0,    0.0]
                4. Middle switch from pos to neg - leave only the first one, then make rest zeros
                [3000.0, 3000.0, 3000.0, -3000.0, -3000.0]
                [3000.0,    0.0,    0.0,     0.0,     0.0]
                5. First op neg rest pos: - make first and second pos, rest zeros, same as #3
                [-3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
                [ 3000.0, 3000.0,    0.0,    0.0,    0.0]
                6. First op pos rest neg - leave first - make rest zeros, make rest zero, as as #4
                [3000.0, -3000.0, -3000.0, -3000.0, -3000.0]
                [3000.0,     0.0,     0.0,     0.0,     0.0]
                7. last op pos, rest neg - same as #3 - make all pos
                [-3000.0, -3000.0, -3000.0, -3000.0, 3000.0]
                [ 3000.0,  3000.0,  3000.0,  3000.0, 3000.0]
                8. last op neg, rest pos - same as #4         
                [3000.0, 3000.0, 3000.0, 3000.0, -3000.0]
                [3000.0,    0.0,    0.0,    0.0,     0.0]
        ''' 
        rewards = list(rewards)
        last_n_slice = len(rewards)
        for t in xrange(last_n_slice):
            #find the first positive reward
            if rewards[t] > 0:
                first = []
                second = []
                #slicing - [a:b] a is inclusive, b is exclusive boundary
                #make all before postive
                try:
                    first = list(imap(abs, rewards[0:t+1]))
                except IndexError:
                    pass
                #make all rest zeros
                try:
                    second = list(np.zeros_like(rewards[t+1:last_n_slice]))
                except IndexError:
                    pass            
                reeval_rewards = first + second
                break
            reeval_rewards = rewards
        return reeval_rewards
    
        
    #calculate loss function based on selected policies and the achieved rewards
    def calc_RL_loss(self, log_prob, cfg):
        indices = tf.cast(tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1], cfg[u'datatype']) + self.selections_placeholder
        op_prob = tf.gather(tf.reshape(log_prob, [-1]), tf.cast(indices, tf.int64))

        #comp loss
        loss = -tf.reduce_sum(tf.multiply(op_prob, self.rewards_placeholder))
        #loss = tf.nn.l2_loss(-op_prob)
        
        return loss
    
    def calc_backprop_RL(self, loss, cfg):
        optimizer = tf.train.RMSPropOptimizer(cfg[u'learning_rate'])
        #grads = optimizer.compute_gradients(loss, var_list= list(self.params.values()), grad_loss=self.rewards_placeholder)
        grads = optimizer.compute_gradients(loss, var_list= list(self.params.values()))
        #clear none grads
        clean_grads = [(grad, var) for grad, var in grads if grad is not None]
        grads = clean_grads
        if cfg[u'add_noise']:
            noisy_gradients = []
            for grad in grads:
                denom = tf.pow( (1+self.model_vars[u"global_step"]), tf.cast(0.55, cfg[u'datatype']))
                variance =  tf.cast(1/denom, cfg[u'datatype'])
                gradient_shape = grad[0].get_shape()
                noise = random_ops.truncated_normal(gradient_shape, stddev=tf.sqrt(variance), dtype=cfg[u'datatype'])
                noisy_gradients.append((grad[0] + noise, grad[1]))
            grads = noisy_gradients
            
        if cfg[u'norm']:
            print u"norming the grads"
            gradients, variables = izip(*grads)
            grads_normed, norms = tf.clip_by_global_norm(gradients, cfg[u'grad_norm'])
            grads = list(izip(grads_normed, variables))
        else:
            norms = []
        train_step = optimizer.apply_gradients(grads, global_step=self.model_vars[u"global_step"])
        print u"grads are"
        print grads
        return grads, train_step, norms