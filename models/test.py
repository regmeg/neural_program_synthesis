from params import get_cfg
import data_gen

import tensorflow as tf

FLAGS = tf.flags.FLAGS

fn = eval("data_gen."+FLAGS.train_fn)

print(fn.__name__)
