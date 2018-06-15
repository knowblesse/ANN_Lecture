import os
import struct
import numpy as np
import tensorflow as tf
import sys
import gzip
import matplotlib.pyplot as plt
def myfunc():
    with tf.variable_scope('myscope'):
        myvar1 = tf.get_variable("tf_myvar1",[2,3], initializer=tf.zeros_initializer)
        return myvar1

temp = myfunc()

with tf.variable_scope('myscope',reuse=True):
    myvar2 = tf.get_variable('tf_myvar1')

assert temp == myvar2