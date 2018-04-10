from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

def model_info():
    # Load data
    # L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open('logdir/graphpb.txt', 'w') as f: 
        f.write(graphpb_txt)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # graph_def = tf.get_default_graph().as_graph_def()
        writer = tf.summary.FileWriter(logdir='logdir', graph=sess.graph_def)
        writer.flush()

        # Restore parameters
        parameters_numbers = 0
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 

        for var in var_list:
            shape = var.get_shape()
            p_num = 1
            for dim in shape:
                p_num = p_num * dim.value

            parameters_numbers += p_num
            print("{0}: shape: {1}".format(var.name, var.get_shape()))
            print("\t\t\t\t\t\t\t\t\t parameter num: {}".format(p_num))

        print("----------------------------")
        print("total parameter numbers: {}".format(parameters_numbers))

if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    model_info()