from __future__ import print_function

import os
import sys

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
# from small import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
import argparse

FLAGS = {}

def _handle_ops_metrics(sess):
    ops = sess.graph.get_operations()

    op_name_list = [o.name for o in ops]

    for op_name in op_name_list:
        print(op_name)

    print("Total ops count: {}".format(len(op_name_list)))

def _handle_params_metrics(sess):
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


def _save_pb(sess):
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open('logdir/graphpb.txt', 'w') as f: 
        f.write(graphpb_txt)

    writer = tf.summary.FileWriter(logdir='logdir', graph=sess.graph_def)
    writer.flush()


def main(args, **kw):
    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 

        if FLAGS.save_pb:
            _save_pb(sess)

        if FLAGS.metrics == 'params':
            _handle_params_metrics(sess)

        elif FLAGS.metrics == 'ops':
            _handle_ops_metrics(sess)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='metrics',)

    parser.add_argument(
        '--save_pb',
        type=str,
        default=False,
        help='save proto pb',)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)