from __future__ import print_function

import os
import tensorflow as tf
from train import Graph
from tensorflow.python.framework import graph_util
import argparse

FLAGS = {}

def lite_freeze(g, sess, output_file):
    flatbuf = tf.contrib.lite.toco_convert(
            input_data=sess.graph_def,
            input_tensors=[g.L, g.mels, g.prev_max_attentions]
            output_tensors=[g.O]
        )

    with open(output_file, 'w') as f:
        f.write(flatbuf)
        f.close()

    return True

def normal_freeze(sess, output_file):
    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['FINAL/output'])
    tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(output_file),
      os.path.basename(output_file),
      as_text=False)
    tf.logging.info('Saved frozen graph to %s', output_file)

    return True


def main(args, **kw):
    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        if FLAGS.type == 'lite':
            lite_freeze(g, sess, 'frozen.tflite')
        else:
            normal_freeze(sess, 'frozen.pb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='lite',
        help='metrics',)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)