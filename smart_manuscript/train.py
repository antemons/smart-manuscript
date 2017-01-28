#!/usr/bin/python3

"""
    This file is part of Smart Manuscript.

    Smart Manuscript (transcript handwritten notes or inputs)
    Copyright (c) 2017 Daniel Vorberg

    Smart Manuscript is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Smart Manuscript is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Smart Manuscript.  If not, see <http://www.gnu.org/licenses/>.
"""

import pickle
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
import json
from reader import GraphUtilities, labels_to_transcription, \
                   transcription_to_labels
from stroke_features import InkFeatures
from utils import Bunch
from train_utils import Corpus, Corpora, get_trainings_batch_creator, \
                        get_test_batches, load_corpora

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean(
    'restore', False,
    "Whether to restore check point (defined in restore_step)")
tf.app.flags.DEFINE_integer(
    "restore_step", -1, "Which step is restored (-1 means last possible one)")
tf.app.flags.DEFINE_boolean(
    'only_numbers', False, "Whether to learn numbers only")
tf.app.flags.DEFINE_integer(
    'max_steps', 20000, "Number of batches to run")
tf.app.flags.DEFINE_float(
    'learning_rate', 0.003, "Learning rate for Optimizer")
tf.app.flags.DEFINE_float(
    'learning_rate_fine', 0.0006, "Learning rate for fine-tuning")
tf.app.flags.DEFINE_integer(
    'steps_fine_tuning', 10000,
    "Number of batches to run before learning_rate_fine is used")
tf.app.flags.DEFINE_boolean(
    'batch_size', 50, "Size of the training batch")
tf.app.flags.DEFINE_integer(
    'test_batch_size', 500,
    "Batch size of the test batch(es)")
tf.app.flags.DEFINE_integer(
    'test_batch_size_lines', 10,
    "Batch size of the test batch(es)")
tf.app.flags.DEFINE_string(
    'name', "vanilla", "Name of the network")
tf.app.flags.DEFINE_integer(
    'build_data', 2, "Whether to build new (preprocessed) data")
tf.app.flags.DEFINE_boolean(
    'train_textlines', True,  "Whether to use full textline to train")
tf.app.flags.DEFINE_boolean(
    'train_my_writing', False,
    "Whether to use your handwritten notes to train")
tf.app.flags.DEFINE_boolean(
    'test_the_zen_of_python', True,
    "Whether test also on the Zen of Python")
tf.app.flags.DEFINE_string(
    'lstm_sizes', "[120, 120]",  "List of LSTM-layer sizes (json)")
tf.app.flags.DEFINE_integer(
    'num_examples', 4,
    "Number of examples shown for each test during training")
tf.app.flags.DEFINE_boolean(
    'shares_param_first_layer',
    True,
    "Whether the first LSTM-layer shares parameters"
    "in forward and backward layer")

ALPHABET = list("abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "1234567890 ,.:;*+()/!?+-'\"$")


class GraphTraining(GraphUtilities):
    """ Train the graph model """

    def __init__(self, alphabet):
        """
        Args:
            alphabet (list of str): all valid letters/ symbols
        """
        num_classes = len(alphabet) + 1
        self.alphabet = alphabet
        self.path = "graphs/%s/" % FLAGS.name
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.build_graph(
            InkFeatures.NUM_FEATURES, num_classes,
            json.loads(FLAGS.lstm_sizes))

    @staticmethod
    def _build_network(inputs, input_seq_length, lstm_sizes,
                       num_classes, ret_output_states=False):
        """ build the graph to make the predictions

        Args:
            inputs (tf.placeholder(tf.float32, shape=3 * [None]):
                the input to the graph
            lstm_sizes (list of int): size of the lstm in each layer, the
                length is the number of layers
            num_classes (int): number of target classes

        Returns:
            logits
        """
        lstm_layer_input = inputs
        for n_layer, num_hidden_neurons in enumerate(lstm_sizes):
            # TODO: define new OpenLSTMCell(LSTMCell)
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(
                num_hidden_neurons, state_is_tuple=True)
            if n_layer != 0 or not FLAGS.shares_param_first_layer:
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(
                    num_hidden_neurons, state_is_tuple=True)
            else:
                lstm_cell_bw = lstm_cell_fw
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw,
                inputs=lstm_layer_input, dtype=tf.float32,
                scope='BLSTM_' + str(n_layer + 1),
                sequence_length=input_seq_length)
            output_fw = outputs[0]
            output_bw = outputs[1]
            lstm_layer_output = tf.concat(2, [output_fw, output_bw])
            lstm_layer_input = lstm_layer_output
        W = tf.Variable(
            tf.truncated_normal([2 * num_hidden_neurons, num_classes],
                                stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        lstm_layer_output = tf.reshape(
            lstm_layer_output, [-1, 2 * num_hidden_neurons])
        # TODO(daniel): apply sigmoid?
        logits = tf.matmul(lstm_layer_output, W) + b
        batch_size = tf.shape(inputs)[0]
        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        logits = tf.transpose(logits, [1, 0, 2], name="logits")
        return logits if not ret_output_states else logits, output_states

    @staticmethod
    def _build_optimizer(logits, logits_lengths, targets, learning_rate):
        """ build the optimizer for the graph

        Args:
            logits, logits_lengths: the logits
            target:
            learning_rate (float or tf.placeholder()):

        Return:
            a function making one optimizing step
        """
        loss = tf.reduce_mean(
            ctc.ctc_loss(
                logits, targets, logits_lengths, ctc_merge_repeated=False))
        tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
        return train_step

    def build_graph(self, num_features, num_classes, lstm_sizes):
        """ build the graph

        Args:
            num_features (int): number of features in each time-step
            num_classes (int): number of target classes
            lstm_sizes (list of int): size of the lstm in each layer, the
                length is the number of layers
        """

        print("create network", flush=True)

        with tf.device("/gpu:0"):
            batch_size = max_input_seq_len = None
            print(" {:20}".format("placeholder", flush=True, end="\r"))
            self.learning_rate = tf.placeholder(
                tf.float32, name="learning_rate")
            self.input = tf.placeholder(
                tf.float32, name="input",
                shape=[batch_size, max_input_seq_len, num_features])
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=(batch_size), name="input_seq_length")
            self.target_indices = tf.placeholder(
                tf.int64, name="target_indices")
            self.target_values = tf.placeholder(tf.int32, name="target_values")
            self.target_shape = tf.placeholder(tf.int64, name="target_shape")
            self.target = tf.SparseTensor(
                self.target_indices, self.target_values, self.target_shape)

            print(" {:20}".format("create lstm cells"), flush=True, end="\r")
            logits, output_states = self._build_network(
                self.input, self.input_seq_length, lstm_sizes, num_classes)

            print(" {:20}".format("create predictions"), flush=True, end="\r")
            self.predictions = ctc.ctc_beam_search_decoder(
                logits, self.input_seq_length, top_paths=1,
                merge_repeated=False)[0][0]

            print(" {:20}".format("create optimizer"), flush=True, end="\r")
            self.train_step = self._build_optimizer(
                logits, self.input_seq_length, self.target, self.learning_rate)

            print(" {:20}".format("create initializer"), flush=True, end="\r")
            self.init = tf.global_variables_initializer()

            # self.error_rate = tf.reduce_sum(
            #    tf.edit_distance(tf.to_int32(self.predictions),
            # target, normalize=False)) / tf.to_float(tf.size(target.values))

            print(" {:20}".format("create error measure"),
                  flush=True, end="\r")
            self.error_rate = tf.reduce_mean(
                tf.edit_distance(tf.to_int32(self.predictions),
                                 self.target))
            tf.summary.scalar('error_rate', self.error_rate)

            print(" {:20}".format("create saver"), flush=True, end="\r")
            tf.add_to_collection('output_states', output_states[0].c)
            self.saver = tf.train.Saver()
            self.saver.export_meta_graph(self.path + 'model.meta')
            pickle.dump(
                self.alphabet, open(self.path + "alphabet.pkl", "wb"), -1)

            print(" {:20}".format("graph has been created and saved"))

    def train(self, batch_train, batch_tests):
        """ train the network

        Args:
            batch_train (func returning list of tuples (features, labels)):
                the corpus to train
            batch_test (dict of list of tuples (features, labels)):
                on each item a test of the accuracy is performed
        """
        print("train network", flush=True)
        merged = tf.summary.merge_all()

        tests = {}
        for test_name, batch in batch_tests.items():
            tests[test_name] = Bunch(
                predictions=[
                    prediction.values for prediction
                    in tf.sparse_split(0, len(batch), self.predictions)],
                targets=[target.values for target
                         in tf.sparse_split(0, len(batch), self.target)],
                feed_dict=self.convert_to_tf(*list(zip(*batch))))

        config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as session:
            writer_train = tf.summary.FileWriter(
                self.path + "train", session.graph)

            writer_tests = {}
            for test_name in batch_tests:
                writer_tests[test_name] = tf.summary.FileWriter(
                    self.path + test_name, session.graph)

            session.run(self.init)
            if FLAGS.restore:
                begin_step = FLAGS.restore_step
                assert begin_step >= 0, "please specify --restore_step"
                self.saver.restore(
                    session, self.path + 'model_{}.ckpt'.format(begin_step))
            else:
                begin_step = 0
            print(" start training", flush=True)
            for step in range(begin_step, FLAGS.max_steps + 1):
                print("step {}%".format(step), flush=True, end="\r")
                feed_dict = {
                    **self.convert_to_tf((*list(zip(*batch_train(step))))),
                    self.learning_rate:
                        (FLAGS.learning_rate
                         if step < FLAGS.steps_fine_tuning
                         else FLAGS.learning_rate_fine)}

                if step % 25 == 0:
                    # train
                    summary = session.run(merged, feed_dict=feed_dict)
                    writer_train.add_summary(summary, step)

                    # test
                    for test_name, test in tests.items():
                        summary, error_test, predictions, targets\
                            = session.run([
                                merged, self.error_rate,
                                test.predictions, test.targets],
                                feed_dict=test.feed_dict)

                        writer_tests[test_name].add_summary(summary, step)
                        print("Test ({}): {:2f} (step: {})".format(
                            test_name, 100*error_test, step))
                        for i in range(FLAGS.num_examples):
                            print("  {} -> {}".format(
                                labels_to_transcription(targets[i],
                                                        self.alphabet),
                                labels_to_transcription(predictions[i],
                                                        self.alphabet)))
                if ((step % 250 == 0 or (step + 1) == FLAGS.max_steps) and
                        step != begin_step):
                    self.saver.save(
                        session, self.path + "model_%i.ckpt" % step)
                    print("variables have been saved")

                error_train, _ = session.run(
                    [self.error_rate, self.train_step], feed_dict=feed_dict)


def _get_sets_and_alphabet(alphabet=ALPHABET):

    if FLAGS.build_data <= 0:
        corpora = load_corpora()
        pickle.dump(corpora, open("tmp/corpora.pkl", "wb"), -1)

    if FLAGS.build_data <= 1:
        corpora = pickle.load(open("tmp/corpora.pkl", "rb"))
        print("extract features from several corpora,"
              "this may take a few minutes")
        sets = corpora.convert_to_features_and_labels(alphabet)
        pickle.dump(sets, open("tmp/sets.pkl", "wb"), -1)
        pickle.dump(alphabet, open("tmp/alphabet.pkl", "wb"), -1)
    else:
        sets = pickle.load(open("tmp/sets.pkl", "rb"))

    return sets, alphabet


def main():
    data, alphabet = _get_sets_and_alphabet()
    net = GraphTraining(alphabet=alphabet)
    net.train(batch_train=get_trainings_batch_creator(data),
              batch_tests=get_test_batches(data))

if __name__ == "__main__":
    main()
