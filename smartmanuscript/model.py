#!/usr/bin/env python3

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

import numpy as np
import os.path
import itertools
import glob
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.platform.app import flags
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from .utils import cached_property, colored_str_comparison
from . import stroke_features
NUM_FEATURES = stroke_features.InkFeatures.NUM_FEATURES

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

Sequence = namedtuple('Sequence', ('values, length'))

class InferenceModel:
    """ A recurrent neural network infers labels for a given sequence.

    """
    BATCH_SIZE = None
    MAX_INPUT_LEN = None
    NUM_OF_PROPOSALS = 3
    TYPE_NAME = "inference"

    def __init__(self, lstm_sizes,
                 share_param_first_layer, alphabet):


        self.graph = tf.get_default_graph()
        with tf.variable_scope("inputs"):
            self.iterator = self._get_iterator()
            tf.add_to_collection("iterator", self.iterator)
            features, features_length, self.targets = self.iterator.get_next()  # ToDo(dv): target is not input
            self.input = Sequence(
                values=tf.identity(features, "features"),
                length=tf.identity(features_length, "length"))
        self.alphabet =  tf.constant(list(alphabet))
        self.logits = self._forward_pass(*self.input, lstm_sizes, self.NUM_CLASSES, share_param_first_layer)
        self.tokens, self.log_prob = self._get_labels(*self.logits, self.NUM_OF_PROPOSALS)
        labels = self._decode(self.tokens, self.alphabet)
        with tf.variable_scope("output"):
            self.labels = tf.identity(labels, "labels")
            self.probabilities = tf.exp(self.log_prob, "probabilities")
        self._most_likely_tokens = self.tokens[0]
        self._saver = self._get_saver()

    def infer(self, features, ckpt_path, tensors=None):
        feed_dict = {
            self.input.values: np.expand_dims(features, 0),
            self.input.length: np.array([len(features)])}

        tensors = tensors or self.labels
        with tf.Session(graph=self.graph) as sess:
            self._saver.restore(sess, ckpt_path)
            evaled_tensors = sess.run(tensors, feed_dict=feed_dict)
        return evaled_tensors

    @classmethod
    def _get_iterator(cls):
        iterator = tf.data.Iterator.from_structure(
            output_types=(tf.float32, tf.int32, tf.string),
            output_shapes=(tf.TensorShape([None, None, NUM_FEATURES]),
                           tf.TensorShape([None]),
                           tf.TensorShape([None])))
        return iterator

    @staticmethod
    def _decode(tokens, alphabet):
        with tf.variable_scope("decode"):
            n_th_labels = []
            for n_th_prediction in tokens:
                values = tf.gather(alphabet, n_th_prediction.values)
                labels_sparse = tf.SparseTensor(
                    indices=n_th_prediction.indices,
                    values=values,
                    dense_shape=n_th_prediction.dense_shape)
                labels_dense = tf.sparse_tensor_to_dense(labels_sparse, "")
                labels = tf.reduce_join(labels_dense, axis=-1)
                n_th_labels.append(labels)
        return n_th_labels

    @staticmethod
    def _get_labels(logits, lengths, num_proposals):
        with tf.variable_scope("beam_search"):
            labels, log_prob = tf.nn.ctc_beam_search_decoder(
                logits, lengths,
                top_paths=num_proposals,
                merge_repeated=False)
        return labels, log_prob

    @property
    def NUM_CLASSES(self):
        return int(self.alphabet.shape[0]) + 1

    def __str__(self):
        return "CTC-BLSTM"

    @staticmethod
    def _forward_pass(inputs, lengths, lstm_sizes, num_classes, share_param_first_layer):
        with tf.variable_scope("forward_pass"):
            lstm_layer_input = inputs
            for n_layer, num_hidden_neurons in enumerate(lstm_sizes):
                # TODO(dv): add dropout?
                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(
                    num_hidden_neurons, state_is_tuple=True)
                if n_layer != 0 or not share_param_first_layer:
                    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(
                        num_hidden_neurons, state_is_tuple=True)
                else:
                    lstm_cell_bw = lstm_cell_fw
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell_fw, lstm_cell_bw,
                    inputs=lstm_layer_input, dtype=tf.float32,
                    scope='BLSTM_' + str(n_layer + 1),
                    sequence_length=lengths)
                lstm_layer_output = tf.concat(outputs, 2)
                lstm_layer_input = lstm_layer_output
            output_values = tf.layers.dense(lstm_layer_output, num_classes)
            output_values = tf.transpose(output_values, [1, 0, 2], name="output")
            output_lengths = tf.identity(lengths, name="lengths")
        with tf.variable_scope("logits"):
            logit_values = tf.identity(output_values, name="values")
            logit_lengths = tf.identity(output_lengths, name="lengths")
        return Sequence(logit_values, logit_lengths)

    def save_meta(self, path):
        self._saver.export_meta_graph(path)
        #print("graph and encoder have been saved")

    def _get_saver(self):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list)
        return saver



class EvaluationModel(InferenceModel):
    TYPE_NAME = "evaluation"
    SUMMARY_SCALARS = ["loss", "error"]

    def __init__(self,
                 lstm_sizes,
                 share_param_first_layer, alphabet):

        super().__init__(lstm_sizes, share_param_first_layer, alphabet)

        self.target_tokens = self._encode(self.targets, self.alphabet)
        self.error = self._get_error(self._most_likely_tokens, self.target_tokens)
        self.loss = self._get_loss(self.target_tokens, *self.logits)



    def _get_summary_and_writer(self, log_path):
        with tf.name_scope('summaries'):
            for name in self.SUMMARY_SCALARS:
                tf.summary.scalar(name, self.__getattribute__(name))
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(log_path, self.TYPE_NAME),
            flush_secs=30)
        return summary, summary_writer

    @staticmethod
    def _encode(labels, character_lookup):
        with tf.variable_scope("encode"):
            labels_splitted = tf.string_split(labels, delimiter='')
            table = tf.contrib.lookup.index_table_from_tensor(
                mapping=character_lookup)
            tokens = table.lookup(labels_splitted)
        return tokens

    def get_evaluation(self, dataset, log_path=None):
        with self.graph.as_default():
            feed_iterator_from_dataset = self.iterator.make_initializer(dataset)
            if log_path is not None:
                summary, summary_writer = self._get_summary_and_writer(log_path)
            else:
                summary, summary_writer = tf.no_op(), None
            def evaluation(model_path, save_with_global_step=None):
                with tf.Session(graph=self.graph) as sess:
                    self._saver.restore(sess, model_path)
                    sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
                    sess.run(feed_iterator_from_dataset)
                    tensors = self.error, self.labels, self.loss, summary
                    *evaled_tensors, evaled_summary = sess.run(tensors)
                    if save_with_global_step is not None:
                        summary_writer.add_summary(evaled_summary,
                                                   save_with_global_step)
                return evaled_tensors
        return evaluation

    @staticmethod
    def _get_loss(targets, logits, lengths):
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.ctc_loss(
                tf.to_int32(targets), logits, lengths,
                ctc_merge_repeated=False))
            tf.summary.scalar('loss', loss)
        return loss

    @staticmethod
    def _get_error(predictions, targets):
        with tf.variable_scope("error"):
            error = tf.reduce_mean(tf.edit_distance(predictions, targets))
        tf.summary.scalar('error', error)
        return error

class TrainingModel(EvaluationModel):
    TYPE_NAME = "training"
    SUMMARY_SCALARS = ["loss", "error", "global_step", "global_step", "epoch"]
    DEFAULT_ALPHABET = list("abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "1234567890 ,.:;*+()/!?&-'\"$")

    def __init__(self, lstm_sizes=[128, 128],
                 share_param_first_layer=True):

        super().__init__(lstm_sizes, share_param_first_layer,
                         alphabet=self.DEFAULT_ALPHABET)
        self._lstm_sizes = lstm_sizes
        self._share_param_first_layer = share_param_first_layer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.epoch = tf.Variable(0, name='epoch_num', trainable=False)
        self.increment_epoch = tf.assign_add(self.epoch, 1)
        self.learning_rate = tf.Variable(0.003, name='learning_rate', trainable=False)
        self.train_op = self._get_train_op(self.loss, self.global_step, self.learning_rate)
        self.initializer = self._get_initializer()
        self.summary = tf.summary.merge_all()

    @staticmethod
    def _get_train_op(loss, global_step, learning_rate):
        """ return function to perform a single training step
        """
        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss,
                                          global_step=global_step)
        return train_op

    @staticmethod
    def _get_initializer():
        return [tf.global_variables_initializer(),
                tf.local_variables_initializer()]

    def save_checkpoint(self, sess, global_step=None, path=None):
        path = path or self.path
        name = "model.ckpt"
        path = os.path.join(self.path, name)
        self._saver.save(
            sess, path, global_step=global_step)
        print("variables have been saved")

    def _get_saver(self):
        var_list = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) +
                    tf.get_collection('alphabet'))
        saver = tf.train.Saver(
            var_list,
            keep_checkpoint_every_n_hours=0.5)
        return saver

    def train(self, dataset, path, steps_per_checkpoint=10, epoch_num=1):
        with tf.Graph().as_default():
            model = InferenceModel(self._lstm_sizes,
                                   self._share_param_first_layer,
                                   self.DEFAULT_ALPHABET)
            model.save_meta(os.path.join(path, "inference.meta"))
        with tf.Graph().as_default():
            model = EvaluationModel(self._lstm_sizes,
                                    self._share_param_first_layer,
                                    self.DEFAULT_ALPHABET)
            model.save_meta(os.path.join(path, "evaluation.meta"))
        feed_iterator_from_dataset = self.iterator.make_initializer(dataset)
        summary, summary_writer = self._get_summary_and_writer(
            os.path.join(path, "log"))
        with tf.Session(graph=self.graph) as sess:
            #self._saver.restore(sess, model_path)
            sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
            sess.run(self.initializer)
            sess.run(feed_iterator_from_dataset)
            while sess.run(self.epoch) < epoch_num:
                for step_in_epoch in itertools.count():
                    if step_in_epoch % steps_per_checkpoint == 0:
                        global_step = sess.run(self.global_step)
                        checkpoints_path = self._saver.save(
                            sess, os.path.join(path, "model"),
                            global_step=global_step)
                        #if eval_func is not None:
                        #    eval_func(checkpoints_path, global_step)
                        #    print(checkpoints_path, global_step)
                    try:
                        #_, global_step, summary, loss =
                        print("asd")
                        _, global_step, evaled_summary, loss = sess.run(
                                [self.train_op, self.global_step, summary, self.loss])
                        summary_writer.add_summary(evaled_summary, global_step)
                        if step_in_epoch % steps_per_checkpoint == 0:
                            print(sess.run(self.epoch), step_in_epoch, loss)
                    except tf.errors.OutOfRangeError:
                        sess.run(self.increment_epoch)
                        print("epoch completed")
                        break



NeuralNetworks = TrainingModel
#
# class Training:
#     Evaluation = namedtuple("Evaluation", ["batch", "writer"])
#     EVALUATION_INTERVAL = 25
#     SAVE_INTERVAL = 250
#     EVALUATION_BATCH_SIZE = 300
#     DEFAULT_BATCH_SIZE = 50
#
#     IBM_TEST_SET = ["GPIquery_01.tfrecords", "MVLquery_50.tfrecords"]
#
#     @property
#     def IAM_TEST_SET(self):
#         return [line.rstrip("\n").replace(".inkml" , "")
#                         for line in open("data/IAMonDo-db-1.0/0.set", "r")]
#
#     def __init__(self, data_path="tmp"):
#         self._data_path = data_path
#
#     def test_records(self):
#
#         result =  dict(
#             daniel=glob.glob(self._data_path + "/my_test/*.tfrecords"),
#             python_zen=glob.glob(self._data_path + "/zen_test/*.tfrecords"),
#             iam_word=[filename
#                 for filename in glob.glob(self._data_path + "/iam_word/*.tfrecords")
#                 if any(k in filename for k in self.IAM_TEST_SET)],
#             iam_line=[filename
#                 for filename in glob.glob(self._data_path + "/iam_line/*.tfrecords")
#                 if any(k in filename for k in self.IAM_TEST_SET)],
#             ibm=[filename
#                  for filename in glob.glob(self._data_path + "/ibm/*.tfrecords")
#                  if any(k in filename for k in self.IBM_TEST_SET)])
#         result = {k: v for k, v in result.items() if v}
#         return result
#
#     @classmethod
#     def train_records(cls, data_path):
#
#         result =  (10 * glob.glob(data_path + "/my_train/*.tfrecords") +
#                 [filename
#                  for filename in glob.glob(data_path + "/iam_word/*.tfrecords")
#                  if not any(k in filename for k in cls.IAM_TEST_SET)] +
#                 2 * [filename
#                  for filename in glob.glob(data_path + "/iam_line/*.tfrecords")
#                  if not any(k in filename for k in cls.IAM_TEST_SET)] +
#                 [filename
#                  for filename in glob.glob(data_path + "/ibm/*.tfrecords")
#                  if not any(k in filename for k in cls.IBM_TEST_SET)])
#         return result
#
#     @classmethod
#     def train_batch(cls, data_path):
#         return record_to_batch(
#             filenames=cls.train_records(data_path),
#             batch_size=cls.DEFAULT_BATCH_SIZE)
#
#     @cached_property
#     def batch_size(self):
#         return tf.placeholder_with_default(
#             tf.constant(self.DEFAULT_BATCH_SIZE, tf.int32), [])
#
#     def evaluation(self, filenames, name, net, graph=None):
#         writer = tf.summary.FileWriter(
#             os.path.join(net.path, name), graph)
#         with tf.Graph().as_default():
#             batch = record_to_batch(filenames,
#                                     batch_size=self.EVALUATION_BATCH_SIZE,
#                                     allow_smaller_final_batch=True,
#                                     shuffle=True, num_epochs=1)
#             coord = coordinator.Coordinator()
#             with tf.Session() as sess:
#                 sess.run([tf.global_variables_initializer(),
#                           tf.local_variables_initializer()])
#                 # see: https://github.com/tensorflow/tensorflow/issues/1045
#                 threads = queue_runner.start_queue_runners(sess, coord)
#                 evaled_batch = sess.run(batch)
#                 coord.request_stop()
#                 coord.join(threads)
#         print("loaded evaluation from {} with {} elements"\
#             .format(filenames, evaled_batch.target.dense_shape[0]))
#         return self.Evaluation(batch=evaled_batch, writer=writer)
#
#
#     def evaluate(self, evaluation, sess, net, name, num_examples=1):
#         feed_dict = {
#             net.input: evaluation.batch.input,
#             net.target: evaluation.batch.target}
#
#         summary, error, prediction, target, step = sess.run(
#             [net.summary, net.error, net.prediction,
#              net.target, net.global_step], feed_dict=feed_dict)
#
#         evaluation.writer.add_summary(summary, step)
#         print("Evaluation ({}): {:2.0f}%".format(name, 100*error))
#         for predicted_transcription, target_transcription \
#             in zip(net.decode(prediction)[:num_examples], net.decode(target)):
#             if len(predicted_transcription) > 1.5 * len(target_transcription):
#                 predicted_transcription = "..."
#             print(colored_str_comparison(target_transcription,
#                                          predicted_transcription))
#         return error
#
#     @staticmethod
#     def create_learning_profile(
#         learning_rate_default, total_num_steps,
#         final_learning_rate, num_final_steps):
#
#         def learning_rate(step):
#             if step < total_num_steps - num_final_steps:
#                 return learning_rate_default
#             else:
#                 return final_learning_rate
#         return learning_rate
#
#
#     def train(self, net, num_steps, learning_rate_default, final_learning_rate = None,
#               num_final_steps=0, restore_from=None):
#         """ train the network
#
#         Args:
#             batch_test (dict of list of tuples (features, labels)):
#                 on each item a test of the accuracy is performed
#         """
#         print("train network", flush=True)
#
#         evaluations = {
#             name: self.evaluation(filename, name, net, tf.get_default_graph())
#             for name, filename in self.test_records().items()}
#
#         writer_train = tf.summary.FileWriter(
#             os.path.join(net.path, "train"), tf.get_default_graph())
#
#         config = tf.ConfigProto(allow_soft_placement=True,
#                                 log_device_placement=False)
#
#         learning_rate = self.create_learning_profile(
#             learning_rate_default, num_steps, final_learning_rate, num_final_steps)
#         with tf.Session(config=config) as sess:
#             sess.run(net.initializer)
#             tf.train.start_queue_runners(sess=sess)
#             if restore_from is not None and restore_from != "":
#                 print ("restore model from: {}".format(restore_from))
#                 net._saver.restore(sess, restore_from)
#
#             while True:
#                 step = sess.run(net.global_step)
#                 if step % 10 == 0:
#                     print("step: {:6}".format(step), flush=True)
#
#
#                 if step % self.EVALUATION_INTERVAL == 0:
#                     for key, evaluation in evaluations.items():
#                         self.evaluate(evaluation, sess, net, key)
#
#                 if (step % self.SAVE_INTERVAL == 0 or step == num_steps):
#                     net.save_checkpoint(sess, step)
#
#
#                 _, target, summary = sess.run(
#                     [net.train_step, net.target, net.summary],
#                     feed_dict={net.learning_rate: learning_rate(step)})
#                 writer_train.add_summary(summary, step)
#                 #decoded_target = net.decode(target)
#                 #print(decoded_target)
#
#                 if step == num_steps:
#                     print("training has been finished")
#                     break
