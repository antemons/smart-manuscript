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
from functools import reduce
import os.path
import itertools
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.platform.app import flags
from tensorflow.python.client import timeline
from tensor2tensor.utils.data_reader import bucket_by_sequence_length
from .utils import cached_property, colored_str_comparison
from . import writing

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

NUM_FEATURES = writing.InkFeatures.NUM_FEATURES
Sequence = namedtuple('Sequence', ('values, length'))

def parse_tfrecords(proto):
    sequence_features = {
        'input': tf.FixedLenSequenceFeature([NUM_FEATURES], tf.float32)}
    context_features = {'label': tf.FixedLenFeature([], tf.string)}
    context, sequence = tf.parse_single_sequence_example(
        proto,
        sequence_features=sequence_features,
        context_features=context_features)
    label = context['label']
    inputs = sequence['input']
    return inputs, label


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
            #tf.add_to_collection("iterator", self.iterator)
            features, features_length, self.targets = self.iterator.get_next()  # ToDo(dv): target is not input
            self.input = Sequence(
                values=tf.identity(features, "features"),
                length=tf.identity(features_length, "length"))
        self.alphabet =  tf.constant(list(alphabet))
        self.logits = self._forward_pass(
            *self.input, lstm_sizes,
            self.NUM_CLASSES, share_param_first_layer)
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
            lstm_layer_input = tf.transpose(inputs, [1, 0, 2], name="inputs")
            for n_layer, num_hidden_neurons in enumerate(lstm_sizes):
                with tf.variable_scope("BLSTM_layer_" + str(n_layer+1)):
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
                        sequence_length=lengths,
                        time_major=True)
                    lstm_layer_output = tf.concat(outputs, 2)
                    lstm_layer_input = lstm_layer_output
                    lengths = tf.identity(lengths, "length")
            with tf.variable_scope("dense_layer"):
                output_values = tf.layers.dense(lstm_layer_output, num_classes)
                output_lengths = tf.identity(lengths, name="lengths")
        with tf.variable_scope("logits"):
            logit_values = tf.identity(output_values, name="values")
            logit_lengths = tf.identity(output_lengths, name="lengths")
        return Sequence(logit_values, logit_lengths)

    def save_meta(self, path):
        self._saver.export_meta_graph(path)

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
        self.summary = self._get_summary()

    def feed_iterator_from_records(self, patterns, batch_size, seed=None, bucketing=False):
        with self.graph.as_default():
            datasets = [tf.data.Dataset.list_files(pattern)
                        for pattern in patterns]
            dataset = reduce(tf.data.Dataset.concatenate, datasets)
            dataset = dataset.shuffle(10**5, seed)
            dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename), cycle_length=5)
            #dataset = dataset.flat_map(lambda filename: tf.data.TFRecordDataset(filename))
            dataset = dataset.map(parse_tfrecords)
            dataset = dataset.shuffle(1000, seed)
            dataset = dataset.map(
                lambda inputs, label: {"inputs": inputs,
                                       "length": tf.shape(inputs)[0],
                                       "label": label})

            padded_shapes = {"inputs": tf.TensorShape([None, NUM_FEATURES]),
                             "length": tf.TensorShape([]),
                             "label": tf.TensorShape([])}
            if not bucketing:
                dataset = dataset.padded_batch(
                    batch_size=batch_size,
                    padded_shapes=padded_shapes)
            else:
                dataset = bucket_by_sequence_length(
                    dataset, lambda x: x["length"],
                    [2 ** i for i in range(4, 12)],
                    [batch_size] * 12, #[max(batch_size, 2 ** (14 - i)) for i in range(4, 12)],
                    padded_shapes)
            dataset = dataset.map(
                lambda x: (x["inputs"], x["length"], x["label"]))
            feed_iterator = self.iterator.make_initializer(dataset)
        return feed_iterator

    def _get_summary(self):
        with tf.name_scope('summaries'):
            self._add_summary()
            summary = tf.summary.merge_all()
        return summary

    def _add_summary(self):
        for name in self.SUMMARY_SCALARS:
            tf.summary.scalar(name, self.__getattribute__(name))
        batch_shape = tf.shape(self.input.values)
        tf.summary.scalar('batch_size', batch_shape[0])
        tf.summary.scalar('bucketing_length', batch_shape[1])


    def _get_summary_and_writer(self, log_path):
        summary_writer = tf.summary.FileWriter(
            os.path.join(log_path, self.TYPE_NAME),
            flush_secs=30)
        return summary_writer

    @staticmethod
    def _encode(labels, character_lookup):
        with tf.variable_scope("encode"):
            labels_splitted = tf.string_split(labels, delimiter='')
            table = tf.contrib.lookup.index_table_from_tensor(
                mapping=character_lookup,
                default_value=NUM_FEATURES)  # ToDo(dv): introduce unknown symbol
            tokens = table.lookup(labels_splitted)
        return tokens

    def get_evaluation(self, dataset=None, feeder=None, log_path=None):
        with self.graph.as_default():
            if feeder is None:
                feeder = self.iterator.make_initializer(dataset)
            if log_path is not None:
                summary_writer = self._get_summary_and_writer(log_path)
            else:
                summary_writer = tf.no_op(), None
            def evaluation(model_path, save_with_global_step=None): #todo ugly
                with tf.Session(graph=self.graph) as sess:
                    self._saver.restore(sess, model_path)
                    sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
                    sess.run(feeder)
                    tensors = self.error, self.labels, self.loss, self.summary
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
        return loss

    @staticmethod
    def _get_error(predictions, targets):
        with tf.variable_scope("error"):
            error = tf.reduce_mean(tf.edit_distance(predictions, targets))
        return error

class TrainingModel(EvaluationModel):
    TYPE_NAME = "training"
    SUMMARY_SCALARS = (EvaluationModel.SUMMARY_SCALARS +
                       ["global_step", "learning_rate", "epoch"])
    DEFAULT_ALPHABET = list("abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "1234567890 ,.:;*+()/!?&-'\"$")
    MAX_ABS_GRAD = 1.
    DEFAULT_LEARNING_RATE = 0.003
    EVALUATION_SIZE = 500

    def __init__(self, lstm_sizes, share_param_first_layer=True):

        self.global_step = tf.Variable(
            0, name='global_step', trainable=False)
        self.epoch = tf.Variable(0, name='epoch_num', trainable=False)
        self.increment_epoch = tf.assign_add(self.epoch, 1)
        self.learning_rate = tf.Variable(self.DEFAULT_LEARNING_RATE,
                                         name='learning_rate',
                                         trainable=False)
        self._get_summary = lambda : None
        super().__init__(lstm_sizes, share_param_first_layer,
                         alphabet=self.DEFAULT_ALPHABET)
        self._lstm_sizes = lstm_sizes
        self._share_param_first_layer = share_param_first_layer
        self.train_op, self.gradients = self._get_train_op(
            self.loss, self.global_step, self.learning_rate)
        self.initializer = self._get_initializer()
        self.summary = EvaluationModel._get_summary(self)


    def _add_summary(self):
        super()._add_summary()
        gradient_is_capped = [(tf.to_float(tf.equal(tf.abs(grad), self.MAX_ABS_GRAD)), var)
                              for grad, var in self.gradients]
        frac_capped_gradients = (sum(tf.reduce_mean(v) * tf.to_float(tf.size(v)) for v, _ in gradient_is_capped) /
                                 tf.to_float(sum(tf.size(v) for v, _ in gradient_is_capped)))
        tf.summary.scalar("frac_capped_gradients", frac_capped_gradients)
        # frac_capped_gradients = {var.name: tf.reduce_mean(val) for val, var in gradient_is_capped}
        # for name, value in frac_capped_gradients.items():
        #     with tf.name_scope('summaries'):
        #         with tf.name_scope('frac_capped_gradients'):
        #             tf.summary.scalar(name, value)


    def _get_train_op(self, loss, global_step, learning_rate):
        """ return function to perform a single training step
        """
        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients = optimizer.compute_gradients(loss)

            capped_gradients = [(tf.clip_by_value(grad, -self.MAX_ABS_GRAD, self.MAX_ABS_GRAD), var)
                                for grad, var in gradients]

            train_op = optimizer.apply_gradients(capped_gradients,
                                                 global_step=global_step)
        return train_op, capped_gradients

    @staticmethod
    def _get_initializer():
        return [tf.global_variables_initializer(),
                tf.local_variables_initializer()]

    def save_checkpoint(self, sess, path, global_step):
        checkpoints_path = self._saver.save(
            sess, os.path.join(path, "model"),
            global_step=global_step)
        return checkpoints_path


    def _get_saver(self):
        var_list = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) +
                    tf.get_collection('alphabet'))
        saver = tf.train.Saver(
            var_list,
            keep_checkpoint_every_n_hours=0.5)
        return saver

    def train(self,
              dataset_patterns,
              path,
              test_datasets_pattern=None,
              steps_per_checkpoint=250,
              epoch_num=1,
              batch_size=32,
              learning_rate=None,
              fine_tuning=True,
              profiling_steps=None,
              bucketing=False):
        path_models = os.path.join(path, "models")
        path_timeline = os.path.join(path, "timeline")
        if profiling_steps is not None:
            if not os.path.exists(path_timeline):
                os.makedirs(path_timeline)
        with tf.Graph().as_default():
            model = InferenceModel(self._lstm_sizes,
                                   self._share_param_first_layer,
                                   self.DEFAULT_ALPHABET)
            model.save_meta(os.path.join(path_models, "inference.meta"))
        with tf.Graph().as_default():
            model = EvaluationModel(self._lstm_sizes,
                                    self._share_param_first_layer,
                                    self.DEFAULT_ALPHABET)
            model.save_meta(os.path.join(path_models, "evaluation.meta"))
        feed_iterator_from_dataset = self.feed_iterator_from_records(
            dataset_patterns, batch_size=batch_size, bucketing=bucketing)
        summary_writer = self._get_summary_and_writer(
            os.path.join(path, "log", "train"))

        if test_datasets_pattern is not None:
            feed_iterator_from_datasets = {
                name: model.feed_iterator_from_records([pattern], self.EVALUATION_SIZE, seed=0)
                for name, pattern in test_datasets_pattern.items()}
            evalation_functions = [
                model.get_evaluation(
                    feeder=feeder,
                    log_path=os.path.join(path, "log", "test", name))
                for name, feeder in feed_iterator_from_datasets.items()]
        else:
            evalation_functions = []

        with tf.Session(graph=self.graph) as sess:
            #self._saver.restore(sess, model_path)
            sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
            sess.run(self.initializer)
            if learning_rate is not None:
                sess.run(tf.assign(self.learning_rate, learning_rate))
            while sess.run(self.epoch) < epoch_num:
                sess.run(feed_iterator_from_dataset)
                if (sess.run(self.epoch) == epoch_num - 1) and fine_tuning:
                    sess.run(tf.assign(self.learning_rate, self.learning_rate/5))
                for step_in_epoch in itertools.count():
                    global_step = sess.run(self.global_step)
                    if (step_in_epoch % steps_per_checkpoint == 0) and global_step > 0:
                        checkpoints_path = self.save_checkpoint(
                            sess, path_models, global_step)
                        for evalation_function in evalation_functions:
                            evalation_function(checkpoints_path, global_step)
                    try:
                        if (profiling_steps is not None) and (global_step in profiling_steps):
                            run_metadata = tf.RunMetadata()
                            args = dict(
                                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                run_metadata=run_metadata)
                        else:
                            args = dict()
                        _, evaled_summary, loss = sess.run(
                                [self.train_op, self.summary, self.loss], **args)
                        summary_writer.add_summary(evaled_summary, global_step)
                        if (profiling_steps is not None) and (global_step in profiling_steps):
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open(os.path.join(path_timeline, 'timeline_{}.json' % (global_step,)), 'w') as f:
                                f.write(chrome_trace)
                        print(sess.run(self.epoch), step_in_epoch, loss)
                    except tf.errors.OutOfRangeError:
                        sess.run(self.increment_epoch)
                        print("epoch completed")
                        break
