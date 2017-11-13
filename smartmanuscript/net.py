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

from collections import namedtuple
from scipy.sparse import csc_matrix
import tensorflow as tf
from tensorflow.python.platform.app import flags
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
import numpy as np
import glob
import os.path

from .utils import cached_property, colored_str_comparison



__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

NUM_FEATURES = 15  # see stroke_features

Batch = namedtuple('Batch', ('target', 'input'))
Target = namedtuple('Target', ('indices', 'values', 'shape'))
Input = namedtuple('Input', ('sequence, length'))


class Example:
    """ Hold a bunch of tensors for input and target
    """
    def __init__(self, target, input_):
        self.input = input_
        self.target = target

    def to_list(self):
        """ return list which is supported by tensorflow operations

        Retruns:
            list of tensors
        """
        return [self.target.indices, self.target.values,
                self.target.dense_shape,
                self.input.sequence, self.input.length]

    @classmethod
    def from_list(cls, tensors):
        """ creation from list

        is helpful when this list was obtained by to_list and passed to an
        tensorflow operaton

        Args:
            tensors: list of tensors
        """
        tensors[0].set_shape((None, 1))
        tensors[1].set_shape((None,))
        tensors[2].set_shape((1,))
        tensors[3].set_shape((None, NUM_FEATURES))
        tensors[4].set_shape(())

        return cls(
            target=tf.SparseTensor(indices=tensors[0],
                           values=tensors[1],
                           dense_shape=tensors[2]),
            input_=Input(sequence=tensors[3], length=tensors[4]))


def record_to_batch(filenames=None, batch_size=5,
                    shuffle=False,
                    allow_smaller_final_batch=False,
                    num_epochs=None,
                    compose=None):
    """


    """

    def get_example_from_records(filenames, num_epochs):
        """ Read single examples from record
        """
        print(filenames)
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)  # TODO(daniel): None?
        #filename_queue = tf.Print(filename_queue, data=[], message="asd")
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'input': tf.FixedLenSequenceFeature(
                    [NUM_FEATURES], tf.float32)},
            #    'label': tf.FixedLenSequenceFeature([], tf.int64)},
            context_features={
                'label': tf.VarLenFeature(tf.int64)})
        length = tf.shape(sequence['input'])[0]
        label = context['label']
        #label = tf.Print(label, data=label, message="asd")
        #label_dense = sequence['label']
        input_ = sequence['input']
        #length = tf.Print(length, data=[label.values], message="asd")

        return Example(
            target=label,
            input_=Input(sequence=input_, length=length))

    def shuffled(tensors):
        """ return the tensor(s) in shuffled order
        """
        shuffle = tf.RandomShuffleQueue(capacity=10000,
            dtypes=[tensor.dtype for tensor in tensors],
            #shapes=[tensor.shape for tensor in tensors],  # does not work
            min_after_dequeue=5000)

        shuffle_ops = [shuffle.enqueue(tensors)]

        tf.train.add_queue_runner(tf.train.QueueRunner(shuffle, shuffle_ops))

        tensors_shuffled = shuffle.dequeue()
        for tensor_shuffled, tensor in zip(tensors_shuffled, tensors):
            tensor_shuffled.set_shape(tensor.get_shape())
        return tensors_shuffled

    # def shuffled_example(example):
    #     """ return the examples in shuffled order
    #     """
    #
    def batched_example(example, batch_size):
        """ return examples in minibatches
        """
        target, sequence, length = tf.train.batch(
            [example.target,
             example.input.sequence,
             example.input.length],
            batch_size=batch_size, capacity=1000, dynamic_pad=True,
            allow_smaller_final_batch=allow_smaller_final_batch)
        return Batch(target=target,
                     input=Input(sequence=sequence, length=length))

    if compose is None:
        example = get_example_from_records(filenames, num_epochs)

    else:
        examples = [get_example_from_records([filename], num_epochs)
                    for filename, batch_size in compose.items()]

        cum_probabilities = np.cumsum([batch_size for _, batch_size in compose.items()])
        cum_probabilities = tf.constant(cum_probabilities, dtype=tf.float32)
        random_number = tf.random_uniform((), maxval=cum_probabilities[-1])
        def get_example(example):
            def func(): return example.to_list()
            return func
        example = Example.from_list(tf.case([
            (tf.less(random_number, cum_probabilities[i]), get_example(example))
            for i, example in enumerate(examples)],
            default=get_example(examples[0]), exclusive=False))

    if shuffle:
        tensors = example.to_list()
        shuffled_tensors = shuffled(tensors)
        example = Example.from_list(shuffled_tensors)
        #example = shuffled_example(example)
    return batched_example(example, [batch_size])


class NeuralNetworks:
    """ a CTC-BLSTM neural network to transcript handwriting

    """
    BATCH_SIZE = None
    MAX_INPUT_LEN = None

    def __init__(self, encoder, name,
                 input_=None, target=None,
                 lstm_sizes=[128, 128],
                 share_param_first_layer=True):
        self.encoder = encoder
        self._lstm_sizes = lstm_sizes
        self.NUM_CLASSES = len(self.encoder.alphabet) + 1
        self.path = f"graphs/{name}/"
        self._share_param_first_layer = share_param_first_layer
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self._input = input_
        self._target = target
        self._build_tensors()
        print(self)

    def __str__(self):
        return f"CTC-BLSTM with {self._lstm_sizes} neurons per LSTM cell"

    def _build_tensors(self):
        self.train_step
        self.error
        #_ = self.initializer

    @cached_property
    def input(self):
        if self._input is not None:
            sequence = tf.placeholder_with_default(
                self._input.sequence, name="input_sequence",
                shape=[self.BATCH_SIZE, self.MAX_INPUT_LEN, NUM_FEATURES])
            length = tf.placeholder_with_default(
                self._input.length, shape=(self.BATCH_SIZE),
                name="input_seq_length")
            return Input(sequence=sequence, length=length)
        else:
            sequence = tf.placeholder(
                dtype=tf.float32, name="input_sequence",
                shape=[self.BATCH_SIZE, self.MAX_INPUT_LEN, NUM_FEATURES])
            length = tf.placeholder(
                dtype=tf.int32, shape=(self.BATCH_SIZE),
                name="input_seq_length")
            return Input(sequence=sequence, length=length)

    @cached_property
    def target(self):
        if self._target is not None:
            return self._target
        else:
            return tf.SparseTensor(self.target_decomposed.indices,
                                   self.target_decomposed.values,
                                   self.target_decomposed.shape)

    @cached_property
    def target_decomposed(self):
        indices = tf.placeholder(
            tf.int64, name="target_indices")
        values = tf.placeholder(tf.int64, name="target_values")
        shape = tf.placeholder(tf.int64, name="target_shape")
        return Target(indices=indices, values=values, shape=shape)

    @cached_property
    def logits(self):
        layer_input = self.input.sequence
        for n_layer, num_units in enumerate(self._lstm_sizes):
            # TODO: define new OpenLSTMCell(LSTMCell)
            # TODO(dv): try to use shared weigth for fw and bw

            def create_cell(num_units, input_size,
                            reuse=None):
                #print(input_size)
                cell = LSTMCell(num_units,
                                state_is_tuple=True,
                                reuse=reuse)
                #return cell
                dropout_cell = DropoutWrapper(
                    cell=cell,
                    output_keep_prob=0.5,
                    state_keep_prob=0.5,
                    variational_recurrent=True,
                    #input_size=input_size,
                    dtype=tf.float32)
                return dropout_cell

            cell_bw, cell_fw = [
                create_cell(num_units,
                            layer_input.get_shape()[1:])
                for _ in ("bw", "fw")]
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw,
                inputs=layer_input, dtype=tf.float32,
                scope=f"layer_{n_layer}",
                sequence_length=self.input.length)
            layer_output = tf.concat(outputs, 2)
            layer_input = layer_output
        W = tf.Variable(
            tf.truncated_normal([2 * num_units, self.NUM_CLASSES],
                                stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.NUM_CLASSES]))
        layer_output = tf.reshape(
            layer_output, [-1, 2 * num_units])
        # TODO(daniel): apply sigmoid or softmax?
        logits = layer_output @ W + b
        batch_size = tf.shape(self.input.sequence)[0]
        logits = tf.reshape(logits, [batch_size, -1, self.NUM_CLASSES])
        logits = tf.transpose(logits, [1, 0, 2], name="logits")
        return logits

    @cached_property
    def prediction(self):
        prediction = ctc.ctc_beam_search_decoder(
            self.logits, self.input.length, top_paths=1,
            merge_repeated=False)[0][0]
        return prediction

    def decode(self, prediction):
        labels = csc_matrix((prediction.values, prediction.indices.transpose()),
                            shape=prediction.dense_shape)
        if labels.shape[1] == 0:
            return labels.shape[0] * [""]
        transcriptions = []
        for i in range(labels.shape[0]):
            row = labels.getrow(i)
            label = row.toarray()[0][:row.nnz]
            transcription = self.encoder.decode(label)
            transcriptions.append(transcription)
        return transcriptions

    @cached_property
    def loss(self):
        loss = tf.reduce_mean(ctc.ctc_loss(
            tf.to_int32(self.target), self.logits,  self.input.length,
            ctc_merge_repeated=False))
        tf.summary.scalar('loss', loss)
        return loss

    @cached_property
    def train_step(self):
        """ return function to perform a single training step
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(self.loss,
                                        global_step=self.global_step)
        return train_step

    @cached_property
    def error(self):
        error = tf.reduce_mean(tf.edit_distance(self.prediction, self.target))
        tf.summary.scalar('error', error)
        return error

    @cached_property
    def global_step(self):
        return tf.Variable(0, name='global_step', trainable=False)

    @cached_property
    def learning_rate(self):
        # learning_rate_default = tf.constant(self.LEARNING_RATE[0])
        # learning_rate_fine = tf.constant(self.LEARNING_RATE[1])
        # step_threshold = tf.constant(self.LEARNING_RATE[2])
        #
        # learning_rate =  tf.cond(tf.less(self.global_step, step_threshold),
        #                          lambda: learning_rate_default,
        #                          lambda: learning_rate_fine,
        #                          name="learning_rate")
        learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        #tf.summary.scalar('learning_rate', learning_rate)
        return learning_rate

    @cached_property
    def initializer(self):
        return [tf.global_variables_initializer(),
                tf.local_variables_initializer()]#()initialize_all_variables

    @cached_property
    def _saver(self):
        return tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

    def save_meta(self, name=None):
        if name is None:
            name = 'model.meta'
        path = os.path.join(self.path, name)
        self._saver.export_meta_graph(path)
        self.encoder.save(self.path)
        print("graph and encoder have been saved")

    def save_checkpoint(self, sess, global_step=None):
        #if name is None:
        name = "model.ckpt"
        path = os.path.join(self.path, name)
        self._saver.save(
            sess, path, global_step=global_step)
        self.encoder.save(self.path)
        print("variables have been saved")

    @cached_property
    def summary(self):
        self.error, self.loss
        return tf.summary.merge_all()


class Training:
    Evaluation = namedtuple("Evaluation", ["batch", "writer"])
    EVALUATION_INTERVAL = 25
    SAVE_INTERVAL = 250
    EVALUATION_BATCH_SIZE = 300
    DEFAULT_BATCH_SIZE = 50

    IAM_TEST_SET = [line.rstrip("\n").replace(".inkml" , "")
                    for line in open("data/IAMonDo-db-1.0/0.set", "r")]
    IBM_TEST_SET = ["GPIquery_01.tfrecords", "MVLquery_50.tfrecords"]

    def __init__(self, data_path="tmp"):
        self._data_path = data_path

    def test_records(self):

        result =  dict(
            daniel=glob.glob(self._data_path + "/my_test/*.tfrecords"),
            python_zen=glob.glob(self._data_path + "/zen_test/*.tfrecords"),
            iam_word=[filename
                for filename in glob.glob(self._data_path + "/iam_word/*.tfrecords")
                if any(k in filename for k in self.IAM_TEST_SET)],
            iam_line=[filename
                for filename in glob.glob(self._data_path + "/iam_line/*.tfrecords")
                if any(k in filename for k in self.IAM_TEST_SET)],
            ibm=[filename
                 for filename in glob.glob(self._data_path + "/ibm/*.tfrecords")
                 if any(k in filename for k in self.IBM_TEST_SET)])
        result = {k: v for k, v in result.items() if v}
        return result

    @classmethod
    def train_records(cls, data_path):

        result =  (10 * glob.glob(data_path + "/my_train/*.tfrecords") +
                [filename
                 for filename in glob.glob(data_path + "/iam_word/*.tfrecords")
                 if not any(k in filename for k in cls.IAM_TEST_SET)] +
                2 * [filename
                 for filename in glob.glob(data_path + "/iam_line/*.tfrecords")
                 if not any(k in filename for k in cls.IAM_TEST_SET)] +
                [filename
                 for filename in glob.glob(data_path + "/ibm/*.tfrecords")
                 if not any(k in filename for k in cls.IBM_TEST_SET)])
        return result

    @classmethod
    def train_batch(cls, data_path):
        return record_to_batch(
            filenames=cls.train_records(data_path),
            batch_size=cls.DEFAULT_BATCH_SIZE)

    @cached_property
    def batch_size(self):
        return tf.placeholder_with_default(
            tf.constant(self.DEFAULT_BATCH_SIZE, tf.int32), [])

    def evaluation(self, filenames, name, net, graph=None):
        writer = tf.summary.FileWriter(
            os.path.join(net.path, name), graph)
        with tf.Graph().as_default():
            batch = record_to_batch(filenames,
                                    batch_size=self.EVALUATION_BATCH_SIZE,
                                    allow_smaller_final_batch=True,
                                    shuffle=True, num_epochs=1)
            coord = coordinator.Coordinator()
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()])
                # see: https://github.com/tensorflow/tensorflow/issues/1045
                threads = queue_runner.start_queue_runners(sess, coord)
                evaled_batch = sess.run(batch)
                coord.request_stop()
                coord.join(threads)
        print("loaded evaluation from {} with {} elements"\
            .format(filenames, evaled_batch.target.dense_shape[0]))
        return self.Evaluation(batch=evaled_batch, writer=writer)


    def evaluate(self, evaluation, sess, net, name, num_examples=1):
        feed_dict = {
            net.input: evaluation.batch.input,
            net.target: evaluation.batch.target}

        summary, error, prediction, target, step = sess.run(
            [net.summary, net.error, net.prediction,
             net.target, net.global_step], feed_dict=feed_dict)

        evaluation.writer.add_summary(summary, step)
        print(f"Evaluation ({name}): {100*error:2.0f}%")
        for predicted_transcription, target_transcription \
            in zip(net.decode(prediction)[:num_examples], net.decode(target)):
            if len(predicted_transcription) > 1.5 * len(target_transcription):
                predicted_transcription = "..."
            print(colored_str_comparison(target_transcription,
                                         predicted_transcription))
        return error

    @staticmethod
    def create_learning_profile(
        learning_rate_default, total_num_steps,
        final_learning_rate, num_final_steps):

        def learning_rate(step):
            if step < total_num_steps - num_final_steps:
                return learning_rate_default
            else:
                return final_learning_rate
        return learning_rate

    def train(self, net, num_steps, learning_rate_default, final_learning_rate = None,
              num_final_steps=0, restore_from=None):
        """ train the network

        Args:
            batch_test (dict of list of tuples (features, labels)):
                on each item a test of the accuracy is performed
        """
        print("train network", flush=True)

        evaluations = {
            name: self.evaluation(filename, name, net, tf.get_default_graph())
            for name, filename in self.test_records().items()}

        writer_train = tf.summary.FileWriter(
            os.path.join(net.path, "train"), tf.get_default_graph())

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)

        learning_rate = self.create_learning_profile(
            learning_rate_default, num_steps, final_learning_rate, num_final_steps)
        with tf.Session(config=config) as sess:
            sess.run(net.initializer)
            tf.train.start_queue_runners(sess=sess)
            if restore_from is not None and restore_from != "":
                print(f"restore model from: {restore_from}")
                net._saver.restore(sess, restore_from)

            while True:
                step = sess.run(net.global_step)
                if step % 10 == 0:
                    print(f"step: {step:6}", flush=True)


                if step % self.EVALUATION_INTERVAL == 0:
                    for key, evaluation in evaluations.items():
                        self.evaluate(evaluation, sess, net, key)

                if (step % self.SAVE_INTERVAL == 0 or step == num_steps):
                    net.save_checkpoint(sess, step)


                _, target, summary = sess.run(
                    [net.train_step, net.target, net.summary],
                    feed_dict={net.learning_rate: learning_rate(step)})
                writer_train.add_summary(summary, step)
                #decoded_target = net.decode(target)
                #print(decoded_target)

                if step == num_steps:
                    print("training has been finished")
                    break
