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
from tensorflow.python.ops.ctc_ops import ctc_beam_search_decoder
import stroke_features as sf
import numpy as np
from os import path

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


tf.app.flags.DEFINE_string(
    "graph_path", path.join(path.dirname(__file__), '..', 'sample_graph'),
    "directory for the tensorflow-graph and variables")


def labels_to_transcription(labels, alphabet):
    """ convert labels (list of int) to transcription (str)
    """
    return "".join([alphabet[n] for n in labels])


def transcription_to_labels(transcription, alphabet):
    """ convert transcription (str) to labels (list of int)
    """
    return [alphabet.index(c) for c in transcription]


class GraphUtilities(object):
    """ Provides a converter to feed the tensorflow placeholders

        The placeholders input, input_seq_length,
        target_shape, target_values, and target_indices must be members of the
        child class
    """

    def convert_to_tf(self, features, labels=None):
        """ prepare tensorflow feed_data

        Args:
            features (nested list of arrays): the features
            labels (list string): labels

        Return: a dict of the tf-placeholders keys
        """
        feed_dict = {}
        max_input_len = max(len(f) for f in features)
        batch_size = len(features)
        input_ = np.zeros([batch_size, max_input_len,
                           sf.StrokeFeatures.NUM_FEATURES])
        input_seq_length = np.zeros([batch_size], int)
        for i, f in enumerate(features):
            input_[i, :len(f)] = f
            input_seq_length[i] = len(f)
        feed_dict[self.input] = input_
        feed_dict[self.input_seq_length] = input_seq_length

        if labels is not None:
            assert batch_size == len(labels)
            target_values = []
            target_indices = []
            for b in range(batch_size):
                for t, l in enumerate(labels[b]):
                    target_values.append(l)
                    target_indices.append([b, t])
            max_target_len = max([len(l) for l in labels])
            feed_dict[self.target_shape] = [batch_size, max_target_len]
            feed_dict[self.target_values] = target_values
            feed_dict[self.target_indices] = target_indices
        return feed_dict




class Reader(GraphUtilities):
    """ read the handwriting with the help of a trained tensorflow graph

    """

    def __init__(self, path):
        """
        Args:
            path (str): path to directory with the tensorflow-graph and
                        -variables
        """

        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph(path + "/model.meta")
        saver.restore(self.session, path + "/model.ckpt")
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.input_seq_length = graph.get_tensor_by_name("input_seq_length:0")
        self.logits = graph.get_tensor_by_name("logits:0")
        self.output_states = tf.get_collection("output_states")[0]

        self.alphabet = pickle.load(open(path + "/alphabet.pkl", "rb"))

    def labels_to_str(self, labels):
        return labels_to_transcription(labels, self.alphabet)

    def recognize(
            self, ink, debug_lstm=False, print_output=False,
            num_proposals=1):
        """ generate the transcription suggenstions

        Args:
            ink (nested list of arrays[N,2]): the trajectories
        """

        features = sf.StrokeFeatures(ink).features
        feed_dict = self.convert_to_tf([features])

        beam_search = ctc_beam_search_decoder(
            self.logits, self.input_seq_length,
            top_paths=num_proposals,
            merge_repeated=False)
        predictions, predictions_prob = self.session.run(
            beam_search, feed_dict=feed_dict)
        prediction = predictions[0].values

        if print_output:
            print(self.labels_to_str(prediction), "[",
                  self.labels_to_str(predictions[1].values),
                  "(%.2f) ," % (predictions_prob[0][1]/predictions_prob[0][0]),
                  self.labels_to_str(predictions[2].values),
                  "(%.2f) ," % (predictions_prob[0][2]/predictions_prob[0][0]),
                  "]")

        if debug_lstm:
            batch_of_features = [features[:i+1] for i in range(len(features))]
            feed_dict = self.convert_to_tf(batch_of_features)
            output_states = self.session.run(
                [self.output_states], feed_dict=feed_dict)
            lstm_c = [output_states[0][i][20] for i in range(len(features))]
            sf.StrokeFeatures(ink).plot(data=lstm_c)
        return [self.labels_to_str(p.values) for p in predictions]

    def recognize_page(
            self, stroke_page, *args, **kwargs):
        """ recognize a full page
        """
        text = ""
        for i, ink_line in enumerate(stroke_page.lines):
            print(
                "Read line: {:3} / {:3}".format(i + 1, len(stroke_page.lines)),
                 flush=True, end="\r")
            line = self.recognize(ink_line, *args, **kwargs)[0]
            print(" {:20}".format(""), end="\r")
            print(line)
            text += line + "\n"
        print("")
        return text
