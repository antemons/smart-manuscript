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

import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops.ctc_ops import ctc_beam_search_decoder
import numpy as np
import os.path

from .encoder import Encoder
from .stroke_features import strokes_to_features, InkPage

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


Input = namedtuple('Input', ('sequence, length'))

class Reader:
    """ read the handwriting with the help of a trained tensorflow graph

    """

    def __init__(self, model_path):
        """
        Args:
            path (str): path to directory with the tensorflow-graph and alphabet
            model (str): name of the model
        """
        model_files = model_path
        encoder_path = os.path.dirname(model_files)
        self.encoder = Encoder.from_file(encoder_path)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph(model_files + ".meta")
        saver.restore(self.session, model_files)
        graph = tf.get_default_graph()
        self.input = Input(
            sequence=graph.get_tensor_by_name("input_sequence:0"),
            length= graph.get_tensor_by_name("input_seq_length:0"))
        self.logits = graph.get_tensor_by_name("logits:0")
        #self.output_states = tf.get_collection("output_states")[0]

    def recognize(
            self, ink, debug_lstm=False, print_output=False,
            num_proposals=1):
        """ generate the transcription suggenstions

        Args:
            ink (nested list of arrays[N,2]): the trajectories
        """

        features = strokes_to_features(ink)
        feed_dict = {self.input.sequence: np.expand_dims(features, 0),
                     self.input.length: np.array([len(features)])}

        beam_search = ctc_beam_search_decoder(
            self.logits, self.input.length,
            top_paths=num_proposals,
            merge_repeated=False)
        predictions, predictions_prob = self.session.run(
            beam_search, feed_dict=feed_dict)
        prediction = predictions[0].values

        if print_output:
            print(self.encoder.decode(prediction), "[",
                  self.encoder.decode(predictions[1].values),
                  "(%.2f) ," % (predictions_prob[0][1]/predictions_prob[0][0]),
                  self.encoder.decode(predictions[2].values),
                  "(%.2f) ," % (predictions_prob[0][2]/predictions_prob[0][0]),
                  "]")

        # if debug_lstm:
        #     batch_of_features = [features[:i+1] for i in range(len(features))]
        #     feed_dict = self.convert_to_tf(batch_of_features)
        #     output_states = self.session.run(
        #         [self.output_states], feed_dict=feed_dict)
        #     lstm_c = [output_states[0][i][20] for i in range(len(features))]
        #     InkFeatures(ink).plot(data=lstm_c)
        return [self.encoder.decode(p.values) for p in predictions]

    def recognize_page(
            self, stroke_page, *args, **kwargs):
        """ recognize a full page
        """
        ink_page = InkPage(stroke_page)
        text = ""
        print("Transcription:")
        for i, ink_line in enumerate(ink_page.lines):
            print(
                "Read line: {:3} / {:3}".format(i + 1, len(ink_page.lines)),
                 flush=True, end="\r")
            line = self.recognize(ink_line.strokes, *args, **kwargs)[0]
            print(" {:20}".format(""), end="\r")
            print(line)
            text += line + "\n"
        print("")
        return text
