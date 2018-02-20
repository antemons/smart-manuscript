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
import numpy as np

from .writing import strokes_to_features, InkPage
from .model import Sequence

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

from . import writing
NUM_FEATURES = writing.InkFeatures.NUM_FEATURES


def get_dataset_from_features(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.float32, output_shapes=[None, NUM_FEATURES])
    dataset = dataset.map(
        lambda features: (features, tf.shape(features)[0], ""))
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(tf.TensorShape([None, NUM_FEATURES]),
                       tf.TensorShape([]),
                       tf.TensorShape([])))
    return dataset


class Reader:
    """ read the handwriting with the help of a trained tensorflow graph

    """

    def __init__(self, model_path):
        """
        Args:
            path (str): path to directory with the tensorflow-graph and alphabet
            model (str): name of the model
        """
        self.graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph(model_path + ".meta")
        self.inputs = Sequence(
            values=self.graph.get_tensor_by_name("inputs/features:0"),
            length= self.graph.get_tensor_by_name("inputs/length:0"))
        self.labels = self.graph.get_tensor_by_name("output/labels:0")
        self.probabilities = self.graph.get_tensor_by_name("output/probabilities:0")

        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver.restore(self.session, model_path)

    def recognize(self, inks):
        """ generate the transcription suggenstions

        Args:
            ink (nested list of arrays[N,2]): the trajectories
        """


        def features():
            return (strokes_to_features(ink.strokes) for ink in inks)
        with self.graph.as_default():
            dataset = get_dataset_from_features(features, batch_size=len(inks))
            *inputs, _ = dataset.make_one_shot_iterator().get_next()
        evaluated_inputs = self.session.run(inputs)
        feed_dict = {self.inputs.values: evaluated_inputs[0],
                     self.inputs.length: evaluated_inputs[1]}

        labels, probabilities = self.session.run(
            [self.labels, self.probabilities],
            feed_dict=feed_dict)

        normalized_probabilities = 100 * probabilities / np.sum(probabilities, axis=1)[:, None]
        decoded_labels = [[proposal.decode() for proposal in example] for example in labels]

        return [l for l in decoded_labels[0]], labels, normalized_probabilities


    def recognize_line(self, ink):
        """ generate the transcription suggenstions

        Args:
            ink (nested list of arrays[N,2]): the trajectories
        """

        features = strokes_to_features(ink)
        feed_dict = {self.inputs.values: np.expand_dims(features, 0),
                     self.inputs.length: np.array([len(features)])}

        labels, probabilities = self.session.run(
            [self.labels, self.probabilities],
            feed_dict=feed_dict)

        normalized_probabilities = 100 * probabilities / np.sum(probabilities, axis=1)
        decoded_labels = [[proposal.decode() for proposal in example] for example in labels]
        print(decoded_labels[0][0], str(int(normalized_probabilities[0][0]))+"%")
        return decoded_labels[0][0], [l[0] for l in decoded_labels], normalized_probabilities[0]

    def recognize_page(
            self, stroke_page):
        """ recognize a full page
        """
        ink_page = InkPage(stroke_page)
        #lines, *_ = self.recognize(ink_page.lines)
        lines = [self.recognize_line(ink_line.strokes)[0]
                 for ink_line in ink_page.lines]
        return "\n".join(lines)
