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

from random import choice
import pylab as plt
import glob
import os
from collections import namedtuple

from .writing import InkPage

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

TranscriptedStrokes = namedtuple('TranscriptedStrokes',
                                 ['transcription', 'strokes'])

class Corpus(list):
    """ Corpus is a list of tuples (transcription, strokes)
    """


    def plot_sample(self, rows=4, columns=2):
        _, axes_arr = plt.subplots(rows, columns)
        for axes in axes_arr.reshape(-1):
            transcription, ink = choice(self)
            ink.plot_pylab(axes, transcription)
        plt.show()

    def __getitem__(self, val):
        if val is slice:
            return type(self)(super().__getitem__(val))
        else:
            return super().__getitem__(val)

    def __add__(self, other):
        result = type(self)()
        result.extend(self)
        result.extend(other)
        return result


class Corpora(dict):
    """ Dict of {name: corpus}
    """
    def merge(self):
        return sum(self.values(), Corpus())
