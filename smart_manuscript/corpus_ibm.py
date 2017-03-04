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

from inkml_new import InkML, TraceGroup
from tensorflow.python.platform.app import flags
import numpy as np
import pylab as plt
import os
from corpus import Corpus
from stroke_features import Ink
from utils import Bunch
import glob

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


flags.DEFINE_string("ibm_ub_path", "data/IBM_UB_1/query/",
                    "path to IBM_UB_1 folder (unzipped)")


class IBMub(InkML):

    TRANSCRIPTION = "Tg_Truth"

    def get_segments(self):
        for trace_group in self._root.childs_of_type(TraceGroup):
            yield Bunch(
                transcription=trace_group.annotation[self.TRANSCRIPTION],
                ink=trace_group.ink())


def load(path):
    FILETYPE = ".inkml"
    corpus = Corpus()
    print(path + "*" + FILETYPE)
    num_files = len(glob.glob(path + "*" + FILETYPE))
    for i, filename in enumerate(glob.glob(path + "*" + FILETYPE)):
        print("load file ({:3}/{:3}): {}".format(i, num_files, filename),
              end="\r", flush=True)
        for segment in IBMub(filename).get_segments():
            corpus.append((segment.transcription, Ink(segment.ink)))
        if i == 10:
            break
    print("{:60}".format(""))
    return corpus


def main():
    """ load the full IBM UB database and show random lines and transcription
    """

    corpus = load(flags.FLAGS.ibm_ub_path)
    while True:
        corpus.plot_sample()


if __name__ == "__main__":
    main()
