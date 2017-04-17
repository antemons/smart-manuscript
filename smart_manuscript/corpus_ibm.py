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

from tensorflow.python.platform.app import flags
import os
import glob

from .inkml import InkML, TraceGroup
from .corpus import Corpus, Corpora, TranscriptedStrokes

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


flags.DEFINE_string("ibm_ub_path", "data/IBM_UB_1/query/",
                    "path to IBM_UB_1 folder (unzipped)")


class IBMub(InkML):

    TRANSCRIPTION = "Tg_Truth"

    def get_segments(self):
        for trace_group in self._root.childs_of_type(TraceGroup):
            yield TranscriptedStrokes(
                transcription=trace_group.annotation[self.TRANSCRIPTION],
                strokes=trace_group.ink())


def load(path, max_files=None):
    """ return corpora (corpus of all files) of the IBM UB database
    """
    FILETYPE = ".inkml"
    corpora = Corpora()
    files = glob.glob(os.path.join(path, './*.inkml'))[:max_files]
    for i, path in enumerate(files):
        name = os.path.basename(path).replace(".inkml", "")
        print("import {:4}/{:4} ({:20})".format(i, len(files), name), end="\r")
        corpora[name] = Corpus(IBMub(path).get_segments())
    print("{:40}".format(""), end="\r")
    return corpora


def main():
    """ load the full IBM UB database and show random lines and transcription
    """

    corpora = load(flags.FLAGS.ibm_ub_path, max_files=10)
    corpus = corpora.merge()
    while True:
        corpus.plot_sample()


if __name__ == "__main__":
    main()
