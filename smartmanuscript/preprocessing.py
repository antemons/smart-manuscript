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

import pickle
from os.path import join as join_path
from collections import namedtuple, defaultdict
import warnings

from .writing import strokes_to_features, NormalizationWarning

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"



class PreprocessWarning(Warning):
    def __init__(self, description, label):
        super().__init__("{} ({})". format(description, label))

warnings.simplefilter('ignore', PreprocessWarning)

LabeledFeatures = namedtuple('LabeledFeatures', ['label', 'feature'])

def preprocessed_transcription(transcription):
    transcription = transcription.replace(" .", ".")
    transcription = transcription.replace(" ;", ";")
    transcription = transcription.replace(" ,", ",")
    transcription = transcription.replace("`", "'")
    transcription = transcription.rstrip(" ")
    transcription = transcription.replace("<Symbol/>", "+")
    transcription = transcription.replace(b'\xc2\xb4'.decode(), "'")
    return transcription

def preprocessed(example, skew_is_horizontal=False):
    """ convert example (TranscriptedStrokes) to LabeledFeatures
    """
    strokes, transcription = example.strokes, example.transcription
    transcription = preprocessed_transcription(transcription)  # TODO(dv): merge
    return LabeledFeatures(
        label=transcription,
        feature=strokes_to_features(example.strokes,
                                    skew_is_horizontal=skew_is_horizontal))

def preprocessed_corpus(corpus, min_words=0, min_letters=1,
                        skew_is_horizontal=False):
    warnings.simplefilter('error', NormalizationWarning)
    result = []
    sorted_out = defaultdict(lambda: 0)
    for i, example in enumerate(corpus):
        if (i % 100) == 0:
            print(
                "{:5}/{:5} {} {:20}".format(
                    i, len(corpus), example.transcription, ""),
                end="\r", flush=True)

        transcription = preprocessed_transcription(
            example.transcription)

        if len(transcription) < max(min_letters, 1):
            warnings.warn(PreprocessWarning("too short transcription",
                                            example.transcription))
            sorted_out["too_few_letters"] += 1
            continue

        if transcription in ['.', ',', "'"]:
            warnings.warn(PreprocessWarning("only a single dot",
                                            example.transcription))
            sorted_out["dot"] += 1
            continue


        if len(transcription.split(" ")) < min_words:
            warnings.warn(PreprocessWarning("too short transcription",
                                            example.transcription))
            sorted_out["too_few_words"] += 1
            continue

        try:
            labeled_features = preprocessed(
                example, skew_is_horizontal=skew_is_horizontal)
        except NormalizationWarning:
            warnings.warn(PreprocessWarning("Normalization failed",
                                            example.transcription))
            sorted_out["normalization_waring"] += 1
            continue


        if (len(labeled_features.feature) < 5 or
            len(labeled_features.feature) < len(labeled_features.label)):

            warnings.warn(PreprocessWarning("Too short feature",
                                            example.transcription))
            sorted_out["too_few_features"] += 1
            continue

        if (max(labeled_features.feature[:, 1]) > 4 or
            min(labeled_features.feature[:, 1]) < -4):

            warnings.warn(PreprocessWarning("Baseline misalign",
                                            example.transcription))
            sorted_out["baseline_misalign"] += 1
            continue
        result.append(labeled_features)

    return result


def plot_all_steps(strokes):
    _, axes = plt.subplots(5, 2)
    ink = Ink.from_corrupted_stroke(strokes)
    ink, _ = normalized_ink.plot(ink, axes=axes[:, 0])
    strokes_features = InkFeatures(ink)
    strokes_features.plot_all(axes=axes[:, 1])
    plt.show()


def main():
    from tensorflow import flags
    from smartmanuscript.records import from_pdf
    from tensorflow.python.platform.app import flags
    from smartmanuscript.corpus import Corpus
    from .writing import Ink
    import pylab as plt
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("num", 0, "select line number")
    flags.DEFINE_string(
        "source", "sample",
        "[sample, iamondb, ibmub] draw examples from IAMonDo-db-1 or IBM_UB_1 "
        "or use the sample")
    flags.DEFINE_boolean(
        "all_steps", True, "")
    flags.DEFINE_string(
        "file", "../smartmanuscript/data/sample_text/The_Zen_of_Python.pdf",
        "file to show features (either PDF or SVG)")

    if FLAGS.source == "sample":
        corpus = from_pdf(FLAGS.file)
    elif FLAGS.source == "iamondb":
        from smartmanuscript.corpus_iam import IAMonDo
        inkml = IAMonDo("data/IAMonDo-db-1.0/002.inkml")
        corpus = Corpus((segment.transcription, Ink(segment.strokes))
                        for segment in inkml.get_segments(IAMonDo.TEXTLINE))
    elif FLAGS.source == "ibmub":
        from smartmanuscript.corpus_ibm import IBMub
        inkml = IBMub("data/IBM_UB_1/query/OALquery_60.inkml")
        corpus = Corpus((segment.transcription, Ink(segment.strokes))
                        for segment in inkml.get_segments())
    else:
        raise ValueError

    ink = corpus[FLAGS.num][1]
    ink.plot_pylab()
    plt.title("original ink")
    plt.show()

    ink, _ = normalized.plot(ink)

    features = strokes_to_features(ink.strokes)
    InkFeatures.plot(features)



if __name__ == "__main__":
    main()
