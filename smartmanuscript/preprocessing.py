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
from collections import namedtuple
import warnings

from .stroke_features import strokes_to_features, NormalizationWarning

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
    sorted_out = dict(too_few_words=0,
                      too_few_letters=0,
                      normalization_waring=0,
                      symbol_not_in_alphabet=0,
                      too_few_features=0,
                      baseline_misalign=0)
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
