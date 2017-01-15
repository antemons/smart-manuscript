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

import xml.etree.ElementTree as et
import numpy as np
import pylab as plt
import pickle
from tensorflow.python.platform.app import flags

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

FLAGS = flags.FLAGS
flags.DEFINE_integer("min_words_per_line", 3,
                     "num of word required to accept as a textline")


def plot_ink(ink, transcription=None, axes=None):
    if transcription is not None:
        axes.set_title(transcription)
    for stroke in ink:
        axes.plot(stroke[:, 0], stroke[:, 1], '-')
    axes.set_aspect('equal')


def _to_float(text):
    text = text.replace("'", " ")
    text = text.replace('"', " ")
    text = text.replace('-', " -")  # to separate via " "
    numbers = text.split(" ")
    return [float(number) for number in numbers if number]


def _get_annotation(xml_node):
    return {child_anno.attrib["type"]: child_anno.text
            for child_anno in xml_node.findall("annotation")}


def _is_type(xml_node, type_):
    return _get_annotation(xml_node).get("type") == type_


def _child_nodes_of_type(xml_node, type_=None):
    for child_node in xml_node.findall("traceView"):
        if type_ is None or _is_type(child_node, type_):
            yield child_node


def _get_transcription(xml_node):
    t = _get_annotation(xml_node).get("transcription")
    if t is not None:
        t = t.replace(" .", ".")
        t = t.replace(" ;", ";")
        t = t.replace(" ,", ",")
        t = t.replace("`", "'")
        t = t.replace(b'\xc2\xb4'.decode(), "'")
    return t


def _rotate(strokes, rot_angle):
    rotations_matrix = np.array(
            [[np.cos(rot_angle), -np.sin(rot_angle)],
             [np.sin(rot_angle),  np.cos(rot_angle)]])
    strokes_rotated = []
    for stroke in strokes:
        strokes_rotated.append(np.dot(rotations_matrix, stroke.T).T)
    return strokes_rotated


def _get_ink(word_node, traces):
    trace_refs = []
    with_correction = False
    for child_trace_ref in word_node.findall("traceView"):
        if _is_type(child_trace_ref, "Correction"):
            with_correction = True
            continue
        trace_refs.append(child_trace_ref.attrib["traceDataRef"])
    if with_correction:
        return None
    return [traces[trace_ref.replace("#", "")]
            for trace_ref in trace_refs]


def _get_all_textline_nodes(xml_root):
    for document_node in _child_nodes_of_type(xml_root, "Document"):
        for textblock_node in _child_nodes_of_type(document_node):
            for textline_node in _child_nodes_of_type(textblock_node,
                                                      "Textline"):
                yield textline_node


def _get_traces(xml_root):
    traces = {}
    for xml_node in xml_root.findall("trace"):
        trace_id = xml_node.get('{http://www.w3.org/XML/1998/namespace}id')
        text = xml_node.text.split(",")
        stroke = []
        for i, entry in enumerate(text):
            if i == 0:
                x, y, t, _ = _to_float(entry)
            if i == 1:
                vx, vy, dt, _ = _to_float(entry)
                x += vx
                y += vy
            if i > 1:
                ax, ay, ddt, _ = _to_float(entry)
                vx += ax
                vy += ay
                x += vx
                y += vy
            stroke.append(np.array([-x, y], float))
        traces[trace_id] = np.array(stroke)
    return traces


def read_inkml_file(filename):
    """ Parse a inkml-file to obtain all single words and textlines

    The words are part of the textlines. It merge single symbols like ",", "."
    with the  precising word.

    Args:
        filename (str): path of the inkml-file

    Returns:
        tuple (word_corpus, text_corpus): each corpus is a nested list of
            tuples (transcription (str), ink (list of arrays))
    """
    xml_tree = et.parse(filename)
    xml_root = xml_tree.getroot()

    traces = _get_traces(xml_root)

    textlines = []
    words = []

    for textline_node in _get_all_textline_nodes(xml_root):
        new_textline_ink = []
        for word_node in _child_nodes_of_type(textline_node, "Word"):
            word_ink = _get_ink(word_node, traces)
            word_transcription = _get_transcription(word_node)
            if word_transcription and word_ink:
                # merge special symbols with the last recogniced word
                if (word_transcription in set(",.:;*+()/!?+-'\"$") and
                        words and new_textline_ink):
                    last_word = words[-1]
                    extension = (_get_transcription(word_node), word_ink)
                    words[-1] = tuple(a + b for a, b in
                                      zip(last_word, extension))
                else:
                    words.append(
                        (_get_transcription(word_node), word_ink))
                new_textline_ink.extend(word_ink)
        textline_transcription = _get_transcription(textline_node)
        if textline_transcription:
            textlines.append((textline_transcription, new_textline_ink))

    word_direction = np.array([0, 0], float)
    for _, strokes in words:
        word_direction += strokes[-1][-1] - strokes[0][0]

    angle = np.arctan2(*word_direction[::-1])
    if -np.pi/4 <= angle and angle < np.pi/4:
        rot_angle = 0  # direction = "normal"
    elif np.pi/4 <= angle and angle < 3*np.pi/4:
        rot_angle = -np.pi/2  # direction = "right"
    elif 3*np.pi/4 <= angle or angle < - 3*np.pi/4:
        rot_angle = np.pi  # direction = "inverted":
    elif - 3*np.pi/4 <= angle and angle < - np.pi/4:
        rot_angle = np.pi/2  # direction = "left"
    else:
        assert False

    words = [(transcription, _rotate(strokes, rot_angle))
             for transcription, strokes in words]
    textlines = [(transcription, _rotate(strokes, rot_angle))
                 for transcription, strokes in textlines]

    return words, textlines
