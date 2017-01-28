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
from stroke_features import Ink
import train_utils
from utils import cached_property

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


class InkML:
    """ Parse a inkml-file to obtain all words, textlines and the full page

    The words are part of the textlines. It merge single symbols like ",", "."
    with the  precising word.
    """

    def __init__(self, filename):
        """
            Args:
                filename (str): path of the inkml-file
        """
        xml_tree = et.parse(filename)
        self.xml_root = xml_tree.getroot()

    @cached_property
    def traces(self):
        traces = {}
        for xml_node in self.xml_root.findall("trace"):
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

    @cached_property
    def lines(self):
        corpus = train_utils.Corpus()
        for textline_node in _get_all_textline_nodes(self.xml_root):
            trace_refs = []
            transcription = _get_transcription(textline_node)
            if transcription:
                for word_node in _child_nodes_of_type(textline_node, "Word"):
                    if _includes_corrections(word_node):
                        continue
                    trace_refs += _get_trace_refs(word_node)
                strokes = _trace_refs_to_strokes(trace_refs, self.traces)
                corpus.append((transcription, Ink(strokes)))
        return corpus

    @cached_property
    def words(self):
        corpus = train_utils.Corpus()
        for textline_node in _get_all_textline_nodes(self.xml_root):
            is_first_word_of_textline = True
            for word_node in _child_nodes_of_type(textline_node, "Word"):
                if _includes_corrections(word_node):
                    continue
                transcription = _get_transcription(word_node)
                trace_refs = _get_trace_refs(word_node)
                strokes = _trace_refs_to_strokes(trace_refs, self.traces)
                if not transcription or not strokes:
                    continue

                # merge special symbols with the last recogniced word
                if transcription in set(",.:;*+()/!?+-'\"$"):
                    if not is_first_word_of_textline:
                        last = corpus.pop()
                        current = (transcription, Ink(strokes))
                        merged = tuple(a + b for a, b in zip(last, current))
                        corpus.append(merged)
                else:
                    corpus.append((transcription, Ink(strokes)))
                    is_first_word_of_textline = False
        return corpus

    @cached_property
    def page(self):
        page = Ink()
        for _, line in self.lines:
            page += line
        return page


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


def _includes_corrections(word_node):
    for child_trace_ref in word_node.findall("traceView"):
        if _is_type(child_trace_ref, "Correction"):
            return True
    return False


def _get_trace_refs(word_node):
    trace_refs = []
    with_correction = False
    for child_trace_ref in word_node.findall("traceView"):
        if _is_type(child_trace_ref, "Correction"):
            with_correction = True
            continue
        trace_refs.append(child_trace_ref.attrib["traceDataRef"])
    if with_correction:
        assert False
        return None
    return trace_refs


def _trace_refs_to_strokes(trace_refs, traces):
    return [traces[trace_ref.replace("#", "")] for trace_ref in trace_refs]


def _get_all_textline_nodes(xml_root):
    for document_node in _child_nodes_of_type(xml_root, "Document"):
        for textblock_node in _child_nodes_of_type(document_node):
            for textline_node in _child_nodes_of_type(textblock_node,
                                                      "Textline"):
                yield textline_node


def main():
    FLAGS = flags.FLAGS
    flags.DEFINE_string("inkml_file", "data/IAMonDo-db-1.0/003.inkml",
                        "path to an inkml-file")
    inkml = InkML(FLAGS.inkml_file)
    inkml.page.plot_pylab()
    plt.show()

    for _, line in inkml.lines:
        line.plot_pylab()
        plt.show()


if __name__ == "__main__":
    main()
