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

from xml.dom import minidom
import pylab as plt
import numpy as np
from svgpathtools import parse_path
from stroke_features import Ink

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def load(svg_file, transcription_file=None):
    """ load svg file in optionally also the transcription

    Args:
        svg_file (str): path to the svg_file
        transcription_file (str): if given, the file with the
            truth transcription of the handwritten note in the svg_file

    Returns:
        strokes if transcription_file is None else (stroke, transcription) with
            strokes (list of array[N,2]): the ink
            transcription (list of str): the transcription
    """

    strokes = _read_svg(svg_file)
    if transcription_file is None:
        return strokes
    else:
        transcription = _read_txt(transcription_file)
        return strokes, transcription


def plot_strokes(strokes, lines=None):
    if lines is not None:
        for line in lines:
            print(line)
    for stroke in strokes:
        plt.plot(stroke[:, 0], stroke[:, 1], '-')
    plt.axes().set_aspect('equal')
    plt.show()


def _read_svg(filename, is_handwritten=None):
    """ Read all strokes from the svg file and save it together with the index

    PDF to SVG: pdftocairo -svg filename.pdf

    Args:
        filename: name (and path) of the svg-file
        is_handwritten: function returning True if the path is a handwriting
            or None, in the later case path is treated as handwritten if and
            only if it is black

    Retruns:
        strokes
    """

    def is_black(path):
        return any(color in path.getAttribute("style")
                   for color in [r"rgb(0%,0%,0%)", r"#000000"])

    if is_handwritten is None:
        is_handwritten = is_black

    svg_dom = minidom.parse(filename)

    def remove_unit(string):
        return ''.join([x for x in string if x.isdigit()])
    element = svg_dom.getElementsByTagName('svg')[0]
    WIDTH = int(remove_unit(element.getAttribute("width")))
    HEIGHT = int(remove_unit(element.getAttribute("height")))

    path_strings = (
        path.getAttribute('d')
        for path in svg_dom.getElementsByTagName('path')
        if is_handwritten(path))

    strokes = Ink([], page_size=(WIDTH, HEIGHT))
    for path_string in path_strings:
        path = parse_path(path_string)
        polynomials = [segment.poly() for segment in path]
        t = np.linspace(0, 1, 10)
        polygon = np.concatenate([segment(t) for segment in polynomials])
        stroke = np.array([polygon.real, polygon.imag]).transpose()
        strokes.append(stroke)
    svg_dom.unlink()

    return strokes


def _read_txt(txt_file):
    """ read the (truth) transcription from a text-file
    """
    with open(txt_file) as f:
        textlines = [l.replace("\n", "") for l in f.readlines()]
    return textlines


def main():
    """ show sample handwritten notes """
    ink, text = load("sample_text/the_zen_of_python.svg",
                     "sample_text/the_zen_of_python.txt")
    plot_strokes(ink, text)


if __name__ == "__main__":
    main()
