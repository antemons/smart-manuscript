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
from svgpathtools import svg2paths
from stroke_features import Ink
from os.path import basename
import subprocess
import re

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def load(filename, transcription_file=None):
    """ load svg file in optionally also the transcription

    Args:
        filename (str): path to the file (either PDF or SVG)
        transcription_file (str): if given, the file with the
            truth transcription of the handwritten note in the svg_file

    Returns:
        strokes if transcription_file is None else (stroke, transcription) with
            strokes (list of array[N,2]): the ink
            transcription (list of str): the transcription
    """

    if filename.endswith('.pdf') or filename.endswith('.PDF'):
        strokes = _read_pdf(filename)
    elif filename.endswith('.svg') or filename.endswith('.SVG'):
        strokes = _read_svg(filename)
    else:
        print("file must be either PDF or SVG")
        exit()

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


def _transform(stroke, matrix):
    """ transform the strokes according to a (svg-)matrix

    Args:
        stroke (float[_, 2]): get transformed column-wise
        matrix (float[6]): transformation matrix
    """
    stroke_e = np.ones([len(stroke), 3])
    stroke_e[:, :2] = stroke
    return np.dot(stroke_e, np.array(matrix).reshape(3, 2))


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
        return any("style" in path and color in path["style"]
                   for color in [r"rgb(0%,0%,0%)", r"#000000"])

    if is_handwritten is None:
        is_handwritten = is_black

    def remove_unit(string):
        unit = re.findall("[a-z]+", string)[-1]
        return string.replace(unit, "")

    svg_dom = minidom.parse(filename)
    element = svg_dom.getElementsByTagName('svg')[0]
    WIDTH = int(remove_unit(element.getAttribute("width")))
    HEIGHT = int(remove_unit(element.getAttribute("height")))
    svg_dom.unlink()

    strokes = Ink([], page_size=(WIDTH, HEIGHT))
    paths, properties = svg2paths(filename)
    for path, property_ in zip(paths, properties):
        if not is_handwritten(property_):
            continue
        polynomials = [segment.poly() for segment in path]
        t = np.linspace(0, 1, 10)
        polygon = np.concatenate([segment(t) for segment in polynomials])
        stroke = np.array([polygon.real, polygon.imag]).transpose()
        m = np.array([[1,  0], [0, -1]])
        if "transform" in property_:
            transform = property_["transform"]
            if "matrix" in transform:
                parameters_str = re.findall("matrix\((.+)\)", transform)[0]
                parameters = [float(p) for p in parameters_str.split(",")]
                stroke = _transform(stroke, parameters)
        stroke = _transform(stroke, [1, 0, 0, -1, 0, HEIGHT])
        strokes.append(stroke)

    return strokes


def _read_txt(txt_file):
    """ read the (truth) transcription from a text-file
    """
    with open(txt_file) as f:
        textlines = [l.replace("\n", "") for l in f.readlines()]
    return textlines


def _pdf_to_svg_tmp(pdf_path):
    pdf_basename = basename(pdf_path)
    svg_basename = pdf_basename.replace(".pdf", ".svg")
    svg_path = "/tmp/" + svg_basename
    # subprocess.call(["inkscape", "-l", svg_path, pdf_path])
    subprocess.call(["/usr/bin/pdftocairo", "-svg", pdf_path, svg_path])
    return svg_path


def _read_pdf(filename, is_handwritten=None):
    svg_filename = _pdf_to_svg_tmp(filename)
    return _read_svg(svg_filename, is_handwritten)

def main():
    """ show sample handwritten notes """
    ink, text = load("sample_text/the_zen_of_python.svg",
                     "sample_text/the_zen_of_python.txt")
    plot_strokes(ink, text)


if __name__ == "__main__":
    main()
