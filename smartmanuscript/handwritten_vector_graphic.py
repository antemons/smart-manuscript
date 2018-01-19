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

from xml.dom import minidom
import pylab as plt
import numpy as np
from svgpathtools import svg2paths
from os.path import basename
import subprocess
import re
from .stroke_features import InkPage

from .utils import Transformation


__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def load(filename):
    """ load svg file in optionally also the transcription

    Args:
        filename (str): path to the file (either PDF or SVG)

    Returns:
        tuple (strokes, page_size):
            strokes (list of array[N,2]): the ink
            page_size: tuple (width, height) are the dimensions
    """

    if filename.endswith('.pdf') or filename.endswith('.PDF'):
        strokes, page_size = _read_pdf(filename)
    elif filename.endswith('.svg') or filename.endswith('.SVG'):
        strokes, page_size = _read_svg(filename)
    else:
        raise Exception("file must be either PDF or SVG")
    return strokes, page_size


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
    WIDTH = float(remove_unit(element.getAttribute("width")))
    HEIGHT = float(remove_unit(element.getAttribute("height")))
    svg_dom.unlink()

    strokes = []
    paths, properties = svg2paths(filename)
    for path, property_ in zip(paths, properties):
        if not is_handwritten(property_):
            continue
        polynomials = [segment.poly() for segment in path]
        t = np.linspace(0, 1, 10)
        polygon = np.concatenate([segment(t) for segment in polynomials])
        stroke = np.array([polygon.real, polygon.imag]).transpose()
        if "transform" in property_:
            transform = property_["transform"]
            if "matrix" in transform:
                parameters_str = re.findall(r"matrix\((.+)\)", transform)[0]
                parameters = np.array(
                    [float(p) for p in parameters_str.split(",")])
                stroke = Transformation(parameters[[0, 1, 4, 2, 3, 5]]) @ stroke
        # stroke = Transformation([1, 0, 0, 0, -1, HEIGHT]) @ stroke
        strokes.append(stroke)
    return strokes, (WIDTH, HEIGHT)


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


def main(svg_filename):
    """ show sample handwritten notes """

    strokes, page_size = load(svg_filename)
    ink = InkPage(strokes, page_size)
    ink.plot_pylab()
    plt.show()

    for line in ink.lines:
        line.plot_pylab()
        plt.show()


if __name__ == "__main__":
    from tensorflow.python.platform.app import flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "file", "sample_text/The_Zen_of_Python.pdf",
        "file to show features (either PDF or SVG)")
    main(FLAGS.file)
