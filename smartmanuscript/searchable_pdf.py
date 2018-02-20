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

import numpy as np
from cairocffi import PDFSurface, Context, Matrix
from tensorflow.python.platform.app import flags
from PyPDF2 import PdfFileWriter, PdfFileReader
import tempfile

import time
from .handwritten_vector_graphic import load as ink_from_file
from . import writing
from .utils import Transformation
from .reader import Reader

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

class TextOverlay:

    def __init__(self, file_, width, height):
        self._file = file_
        self.surface = PDFSurface(self._file, width, height)
        self.context = Context(self.surface)
        self.context.set_source_rgba(1, 1, 1, 1/256)  # almost invisible
        self.context.set_font_size(2)

    def __enter__(self):
        return self

    def add_page(self, transcription, page):
        self.context.set_font_size(2)
        for line_ink, line_transcription in zip(page.lines, transcription):
            ink, transformation = writing.normalized(line_ink)
            self.context.save()
            self.context.transform(Matrix(*(Transformation.translation(
                0, page.page_size[1]).parameter)))
            self.context.transform(Matrix(*(Transformation.mirror(0).parameter)))
            self.context.transform(Matrix(*((~transformation).parameter)))
            self.context.transform(Matrix(*(Transformation.mirror(0).parameter)))
            HANDWRITING_WIDTH = ink.boundary_box[1]
            TYPEWRITING_WIDTH = self.context.text_extents(line_transcription)[2]
            self.context.scale(HANDWRITING_WIDTH / TYPEWRITING_WIDTH, 1)
            self.context.move_to(0, 0)
            self.context.show_text(line_transcription)
            self.context.restore()
        self.context.stroke()
        self.context.show_page()

    def __exit__(self, *_):
        self.surface.finish()
        self._file.flush()


def overlay_pdf(input_name, overlay_file, output_name):
    layer_original = PdfFileReader(open(input_name, "rb"))
    layer_text = PdfFileReader(overlay_file)

    writer = PdfFileWriter()
    for page, overlay in zip(layer_original.pages, layer_text.pages):
        page.mergePage(overlay)
        writer.addPage(page)
    with open(output_name, "wb") as f:
        writer.write(f)


class SearchablePDF(Reader):

    def __init__(self, model_path):
        super().__init__(model_path)

    def generate(self, input_pdf, output_pdf):

        with open(input_pdf, 'rb') as input_pdf_file:
            input_pdf_reader = PdfFileReader(input_pdf_file)

            ink_pages = []
            for page in input_pdf_reader.pages:
                page_writer = PdfFileWriter()
                page_writer.addPage(page)
                with tempfile.NamedTemporaryFile(suffix='.pdf') as page_file:
                    # Todo(dv): use TemporaryFile and stdin for pdftocairo
                    page_writer.write(page_file)
                    page_file.flush()
                    ink_pages.append(writing.InkPage(*ink_from_file(page_file.name)))

        with tempfile.TemporaryFile() as text_overlay_file:
            with TextOverlay(text_overlay_file, *ink_pages[0].page_size) as overlay:
                for page in ink_pages:
                    transcription = self.recognize_page(page.strokes).split("\n")
                    overlay.add_page(transcription, page)
            overlay_pdf(input_pdf, text_overlay_file, output_pdf)
