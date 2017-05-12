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

from .handwritten_vector_graphic import load as ink_from_file
from .stroke_features import normalized_ink, Transformation, InkPage
from .reader import Reader

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

class SearchablePDF(Reader):

    def __init__(self, model_path):
        super().__init__(model_path)

    def _generate_layer(self, transcription, page, layer):
        surface = PDFSurface(layer, *page.page_size)
        context = Context(surface)
        # context.select_font_face('Georgia')
        context.set_source_rgba(1, 1, 1, 1/256)  # almost invisible
        context.set_font_size(2)
        for line_ink, line_transcription in zip(page.lines, transcription):
            ink, transformation = normalized_ink(line_ink)
            context.save()
            context.transform(Matrix(*(Transformation.translation(0, page.page_size[1]).parameter)))
            context.transform(Matrix(*(Transformation.mirror(0).parameter)))
            context.transform(Matrix(*((~transformation).parameter)))
            context.transform(Matrix(*(Transformation.mirror(0).parameter)))
            HANDWRITING_WIDTH = ink.boundary_box[1]
            TYPEWRITING_WIDTH = context.text_extents(line_transcription)[2]
            context.scale(HANDWRITING_WIDTH/TYPEWRITING_WIDTH, 1)
            context.move_to(0, 0)
            context.show_text(line_transcription)
            context.restore()
        context.stroke()
        context.show_page()
        surface.flush()

    def _add_layer_to_pdf(self, input_pdf, layer, output_pdf):

        transcription_pdf = PdfFileReader(layer)
        original_pdf = PdfFileReader(open(input_pdf, 'rb'))
        page = original_pdf.getPage(0)
        page.mergePage(transcription_pdf.getPage(0))

        output = PdfFileWriter()
        output.addPage(page)

        with open(output_pdf, 'wb') as f:
            output.write(f)
        print("Transcribed manuscript have been generated:", output_pdf)

    def generate(self, input_pdf, output_pdf):
        strokes, page_size = ink_from_file(input_pdf)
        page = InkPage(strokes, page_size)
        transcription = self.recognize_page(strokes).split("\n")
        with tempfile.TemporaryFile() as transcription_layer:
            self._generate_layer(
                transcription, page=page, layer=transcription_layer)
            self._add_layer_to_pdf(input_pdf, transcription_layer, output_pdf)
