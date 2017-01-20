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

import numpy as np
from cairo import PDFSurface, Context
from reader import Reader
from handwritten_vector_graphic import load as load_file
from tensorflow.python.platform.app import flags
from PyPDF2 import PdfFileWriter, PdfFileReader
import tempfile


__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "file", "sample_text/The_Zen_of_Python.pdf", "pdf-file to transcript")
flags.DEFINE_string(
    "output", "", "pdf-file which is augmented with the transcription")

class SearchablePDF(Reader):

    def __init__(self):
        super().__init__(path=FLAGS.graph_path)

    def _generate_layer(self, transcription, ink, layer):
        surface = PDFSurface(layer, *ink.page_size)
        context = Context(surface)
        context.select_font_face('Sans')
        avg_writing_height = np.average([
            l.boundary_box[3] - l.boundary_box[2] for l in ink.lines])
        TEXT_HEIGHT = avg_writing_height/2
        context.set_font_size(TEXT_HEIGHT)
        context.set_source_rgba(1, 1, 1, 1/256)  # almost invisible
        for line_ink, line_transcription in zip(ink.lines, transcription):
            TEXT_WIDTH = context.text_extents(line_transcription)[2]
            context.save()
            boundary_box = line_ink.boundary_box
            context.move_to(boundary_box[0],
                            ink.page_size[1] - boundary_box[3] + TEXT_HEIGHT)
            context.scale((boundary_box[1]-boundary_box[0])/TEXT_WIDTH, 1)
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

    def generate(self, input_pdf, output_pdf):
        ink = load_file(input_pdf)
        transcription = self.recognize_page(ink).split("\n")
        with tempfile.TemporaryFile() as transcription_layer:
            self._generate_layer(
                transcription, ink=ink, layer=transcription_layer)
            self._add_layer_to_pdf(input_pdf, transcription_layer, output_pdf)





def main():
    output_pdf = FLAGS.output
    if not output_pdf:
        output_pdf = flags.file.replace(".pdf", "_searchable.pdf")
    SearchablePDF().generate(FLAGS.file, output_pdf)


if __name__ == "__main__":
    main()
