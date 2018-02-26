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

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

from tensorflow.python.platform.app import flags
import numpy as np
from scipy import interpolate
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import cairo

try:
    from xinput import operate_xinput_device, MODE_ENABLE, MODE_DISABLE
    XINPUT_IS_IMPORTED = True
except ImportError:
    XINPUT_IS_IMPORTED = False

from .reader import Reader


def read_flags():
    flags.DEFINE_integer(
        "num_proposals", 3, "How many recognation proposals are shown")
    flags.DEFINE_string(
        "deactivate_touchscreen", "",  # e.g. maXTouch Digitizer
        "When given this touchscreen is deactivated (for name see xinput)")
    dafault_model_path = os.path.join(
        os.path.dirname(__file__), 'data', 'model', 'model.ckpt')
    flags.DEFINE_string(
        "model_path", dafault_model_path, "path of the model")
    return flags.FLAGS


def stroke_to_spline(stroke):
    if len(stroke) == 1:
        t = 4 * [0] + 4 * [1]
        c = np.array(stroke)[0][:, None] * np.ones(4)[None, :]
        spline = (t, c, 3)
    elif len(stroke) == 2:
        t = 4 * [0] + 4 * [1]
        c = np.stack([stroke[0],
                      stroke[1],
                      stroke[0],
                      stroke[1]])
        c = list(c.transpose())
        spline = (t, c, 3)
    elif len(stroke) == 3:
        (t, c, k), _ = interpolate.splprep(np.array(stroke).transpose(), u=None, k=2, s=0)
        c = np.array(c)
        new_c = np.stack([c[:, 0],
                          1/3 * c[:, 0] + 2/3 * c[:, 1],
                          2/3 * c[:, 1] + 1/3 * c[:, 2],
                          c[:, 2]])
        new_c = list(new_c.transpose())
        new_t = 4 * [0] + 4 * [1]
        spline = (new_t, new_c, 3)
    else:
        spline, u = interpolate.splprep(
            np.array(stroke).transpose(), k=3, s=len(stroke) * 10)
        for knot in 3 * list(spline[0][4:-4]):
            spline = interpolate.insert(knot, spline)
    control_points = np.array(spline[1]).transpose()
    control_points = control_points[:max(len(control_points)-4, 4)]
    return control_points


class HandwrittenInput(Gtk.Window):
    """ Window to record the handwriting and show its recognation
    """

    def __init__(self, model_path, num_proposals=3,
                 deactivate_touchscreen=None):
        """
        Args:
            graph_path (str): path to the trained tensorflow-graph
                and -variables
        """

        super(HandwrittenInput, self).__init__()
        self._num_proposals = num_proposals
        self._deactivate_touchscreen = deactivate_touchscreen
        self.recognizer = Reader(model_path=model_path)
        self.strokes = []   # on-line writing information, grouped by strokes
        self.splines = []

        self.set_title("Smart Manuscript Writer")
        self.resize(1000, 300)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", self.quit)
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(self.box)

        self.canvas = Gtk.DrawingArea()
        self.canvas.set_size_request(1000, 250)
        self.canvas.connect("draw", self.on_draw)
        self.canvas.connect("button-press-event", self.on_button_press)
        self.canvas.connect("button-release-event", self.on_button_release)
        self.canvas.connect("motion-notify-event", self.on_mouse_move)
        self.canvas.set_events(self.canvas.get_events() |
                               Gdk.EventMask.BUTTON_MOTION_MASK |
                               Gdk.EventMask.BUTTON_PRESS_MASK |
                               Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.box.pack_start(self.canvas, True, True, 0)

        self.box_buttons = Gtk.Box(spacing=6)
        self.box.pack_start(self.box_buttons, True, True, 0)
        self.show_all()
        self.buttons = []
        for _ in range(self._num_proposals + 1):
            button = Gtk.Button(label="")
            button.connect("clicked", self.on_button_clicked)
            self.box_buttons.pack_start(button, True, True, 0)
            self.buttons.append(button)

        self.clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.clipboard.set_text("", -1)
        if XINPUT_IS_IMPORTED and self._deactivate_touchscreen:
            operate_xinput_device(MODE_DISABLE, self._deactivate_touchscreen)

    def draw_auxiliary_lines(self, cr):
        cr.set_source_rgb(.5, .5, .5)
        cr.set_line_width(.5)

        cr.move_to(0, 100)
        cr.line_to(1000, 100)

        cr.move_to(0, 200)
        cr.line_to(1000, 200)

        cr.stroke()

    def on_button_clicked(self, widget):
        """ clear board, hide suggestion buttons
        """
        self.strokes = []
        self.splines = []
        self.canvas.queue_draw()
        text = self.clipboard.wait_for_text()
        if text is None:
            text = ""
        self.clipboard.set_text(text + " " + widget.get_label(), -1)
        print(widget.get_label(), end=" ", flush=True)
        for button in self.buttons:
            button.hide()

    def quit(self, *_):
        """ quit application
        """
        self.recognizer.session.close()
        if XINPUT_IS_IMPORTED and self._deactivate_touchscreen:
            operate_xinput_device(MODE_ENABLE, self._touchscreen_name)
        Gtk.main_quit()

    def on_button_press(self, _, event):
        """ store starting point for new stroke, hide suggestion buttons
        """

        self.strokes.append([[event.x, event.y]])
        self.canvas.queue_draw()
        for button in self.buttons:
            button.hide()

    def on_button_release(self, *_):
        """ make transcription suggestions
        """
        self.splines.append(stroke_to_spline(self.strokes[-1]))
        self.recognize()

    def on_mouse_move(self, _, event):
        """ add new point to the strokes
        """

        if (event.x, event.y) != tuple(self.strokes[-1][-1]):
            point = [event.x, event.y]
            self.strokes[-1].append(point)
            self.canvas.queue_draw()

    def on_draw(self, _, cr):
        """ redraw the ink
        """

        self.draw_auxiliary_lines(cr)

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_BEVEL)

        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(2)

        for spline in self.splines:

            cr.move_to(*spline[0])
            for control_1, control_2, knot in zip(spline[1:][::4],
                                                  spline[2:][::4],
                                                  spline[3:][::4]):
                cr.curve_to(*control_1, *control_2, *knot)

        for stroke in self.strokes[len(self.splines):]:
            cr.move_to(*stroke[0])
            for dot in stroke:
                cr.line_to(*dot)
        cr.stroke()

    def recognize(self):
        """ suggest transcrition and show it
        """
        strokes = []
        for stroke in self.strokes:
            stroke = np.array(stroke)
            stroke[:, 1] = - stroke[:, 1] + 200
            stroke = stroke / 100
            strokes.append(stroke)
        top_prediction, predictions, probabilities = self.recognizer.recognize_line(strokes)
        for button, prediction in zip(self.buttons, predictions):
            button.set_label(prediction)
            button.show()
        self.buttons[-1].show()


def main():
    """ start the application
    """


    FLAGS = read_flags()

    HandwrittenInput(
        model_path=FLAGS.model_path,
        num_proposals=FLAGS.num_proposals,
        deactivate_touchscreen=FLAGS.deactivate_touchscreen)
    Gtk.main()

if __name__ == "__main__":
    main()
