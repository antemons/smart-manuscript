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

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


from tensorflow.python.platform.app import flags
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from reader import Reader
try:
    from xinput import operate_xinput_device, MODE_ENABLE, MODE_DISABLE
    XINPUT_IS_IMPORTED = True
except ImportError:
    XINPUT_IS_IMPORTED = False


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "num_proposals", 3, "How many recognation proposals are shown")
flags.DEFINE_string(
    "touchscreen_name", "maXTouch Digitizer",
    "Name of the touchscreen to deactivate (see xinput)")
flags.DEFINE_boolean(
    "deactivate_touchscreen", False, "Whether to deactivate touchscreen")


class HandwrittenInput(Gtk.Window):
    """ Main window of the handwriting-recognation application
    """

    def __init__(self, graph_path):
        """
        Args:
            graph_path (str): path to the trained tensorflow-graph
                and -variables
        """

        super(HandwrittenInput, self).__init__()
        self.recognizer = Reader(path=graph_path)
        self.strokes = []   # on-line writing information, grouped by strokes

        self.set_title("Handwriting")
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
        for _ in range(FLAGS.num_proposals+1):
            button = Gtk.Button(label="")
            button.connect("clicked", self.on_button_clicked)
            self.box_buttons.pack_start(button, True, True, 0)
            self.buttons.append(button)

        self.clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.clipboard.set_text("", -1)
        if XINPUT_IS_IMPORTED and FLAGS.deactivate_touchsceen:
            operate_xinput_device(MODE_DISABLE, FLAGS.touchscreen_name)

    def on_button_clicked(self, widget):
        """ clear board, hide suggestion buttons
        """
        self.strokes = []
        self.canvas.queue_draw()
        text = self.clipboard.wait_for_text()
        if text is None:
            text = ""
        self.clipboard.set_text(text + " " + widget.get_label(), -1)
        for button in self.buttons:
            button.hide()

    def quit(self, *_):
        """ quit application
        """
        self.recognizer.session.close()
        if XINPUT_IS_IMPORTED and FLAGS.deactivate_touchsceen:
            operate_xinput_device(MODE_ENABLE, FLAGS.touchscreen_name)
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
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.set_line_width(.5)

        cr.move_to(0, 100)
        cr.line_to(1000, 100)
        cr.stroke()

        cr.move_to(0, 200)
        cr.line_to(1000, 200)
        cr.stroke()

        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(2)

        for stroke in self.strokes:
            for i, point in enumerate(stroke):
                if len(stroke) == 1:
                    radius = 2
                    cr.arc(point[0], point[1],
                           radius, 0, 2.0 * np.pi)
                    cr.fill()
                    cr.stroke()
                elif i != 0:
                    cr.move_to(stroke[i-1][0], stroke[i-1][1])
                    cr.line_to(point[0], point[1])
                    cr.stroke()

    def recognize(self):
        """ make and show transcription suggestions
        """
        strokes = []
        for stroke in self.strokes:
            stroke = np.array(stroke)
            stroke[:, 1] = - stroke[:, 1] + 200
            stroke = stroke / 100
            strokes.append(stroke)
        predictions = self.recognizer.recognize(
            strokes, num_proposals=FLAGS.num_proposals)
        for button, prediction in zip(self.buttons, predictions):
            button.set_label(prediction)
            button.show()
        self.buttons[-1].show()


def main():
    """ start the application
    """
    HandwrittenInput(graph_path=FLAGS.graph_path)
    Gtk.main()

if __name__ == "__main__":
    main()
