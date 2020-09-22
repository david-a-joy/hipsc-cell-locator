#!/usr/bin/env python

""" Locate single cells

Annotate the files with points in a nested layout:

.. code-block:: bash

    $ ./cell_locator.py \\
        --sel-mode point \\
        --layout nested \\
        -r /path/to/data

Annotate tiles within a certain range:

.. code-block:: bash

    $ ./cell_locator.py \\
        --sel-mode point \\
        --layout nested \\
        -r /path/to/data \\
        --min-timepoint 5 \\
        --max-timepoint 15

Ignore already annotated tiles:

.. code-block:: bash

    $ ./cell_locator.py \\
        --sel-mode point \\
        --layout nested \\
        -r /path/to/data \\
        --skip-tagged

Putting it all together, here's what I use to annotate 20x inverted images:

.. code-block:: bash

    $ ./cell_locator.py \\
        --sel-mode point \\
        --layout nested \\
        -r /data/Experiment/2017-03-03 \\
        --max-timepoint 12 \\
        --skip-tagged

API Documentation
-----------------

"""

# Imports

# Standard lib
import sys
import time
import pathlib
import tkinter
import platform

# 3rd party
import numpy as np

from PIL import Image

import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Our own imports
from model import load_selection_db
import gui

# Constants

UNAME = platform.system().lower()

MARKERSIZE = 12

NIGHT_MODE = True

SEL_CLASS_COLORS = {
    1: 'red',
    2: 'orange',
    3: 'gold',
    4: 'green',
    5: 'blue',
    6: 'indigo',
    7: 'violet',
    8: 'magenta',
    9: 'cyan',
    0: 'darkgreen',
}
SEL_CLASS_DEFAULT = 1

SEL_MODE_DEFAULT = 'point'


# Functions


def find_all_images(rootdir):
    """ Find all the images under rootdir """
    return (p for p in sorted(rootdir.iterdir())
            if p.is_file() and p.suffix in ('.tif', '.png'))


# Classes


class Crosshair(object):
    """ Draw a crosshair on the plot """

    def __init__(self, mode='off', window=None):

        self.set(mode)

        self.window = window

        self.should_draw = False
        self.cur_cross = None
        self.cur_region = None

    def set(self, mode):

        if mode in ('on', True):
            mode = True
        elif mode in ('off', False):
            mode = False
        else:  # Toggle
            mode = not self.should_draw
        self.should_draw = mode

    def add(self, x=0.0, y=0.0):
        """ Add a cross at these coordinates """

        bbox = self.window.fig.get_window_extent().bounds
        x0, y0, width, height = bbox

        horz_line = Line2D([x0, x0+width], [y, y], linewidth=2, linestyle='--', color=(0.6, 0.6, 0.6))
        vert_line = Line2D([x, x], [y0, y0+height], linewidth=2, linestyle='--', color=(0.6, 0.6, 0.6))

        horz_line.set_animated(True)
        vert_line.set_animated(True)

        self.cur_region = self.window.canvas.copy_from_bbox(self.window.fig.bbox)

        self.window.fig.lines.append(horz_line)
        self.window.fig.lines.append(vert_line)

        self.cur_cross = horz_line, vert_line

    def remove(self):
        if self.cur_cross is None:
            return

        horz_line, vert_line = self.cur_cross
        self.window.canvas.restore_region(self.cur_region)

        self.cur_cross = None
        self.cur_region = None

        horz_line.set_animated(False)
        vert_line.set_animated(False)

        self.window.fig.lines.remove(horz_line)
        self.window.fig.lines.remove(vert_line)

    def update(self, x, y):
        if self.cur_cross is None:
            return

        horz_line, vert_line = self.cur_cross

        horz_line.set_ydata([y, y])
        vert_line.set_xdata([x, x])

        self.window.canvas.restore_region(self.cur_region)
        self.window.fig.draw_artist(horz_line)
        self.window.fig.draw_artist(vert_line)
        self.window.canvas.blit(self.window.fig.bbox)


class ImageTagger(object):

    def __init__(self,
                 sel_class=SEL_CLASS_DEFAULT,
                 sel_mode=SEL_MODE_DEFAULT):

        # Use some file location lookup to find the data tables
        if getattr(sys, 'frozen', False):
            thisfile = pathlib.Path(sys.executable).resolve()
        elif __file__:
            thisfile = pathlib.Path(__file__).resolve()

        rootdir = thisfile.parent / 'data'
        if not rootdir.is_dir():
            raise OSError('Cannot find root directory: {}'.format(rootdir))

        self.rootdir = rootdir
        self.imagedir = rootdir.parent / 'images'
        self.dbfile = rootdir / 'RegionDB.sqlite3'
        print('Rootdir: {}'.format(self.rootdir))
        print('DBfile: {}'.format(self.dbfile))

        self.db = load_selection_db(self.dbfile)

        self.records = list(find_all_images(rootdir))
        self.annotated_records = self.db.find_annotated_records()
        self.cur_record = None
        self.cur_record_idx = 0
        self.cur_record_start = None

        self.cur_region = None
        self.cur_x0 = None
        self.cur_y0 = None
        self.cur_sel_class = sel_class
        self.cur_sel_mode = sel_mode

        self.display_mode = 'normal'
        self.help_objects = []
        self.encourage_objects = []

        self.dpi = None

        self.cur_cross = Crosshair(mode='off', window=self)

        self.shape_manager = gui.ShapeManager(window=self)

        self.points = {}
        self.rects = {}

    @property
    def cur_filepath(self):
        if self.cur_record is None:
            return None
        return str(self.cur_record.relative_to(self.rootdir))

    @property
    def figsize(self):
        return self.fig.get_size_inches()

    @property
    def markersize(self):
        # Scale the marker size
        return round(max([self.figsize[0] / 38.4, self.figsize[1] / 21.3])*MARKERSIZE)

    def get_color(self, sel_class=None):
        """ Get the color for the current box """
        if sel_class is None:
            sel_class = self.cur_sel_class
        return SEL_CLASS_COLORS[sel_class]

    def load_window(self):
        """ Create the figure, axis, and canvas """

        window = plt.get_current_fig_manager().window
        screen_x, screen_y = None, None

        # FIXME: Make this work with non-TkAgg backends
        screen_x, screen_y = window.wm_maxsize()
        print('Screen: {}x{}'.format(screen_x, screen_y))
        self.dpi = int(mpl.rcParams['figure.dpi'])
        print('DPI: {}'.format(self.dpi))

        figsize = (screen_x / self.dpi, screen_y / self.dpi)

        # Force the window to be as fullscreen as we can
        self.fig = plt.gcf()
        self.fig.set_size_inches(figsize[0], figsize[1])
        self.fig.canvas.set_window_title('Cell Locator')
        try:
            window.state('zoomed')
        except tkinter.TclError:
            window.state('normal')

        plt.draw()

        self.ax = self.fig.gca()
        self.canvas = self.fig.canvas
        if NIGHT_MODE:
            self.fig.patch.set_facecolor('black')
        # Disable the default shortcut keys
        self.canvas.mpl_disconnect(self.canvas.manager.key_press_handler_id)

        self.ax_img = None

    def load_image(self, step=1):
        """ Load the next image """

        self.cur_record_idx = (self.cur_record_idx + step) % len(self.records)
        self.cur_record = self.records[self.cur_record_idx]
        self.cur_record_start = time.monotonic()

        img = Image.open(str(self.cur_record))
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        assert img.ndim == 3
        assert img.shape[2] == 3

        self.cur_image = img

        if self.ax_img is None:
            self.ax_img = self.ax.imshow(self.cur_image, aspect='equal')
        else:
            rows, cols = img.shape[:2]
            self.ax_img.set_data(self.cur_image)
            self.ax_img.set_extent((0, cols, rows, 0))

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        plt.tight_layout()

    def load_bounds(self):
        """ Calculate absolute bounds """

        # This one seems to actually follow the cells
        ax_tight_bbox = self.ax.get_tightbbox(self.canvas)

        im_bbox = ((ax_tight_bbox.x0, ax_tight_bbox.y0),
                   (ax_tight_bbox.x1, ax_tight_bbox.y1))
        # print('im_bbox: {}'.format(im_bbox))

        # We have to correct for aspect ratio too?
        aspect = self.ax.get_data_ratio()
        self.shape_manager.load_axis_bounds(im_bbox, aspect)

    def connect(self):
        self.cid_close = self.canvas.mpl_connect(
            'close_event', self.on_window_close)

        self.cid_press = self.canvas.mpl_connect(
            'button_press_event', self.on_mouse_press)

        self.cid_keypress = self.canvas.mpl_connect(
            'key_press_event', self.on_key_press)

        self.cid_resize = self.canvas.mpl_connect(
            'resize_event', self.on_resize)

    def clear_shapes(self, draw=True):
        """ Clear all the rects """

        self.shape_manager.on_clear_all()

        if draw:
            self.canvas.draw()

    def load_points(self):
        """ Load the points from the database """

        points = self.db.find_points(self.cur_filepath)

        for p_class, px, py in points:
            self.shape_manager.on_point_complete(
                p_class, px, py)
        self.canvas.draw()

    def save_points(self):
        """ Save the selected points to the database """

        points = self.shape_manager.points

        classes = [s.sel_class for s in points]
        points = [(s.x, s.y) for s in points]
        self.db.set_points(self.cur_filepath,
                           classes=classes,
                           points=points)
        self.db.add_view(self.cur_filepath,
                         self.cur_record_start,
                         time.monotonic())

    def draw_point(self, point_obj):
        """ Draw a single point """

        if point_obj in self.points:
            return

        p_class, px, py = point_obj

        fx, fy = self.shape_manager.warp_to_figure(
            px, py)
        p_color = self.get_color(p_class)

        bbox = self.fig.get_window_extent().bounds
        x0, y0, _, _ = bbox

        line = Line2D([fx+x0], [fy+y0], markersize=self.markersize,
                      linestyle='-', marker='o', color=p_color)
        self.fig.lines.append(line)

        self.points[point_obj] = line

    def remove_point(self, point_obj):
        """ Remove a single point """

        if point_obj not in self.points:
            return

        line = self.points[point_obj]

        self.fig.lines.remove(line)
        del self.points[point_obj]

    def load_last_index(self):
        """ Work out the index of the last image loaded """
        last_record = self.db.get_last_viewed()
        if last_record is not None:
            last_record = last_record[0]

        last_index = [i for i, r in enumerate(self.records) if r.name == last_record]
        if len(last_index) != 1:
            cur_record_idx = 0
        else:
            cur_record_idx = last_index[0]
        self.cur_record_idx = cur_record_idx

    def load_next_record(self, step=1):
        """ Load the next image tile """

        # Reset
        self.points = {}
        self.cur_record = None

        self.shape_manager.on_reset_actions()

        self.load_image(step=step)

        if self.cur_record is None:
            print('No more records to process...')
            plt.close()
            return

        self.load_bounds()
        self.load_points()
        self.canvas.draw()

    def maybe_draw_encouragement(self):
        """ Try to draw a screen to encourage the user """

        if self.display_mode != 'normal':
            return

        # See if we've added any new annotated images since last save
        annotated_records = self.db.find_annotated_records()
        new_records = annotated_records - self.annotated_records
        if new_records == set():
            return

        milestones = [float(p.stem) for p in self.imagedir.iterdir()
                      if p.suffix == '.jpg']

        # Cool, now did we cross a milestone
        pct_new = len(annotated_records) / len(self.records) * 100
        pct_old = len(self.annotated_records) / len(self.records) * 100

        print('{:0.1f}% done!'.format(pct_new))

        new_milestone = None
        for milestone in milestones:
            if pct_new >= milestone and pct_old < milestone:
                new_milestone = milestone
                break

        self.annotated_records = annotated_records
        if new_milestone is None:
            return

        image_file = self.imagedir / '{:d}.jpg'.format(int(round(new_milestone)))
        img = np.asarray(Image.open(str(image_file)))
        rows, cols, _ = img.shape

        # Okay, in here we need to draw an overlay
        self.display_mode = 'encouragement' if new_milestone < 100 else 'finished'
        encourage_objects = []

        bbox = self.fig.get_window_extent().bounds
        x0, y0, x1, y1 = bbox

        xct = (x1 + x0)/2
        yct = (y1 + y0)/2

        # Draw a black background over the image
        bg_patch = Rectangle((x0, y0), (x1-x0), (y1-y0),
                             fill=True, alpha=0.9, color=(0, 0, 0), zorder=99)
        encourage_objects.append(bg_patch)
        self.fig.patches.append(bg_patch)

        # Draw some encouraging text
        title = self.fig.text(0.5, 0.9, '{:1.0f}% Complete!'.format(new_milestone),
                              color='white',
                              visible=True,
                              horizontalalignment='center',
                              family='sans-serif',
                              zorder=100,
                              fontsize=32)
        encourage_objects.append(title)

        if new_milestone >= 100:
            enc_text = self.fig.text(0.5, 0.1, 'Press any key to exit',
                                     color='white',
                                     visible=True,
                                     horizontalalignment='center',
                                     family='sans-serif',
                                     zorder=100,
                                     fontsize=24)
        else:
            enc_text = self.fig.text(0.5, 0.1, 'Press any key to continue',
                                     color='white',
                                     visible=True,
                                     horizontalalignment='center',
                                     family='sans-serif',
                                     zorder=100,
                                     fontsize=24)
        encourage_objects.append(enc_text)

        # Scale the encouragement image
        yext = abs(y1 - y0) * 0.65
        xext = cols / rows * yext

        simg = Image.fromarray(img)
        simg = simg.resize((int(np.floor(xext)), int(np.floor(yext))))
        simg = np.asarray(simg)

        srows, scols, _ = simg.shape

        enc_img = self.fig.figimage(simg, xo=xct-scols//2, yo=yct-srows//2, zorder=100, alpha=1.0)
        encourage_objects.append(enc_img)

        self.encourage_objects = encourage_objects

        plt.draw()

    def clear_encouragement(self):
        """ Clear the encouragement display """
        self.display_mode = 'normal'

        for obj in self.encourage_objects:
            if obj in self.fig.patches:
                self.fig.patches.remove(obj)
            if obj in self.fig.texts:
                self.fig.texts.remove(obj)
            if obj in self.fig.images:
                self.fig.images.remove(obj)

        self.encourage_objects = []
        plt.draw()

    def draw_help(self):
        """ Draw the help overlay """

        if self.display_mode != 'normal':
            return
        self.display_mode = 'help'

        help_objects = []
        bbox = self.fig.get_window_extent().bounds
        x0, y0, x1, y1 = bbox

        # Draw a black background over the image
        bg_patch = Rectangle((x0, y0), (x1-x0), (y1-y0),
                             fill=True, alpha=0.9, color=(0, 0, 0), zorder=99)
        help_objects.append(bg_patch)
        self.fig.patches.append(bg_patch)

        # Draw some help text
        title = self.fig.text(0.5, 0.9, 'Cell Locator',
                              color='white',
                              visible=True,
                              horizontalalignment='center',
                              family='sans-serif',
                              zorder=100,
                              fontsize=32)
        help_objects.append(title)

        help_text = self.fig.text(0.5, 0.75, 'Select the center of each cell',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='center',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        if UNAME == 'darwin':
            help_text = self.fig.text(0.2, 0.60, 'Single Press: Select a cell',
                                      color='white',
                                      visible=True,
                                      horizontalalignment='left',
                                      family='sans-serif',
                                      zorder=100,
                                      fontsize=24)
            help_objects.append(help_text)
        else:
            help_text = self.fig.text(0.2, 0.60, 'Left Click: Select a cell',
                                      color='white',
                                      visible=True,
                                      horizontalalignment='left',
                                      family='sans-serif',
                                      zorder=100,
                                      fontsize=24)
            help_objects.append(help_text)

        if UNAME == 'darwin':
            help_text = self.fig.text(0.2, 0.5, 'Double Press: Unselect a cell',
                                      color='white',
                                      visible=True,
                                      horizontalalignment='left',
                                      family='sans-serif',
                                      zorder=100,
                                      fontsize=24)
            help_objects.append(help_text)
        else:
            help_text = self.fig.text(0.2, 0.5, 'Right Click: Unselect a cell',
                                      color='white',
                                      visible=True,
                                      horizontalalignment='left',
                                      family='sans-serif',
                                      zorder=100,
                                      fontsize=24)
            help_objects.append(help_text)

        help_text = self.fig.text(0.2, 0.4, 'Spacebar: Save and Next Image',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='left',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        help_text = self.fig.text(0.2, 0.3, 'Escape: Save and Exit',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='left',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        help_text = self.fig.text(0.55, 0.6, 'Left Arrow: Previous Image',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='left',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        help_text = self.fig.text(0.55, 0.5, 'Right Arrow: Next Image',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='left',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        help_text = self.fig.text(0.55, 0.3, 'F1 or "h": Show Help',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='left',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        help_text = self.fig.text(0.5, 0.1, 'Press any key to continue',
                                  color='white',
                                  visible=True,
                                  horizontalalignment='center',
                                  family='sans-serif',
                                  zorder=100,
                                  fontsize=24)
        help_objects.append(help_text)

        self.help_objects = help_objects

        plt.draw()

    def clear_help(self):
        """ Clear the help display """
        self.display_mode = 'normal'

        for obj in self.help_objects:
            if obj in self.fig.patches:
                self.fig.patches.remove(obj)
            if obj in self.fig.texts:
                self.fig.texts.remove(obj)

        self.help_objects = []
        plt.draw()

    # Callbacks

    def on_key_press(self, event):

        if self.display_mode == 'help':
            self.clear_help()
            return

        if self.display_mode == 'encouragement':
            self.clear_encouragement()
            return

        if self.display_mode == 'finished':
            self.save_points()
            plt.close()
            return

        event_key = event.key.lower()
        if event_key == 'c':
            print('Clearing...')
            self.clear_shapes()
        elif event_key in (' ', 's', 'right'):
            print('Saving...')
            self.save_points()
            self.maybe_draw_encouragement()
            self.clear_shapes()
            self.load_next_record()
        elif event_key == 'left':
            print('Saving...')
            self.save_points()
            self.clear_shapes()
            self.load_next_record(step=-1)
        elif event_key in ('q', 'escape'):
            self.save_points()
            plt.close()
        elif event_key == 'u':
            self.shape_manager.on_undo()
            self.canvas.draw()
        elif event_key == 'y':
            self.shape_manager.on_redo()
            self.canvas.draw()
        elif event_key in ('?', 'h', 'f1'):
            self.draw_help()
        else:
            print('Key press: "{}"'.format(event.key))

    def on_window_close(self, event):
        print('Shutting down...')
        plt.close()

    def on_mouse_press(self, event):
        """ When the mouse button is pressed """

        if self.display_mode == 'help':
            self.clear_help()
            return

        if self.display_mode == 'encouragement':
            self.clear_encouragement()
            return

        if self.display_mode == 'finished':
            self.save_points()
            plt.close()
            return

        if self.cur_sel_mode == 'point':
            if event.button == 1:  # Left mouse
                px, py = self.shape_manager.warp_to_axis(event.x, event.y)
                self.shape_manager.on_point_complete(
                    sel_class=self.cur_sel_class,
                    x=px, y=py)
            elif event.button == 3:  # Right mouse
                px, py = self.shape_manager.warp_to_axis(event.x, event.y)
                for point in self.shape_manager.find_near_points(px, py, radius=0.01):
                    self.shape_manager.remove_shape(point)
            self.canvas.draw()

    def on_resize(self, event):
        """ Resize the window """
        self.shape_manager.on_window_resize()
        self.canvas.draw()

    # Main Method

    def show(self):
        self.load_window()
        self.connect()
        self.load_last_index()
        self.load_next_record(step=0)

        self.draw_help()

        plt.show()


# Command line interface


def main():
    tagger = ImageTagger()
    tagger.show()


if __name__ == '__main__':
    main()
