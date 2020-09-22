""" Selection databse """

# Standard lib
import sqlite3
import pathlib
from collections import namedtuple

# Constants

PointROI = namedtuple('PointROI', 'sel_class, x, y')

# Classes


class SelectionDB2(object):

    q_has_tables = """
        SELECT name FROM sqlite_master WHERE type='table' AND name='points';
    """
    q_create_table_points = """
        CREATE TABLE points (path text, sel_class int, x float, y float)
    """
    q_create_table_images = """
        CREATE TABLE images (path text, start_view timestamp, end_view timestamp)
    """

    q_find_points = """
        SELECT sel_class, x, y FROM points WHERE path=?
    """
    q_find_annotated_paths = """
        SELECT path FROM points
    """
    q_insert_points = """
        INSERT INTO points VALUES (?, ?, ?, ?)
    """

    q_find_last_viewed = """
        SELECT path FROM images ORDER BY end_view desc LIMIT 1
    """
    q_add_view = """
        INSERT INTO images(path, start_view, end_view) VALUES (?, ?, ?)
    """

    def __init__(self, dbpath):
        self.dbpath = pathlib.Path(dbpath)
        self._db = None

    def _dedup_points(self, classes, points):
        """ Deduplicate the selections """

        rec_points = {}
        for sel_class, (x, y) in zip(classes, points):
            # Prevent storing the same ROI twice
            sel_point = x, y
            have_points = rec_points.get(sel_class, [])
            dup_points = False
            for have_point in have_points:
                if all([abs(s-r) < 1e-5 for s, r in zip(sel_point, have_point)]):
                    dup_points = True
                    break
            if dup_points:
                continue
            rec_points.setdefault(sel_class, []).append(sel_point)
            yield sel_class, x, y

    def connect(self, create=True):
        """ Connect to the database """

        self._db = sqlite3.connect(str(self.dbpath))

        # See if we need to create the data tables
        rec = self._db.execute(self.q_has_tables).fetchone()
        if rec is None:
            self._db.execute(self.q_create_table_points)
            self._db.execute(self.q_create_table_images)
            self._db.commit()

    def get_last_viewed(self):
        """ Get the last viewed image """
        return self._db.execute(self.q_find_last_viewed).fetchone()

    def add_view(self, path, start_time, end_time):
        """ Add an image view record """
        self._db.execute(self.q_add_view, (path, start_time, end_time))
        self._db.commit()

    def find_points(self, filepath):
        """ Find all the points for this file """
        if filepath is None:
            return []
        recs = self._db.execute(self.q_find_points, (filepath, ))
        return (PointROI(r[0], r[1], r[2]) for r in recs)

    def find_annotated_records(self):
        """ Find the list of images that have been annotated """
        return {r[0] for r in self._db.execute(self.q_find_annotated_paths)}

    def set_points(self, filepath, classes=1, points=None):
        """ Set the points for this file """

        if isinstance(classes, int):
            classes = [classes for _ in range(len(points))]
        if len(classes) != len(points):
            err = 'Got {} classes but {} points'.format(len(classes),
                                                        len(points))
            raise ValueError(err)

        dedup_points = self._dedup_points(classes, points)

        # Clear and then reinsert the data
        self._db.execute('DELETE FROM points where path=?', (filepath, ))
        for sel_class, x, y in dedup_points:
            self._db.execute(self.q_insert_points, (filepath, sel_class, x, y))
        self._db.commit()


# Main Function


def load_selection_db(dbpath):
    """ Load the selection database """
    db = SelectionDB2(dbpath)
    db.connect()
    return db
