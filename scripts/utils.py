import os
from os.path import join
import cv2
import glob
import tqdm
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import weakref
from typing import Union
import h5py


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def image_files_from_folder(dir):
    return sorted(glob.glob(join(dir, "*.png")))

def image_iterator_from_folder(dir, return_path=False, enum=False, grayscale=True):
    files = image_files_from_folder(dir)
    for i, file in enumerate(tqdm.tqdm(files)):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret = [img]
        if return_path:
            ret = [file] + ret
        if enum:
            ret = [i] + ret
        yield ret



@dataclass(frozen=True)
class Events:
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray

    size: int = field(init=False)

    def __post_init__(self):
        assert self.x.dtype == np.uint16
        assert self.y.dtype == np.uint16
        assert self.p.dtype == np.uint8
        assert self.t.dtype == np.int64

        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        # Without the frozen option, we could just do: self.size = self.x.size
        super().__setattr__('size', self.x.size)

        if self.size > 0:
            assert np.max(self.p) <= 1



class H5Writer:
    def __init__(self, outfile: Path, floating_point_coords=False):
        assert not outfile.exists(), str(outfile)
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        coord_dtype = "u2" if not floating_point_coords else "f4"

        # create hdf5 datasets
        shape = (2**16,)
        maxshape = (None,)
        compression = 'lzf'
        self.h5f.create_dataset('x', shape=shape, dtype=coord_dtype, chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('y', shape=shape, dtype=coord_dtype, chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('p', shape=shape, dtype='u1', chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('t', shape=shape, dtype='i8', chunks=shape, maxshape=maxshape, compression=compression)
        self.row_idx = 0

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def add_data(self, events: Events):
        current_size = events.size
        new_size = self.row_idx + current_size
        self.h5f['x'].resize(new_size, axis=0)
        self.h5f['y'].resize(new_size, axis=0)
        self.h5f['p'].resize(new_size, axis=0)
        self.h5f['t'].resize(new_size, axis=0)

        self.h5f['x'][self.row_idx:new_size] = events.x
        self.h5f['y'][self.row_idx:new_size] = events.y
        self.h5f['p'][self.row_idx:new_size] = events.p
        self.h5f['t'][self.row_idx:new_size] = events.t

        self.row_idx = new_size