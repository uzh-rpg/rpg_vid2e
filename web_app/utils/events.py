from os.path import join
from numba.cuda.simulator.api import Event
import numpy as np
import glob
import numba 
import h5py
import matplotlib.pyplot as plt
from utils.viz import Visualizer
from utils.utils import EventRenderingType


def load_events(f):
    if f.endswith(".npy"):
        return np.load(f).astype("int64")
    elif f.endswith(".npz"):
        fh = np.load(f)
        return np.stack([fh['x'], fh['y'], fh['t'], fh['p']], -1)
    elif f.endswith(".h5"):
        fh = h5py.File(f, "r")
        return np.stack([np.array(fh['x']), np.array(fh['y']), np.array(fh['t']), np.array(fh['p'])], -1)
    else:
        raise NotImplementedErrort(f"Could not read {f}")

class Events:
    def __init__(self, shape=None, events=None):
        self.shape = shape
        self.events = np.array(events)

    def __len__(self):
        return len(self.events)

    @classmethod
    def from_folder(cls, folder, shape):
        event_files = sorted(glob.glob(join(folder, "*")))
        events = np.concatenate([load_events(f) for f in event_files], 0)
        return cls(shape=shape, events=events)

    @classmethod
    def from_file(cls, file, shape):
        events = load_events(file)
        print(f"Loaded events from {file}, found {len(events)} events with shape {events.shape}")
        return cls(shape=shape, events=events)

    @property
    def p(self):
        return self.events[:,3]

    @property
    def x(self):
        return self.events[:, 0]

    @property
    def y(self):
        return self.events[:, 1]

    @property
    def t(self):
        return self.events[:, 2]

    def downsample(self, n):
        return Events(self.shape, self.events[::n])

    def slice_between_t(self, t0, t1):
        return self.slice_before_t(t1).slice_after_t(t0)