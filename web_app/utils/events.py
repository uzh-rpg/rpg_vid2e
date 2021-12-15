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

    def slice_before_t(self, t, num_events=-1):
        events = self.events[self.events[:, 2] < t]
        if num_events > 0:
            events = events[-num_events:]
        return Events(shape=self.shape, events=events)

    def slice_after_t(self, t):
        return Events(shape=self.shape, events=self.events[self.t>t])

    def slice_num_events(self, num_events):
        return Events(shape=self.shape, events=self.events[-num_events:])

    def chunk(self, i, j):
        return Events(shape=self.shape, events=self.events[i:j])

    def render(self, rendering=None, rendering_type=EventRenderingType.RED_BLUE_NO_OVERLAP):
        if rendering_type == EventRenderingType.RED_BLUE_OVERLAP:
            return _render_overlap(self, rendering, color="red_blue")
        elif rendering_type == EventRenderingType.RED_BLUE_NO_OVERLAP:
            return _render_no_overlap(self, rendering, color="red_blue")
        elif rendering_type == EventRenderingType.BLACK_WHITE_NO_OVERLAP:
            return _render_no_overlap(self, rendering, color="black_white")
        elif rendering_type == EventRenderingType.TIME_SURFACE:
            return _render_timesurface(self)
        elif rendering_type == EventRenderingType.EVENT_FRAME:
            return _render_event_frame(self)

    def __repr__(self):
        return self.events.__repr__()

    def mask(self, mask):
        return Events(shape=self.shape, events=self.events[mask])

    def interactive_visualization_loop(self, window_size_ms, framerate, rendering_type=EventRenderingType.RED_BLUE_OVERLAP):
        visualizer = Visualizer(self, 
                                window_size_ms=window_size_ms, 
                                framerate=framerate,
                                rendering_type=rendering_type)
        visualizer.visualizationLoop()

    def compute_index(self, t):
        return np.searchsorted(self.t, t)-1


def _render_overlap(events, rendering, color="red_blue"):
    white_canvas = np.full(shape=(events.shape[0], events.shape[1], 3), fill_value=255, dtype="uint8")
    rendering = rendering.copy() if rendering is not None else white_canvas
    mask = _is_in_rectangle(events.x, events.y, rendering.shape[:2])
    x, y, p = events.x.astype("int"), events.y.astype("int"), events.p.astype("p")

    if color == "red_blue":
        # map p 0, 1 to -1,0
        rendering[y[mask], x[mask], :] = 0
        rendering[y[mask], x[mask], p[mask]-1] = 255
    return rendering

def _render_no_overlap(events, rendering, color="red_blue"):
    fill_value = 128 if color == "black_white" else 255
    canvas = np.full(shape=(events.shape[0], events.shape[1], 3), fill_value=fill_value, dtype="uint8")
    rendering = rendering.copy() if rendering is not None else canvas
    H, W = rendering.shape[:2]
    mask = (events.x >= 0) & (events.y >= 0) & (events.x <= W - 1) & (events.y <= H - 1)

    red = np.array([255, 0, 0])
    blue = np.array([0,0,255])
    black = np.array([0,0,0])
    white = np.array([255,255,255])

    if color == "red_blue":
        pos = blue
        neg = red
    elif color == "black_white":
        pos = white
        neg = black

    visited_mask = np.ones(shape=events.shape) == 0
    return _render_no_overlap_numba(rendering,
                                    visited_mask,
                                    events.x[mask][::-1].astype("int"),
                                    events.y[mask][::-1].astype("int"),
                                    events.p[mask][::-1].astype("int"),
                                    pos, neg)


@numba.jit(nopython=True)
def _render_no_overlap_numba(rendering, mask, x, y, p, pos_color, neg_color):
    for x_, y_, p_ in zip(x, y, p):
        if not mask[y_, x_]:
            rendering[y_, x_] = pos_color if p_ > 0 else neg_color
            mask[y_, x_] = True
    return rendering

def _render_timesurface(events):
    image = np.zeros(events.shape, dtype="float32")
    cm = plt.get_cmap("jet")
    tau = 3e4
    t = events.t.astype("int")

    if len(events) > 2:
        value = np.exp(-(t[-1]-t)/float(tau))
        _aggregate(image, events.x, events.y, value)
        image = cm(image)
    else:
        image = image.astype("uint8")

    return image

def _render_event_frame(events):
    img = np.zeros(shape=events.shape, dtype="float32")
    img = _aggregate(img, events.x, events.y, 2*events.p-1)
    img_rendered = 10 * img + 128
    return np.clip(img_rendered, 0, 255).astype("uint8")

def _aggregate_int(img, x, y, v):
    np.add.at(img, (y, x), v)

def _aggregate(img, x, y, v):
    if x.dtype == np.float32 or x.dtype == np.float64:
        _aggregate_float(img, x, y, v)
    else:
        _aggregate_int(img, x, y, v)
    return img

def _aggregate_float(img, x, y, v):
    H, W = img.shape
    x_ = x.astype("int32")
    y_ = y.astype("int32")
    for xlim in [x_, x_+1]:
        for ylim in [y_, y_+1]:
            mask = (xlim >= 0) & (ylim >= 0) & (xlim < W) & (ylim < H)
            weight = (1 - np.abs(xlim - x)) * (1 - np.abs(ylim - y))
            _aggregate_int(img, xlim[mask], ylim[mask], v[mask] * weight[mask])

def _is_in_rectangle(x, y, shape):
    return (x >= 0) & (y >= 0) & (x <= shape[1]-1) & (y <= shape[0]-1)