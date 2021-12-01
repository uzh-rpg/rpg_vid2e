import cv2
import numpy as np

from utils.utils import EventRenderingType


class Visualizer:
    def __init__(self, events, window_size_ms=10, framerate=100, rendering_type=EventRenderingType.RED_BLUE_NO_OVERLAP):
        self.events = events
        self.rendering_type = rendering_type
        self.window_size_ms = window_size_ms
        self.framerate = framerate

        self.time_between_screen_refresh_ms = 5

        self.is_paused = False
        self.is_looped = False

        self.update_indices(events, window_size_ms, framerate)

        self.index = 0

        self.cv2_window_name = 'Events'
        self.annotations = {}
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)


        self.print_help()

    def update_indices(self, events, window_size_ms, framerate):
        self.t0_us, self.t1_us = self.compute_event_window_limits(events, window_size_ms, framerate)
        self.t0_index = events.compute_index(self.t0_us)
        self.t1_index = events.compute_index(self.t1_us)

    def compute_event_window_limits(self, events, window_size_ms, framerate):
        t_min_us = events.t[0]
        t_max_us = events.t[-1]
        t1_us = np.arange(t_min_us, t_max_us, 1e6 / framerate)
        t0_us = np.clip(t1_us - window_size_ms * 1e3, t_min_us, t_max_us)
        return t0_us, t1_us

    def pause(self):
        self.is_paused = True

    def unpause(self):
        self.is_paused= False