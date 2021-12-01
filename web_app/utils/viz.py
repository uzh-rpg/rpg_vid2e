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