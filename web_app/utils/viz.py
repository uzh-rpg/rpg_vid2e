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

    def togglePause(self):
        self.is_paused = not self.is_paused

    def toggleLoop(self):
        self.is_looped = not self.is_looped

    def forward(self, num_timesteps = 1):
        if self.is_looped:
            self.index = (self.index + 1) % len(self.t0_index)
        else:
            self.index = min(self.index + num_timesteps, len(self.t0_index) - 1)

    def backward(self, num_timesteps = 1):
        self.index = max(self.index - num_timesteps, 0)

    def goToBegin(self):
        self.index = 0

    def goToEnd(self):
        self.index = len(self.t0_index) - 1

    def render_annotation(self, image, annotation):
        refPt = annotation["refPt"]
        t = annotation["t"]
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.putText(image, f"t={t}", refPt[0], cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(0,0,0))
        return image

    def cycle_colors(self):
        self.rendering_type = (self.rendering_type % len(EventRenderingType))+1
        print("New Rendering Type: ", self.rendering_type)

    def visualizationLoop(self):
        self.refPt = []
        self.cropping = False

        while True:
            self.image = self.update(self.index)
            cv2.imshow(self.cv2_window_name, self.image)

            if not self.is_paused:
                self.forward(1)

            c = cv2.waitKey(self.time_between_screen_refresh_ms)
            key = chr(c & 255)

            if c == 27:             # 'q' or 'Esc': Quit
                break
            elif key == 'r':                      # 'r': Reset
                self.goToBegin()
                self.unpause()
            elif key == 'p' or c == 32:           # 'p' or 'Space': Toggle play/pause
                self.togglePause()
            elif key == "a":                         # 'Left arrow': Go backward
                self.backward(1)
                self.pause()
            elif key == "d":                         # 'Right arrow': Go forward
                self.forward(1)
                self.pause()
            elif key == "s":                         # 'Down arrow': Go to beginning
                self.goToBegin()
                self.pause()
            elif key == "w":                         # 'Up arrow': Go to end
                self.goToEnd()
                self.pause()
            elif key == 'l':                      # 'l': Toggle looping
                self.toggleLoop()
            elif key == "e":
                self.update_window(1.2)
            elif key == "q":
                self.update_window(1/1.2)
            elif key == "c":
                self.cycle_colors()
            elif key == "h":
                self.print_help()

        cv2.destroyAllWindows()

    def print_help(self):
        print("##################################")
        print("#     interactive visualizer     #")
        print("#                                #")
        print("#     a: backward                #")
        print("#     d: forward                 #")
        print("#     w: jump to end             #")
        print("#     s: jump to front           #")
        print("#     e: lengthen time window    #")
        print("#     q: shorten time window     #")
        print("#     c: cycle color scheme      #")
        print("#   esc: quit                    #")
        print("# space: pause                   #")
        print("#     h: print help              #")
        print("##################################")

    def update_window(self, factor):
        self.window_size_ms *= factor
        self.update_indices(self.events, self.window_size_ms, self.framerate)

    def update(self, index):
        index0 = self.t0_index[index]
        index1 = self.t1_index[index]
        image = self.events.chunk(index0, index1).render(rendering_type=self.rendering_type)
        t = self.t1_us[self.index]
        cv2.putText(image, f"t={t}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(0,0,0))
        return image