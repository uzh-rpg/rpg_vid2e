import os
import numpy as np
from .interpolator import Interpolator


class Upsampler:
    def __init__(self, I0, t0):
        # assumes images are not batched
        self.I0 = I0 
        self.t0 = t0 

        path = os.path.join(os.path.dirname(__file__), "../../../pretrained_models/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)

    def upsample_adaptively(self, I, t):
        """
        Returns all images and timestamps up to and including image I, at time t.
        Assumes I is not batched
        """
        assert len(I.shape) == 3
        total_frames, total_timestamps = self._upsample_adaptive(self.I0[None], I[None], self.t0, t)
        total_frames = total_frames + [I]
        timestamps = total_timestamps + [t]

        sorted_indices = np.argsort(timestamps)
        total_frames = [total_frames[j] for j in sorted_indices]
        timestamps = [timestamps[i] for i in sorted_indices]

        self.I0 = I 
        self.t0 = t

        return total_frames, timestamps

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))

        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps
