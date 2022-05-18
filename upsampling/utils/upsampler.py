import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    def __init__(self, I0, t0):
        # assumes images are not batched
        self.I0 = I0 
        self.t0 = t0 

        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net/Style/saved_model")
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



class BatchUpsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str):
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir

    def upsample(self):
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue
            sequence_counter += 1
            print('Processing sequence number {}'.format(src_absdirpath))
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)

        idx = 0
        for (I0, I1), (t0, t1) in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
            if idx == 0:
                upsampler = Upsampler(I0=I0, t0=t0)
                self._write_img(I0, idx, dest_imgs_dir)
                self._write_timestamp(t0, dest_timestamps_filepath)
                
            new_frames, new_timestamps = upsampler.upsample_adaptively(I1, t1)
            for t, frame in zip(new_timestamps, new_frames):
                self._write_img(frame, idx, dest_imgs_dir)
                self._write_timestamp(t, dest_timestamps_filepath)
                idx += 1
    
    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
        # Copy directory structure.
        def ignore_files(directory, files):
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]
        shutil.copytree(src_dir, dest_dir, ignore=ignore_files)

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = np.clip(img * 255, 0, 255).astype("uint8")
        path = os.path.join(imgs_dir, "%08d.png" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamp(timestamp: float, timestamps_filename: str):
        with open(timestamps_filename, 'a') as t_file:
            t_file.write(f"{timestamp}\n")