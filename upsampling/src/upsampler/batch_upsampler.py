import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname
from .utils import get_sequence_or_none
from .upsampler import Upsampler


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