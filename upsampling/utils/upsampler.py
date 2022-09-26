import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from . import Sequence, VideoSequence, ImageSequence
from .const import imgs_dirname, video_formats, fps_filename
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str):
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir

        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)

    def upsample(self):
        sequence_counter = 0
        sequences = self.find_sequences_recursive(self.src_dir)
        for sequence in sequences:
            sequence_counter += 1
            print('Processing sequence number {}'.format(sequence_counter))
            reldirpath = os.path.relpath(sequence.dest_dir, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            os.makedirs(dest_imgs_dir, exist_ok=True)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def find_sequences_recursive(self, src_dir):
        src_dir = Path(src_dir)
        # find all video files
        sequences = []
        for video_format in video_formats:
            video_files = list(src_dir.rglob("*"+video_format))
            for video_file in video_files:
                sequence = VideoSequence(str(video_file))
                sequences.append(sequence)

        # find all fps files, since these should be with images
        fps_files = list(src_dir.rglob("**/" + fps_filename))
        for fps_file in fps_files:
            fps = np.genfromtxt(fps_file)[0]
            sequence = ImageSequence(str(fps_file.parent), fps)
            sequences.append(sequence)

        return sequences

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        idx = 0
        for i, (img_pair, time_pair) in enumerate(tqdm(next(sequence), total=len(sequence), desc=sequence.name)):
            I0 = img_pair[0][None]
            I1 = img_pair[1][None]
            t0, t1 = time_pair

            total_frames, total_timestamps = self._upsample_adaptive(I0, I1, t0, t1)
            total_frames = [I0[0]] + total_frames
            timestamps = [t0] + total_timestamps

            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]

            timestamps_list += timestamps
            for frame in total_frames:
                self._write_img(frame, idx, dest_imgs_dir)
                idx += 1

        timestamps_list.append(t1)
        self._write_img(I1[0, ...], idx, dest_imgs_dir)
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))

        if num_bisections == 0:
            return [], []

        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps

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
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])