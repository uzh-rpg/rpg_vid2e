import os
from pathlib import Path
from typing import Union

from fractions import Fraction
from PIL import Image
import skvideo.io
import numpy as np

from .const import mean, std, img_formats


class Sequence:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageSequence(Sequence):
    def __init__(self, imgs_dirpath: str, fps: float):
        super().__init__()
        self.fps = fps

        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()

    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() in img_formats

    def __next__(self):
        for idx in range(0, len(self.file_names) - 1):
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = [self._pil_loader(f) for f in file_paths]
            times_sec = [idx/self.fps, (idx + 1)/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return len(self.file_names) - 1

    @staticmethod
    def _pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

            w_orig, h_orig = img.size
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            img = img.crop((left, upper, right, lower))
            return np.array(img).astype("float32") / 255

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)


class VideoSequence(Sequence):
    def __init__(self, video_filepath: str, fps: float=None):
        super().__init__()
        metadata = skvideo.io.ffprobe(os.path.abspath(video_filepath))
        self.fps = fps
        if self.fps is None:
            self.fps = float(Fraction(metadata['video']['@avg_frame_rate']))
            assert self.fps > 0, 'Could not retrieve fps from video metadata. fps: {}'.format(self.fps)
            print('Using video metadata: Got fps of {} frames/sec'.format(self.fps))

        # Length is number of frames - 1 (because we return pairs).
        self.len = int(metadata['video']['@nb_frames']) - 1
        self.videogen = skvideo.io.vreader(os.path.abspath(video_filepath))
        self.last_frame = None

    def __next__(self):
        for idx, frame in enumerate(self.videogen):
            h_orig, w_orig, _ = frame.shape
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            frame = frame[upper:lower, left:right].astype("float32") / 255
            assert frame.shape[:2] == (h, w)

            if self.last_frame is None:
                self.last_frame = frame
                continue

            last_frame_copy = self.last_frame.copy()
            self.last_frame = frame
            imgs = [last_frame_copy, frame]
            times_sec = [(idx - 1)/self.fps, idx/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return self.len
