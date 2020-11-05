import os
from pathlib import Path
from typing import Union

from .const import fps_filename, imgs_dirname, video_formats
from .dataset import Sequence, ImageSequence, VideoSequence

def is_video_file(filepath: str) -> bool:
    """
    Determine if filepath is a video file.

    Args:
        filepath: (str): write your description
    """
    return Path(filepath).suffix.lower() in video_formats

def get_fps_file(dirpath: str) -> Union[None, str]:
    """
    Returns the absolute path to a directory.

    Args:
        dirpath: (str): write your description
    """
    fps_file = os.path.join(dirpath, fps_filename)
    if os.path.isfile(fps_file):
        return fps_file
    return None

def get_imgs_directory(dirpath: str) -> Union[None, str]:
    """
    Returns the directory of the given directory.

    Args:
        dirpath: (str): write your description
    """
    imgs_dir = os.path.join(dirpath, imgs_dirname)
    if os.path.isdir(imgs_dir):
        return imgs_dir
    return None

def get_video_file(dirpath: str) -> Union[None, str]:
    """
    Get the video file path.

    Args:
        dirpath: (str): write your description
    """
    filenames = [f for f in os.listdir(dirpath) if is_video_file(f)]
    if len(filenames) == 0:
        return None
    assert len(filenames) == 1
    filepath = os.path.join(dirpath, filenames[0])
    return filepath

def fps_from_file(fps_file) -> float:
    """
    Create a multicastip file.

    Args:
        fps_file: (str): write your description
    """
    assert os.path.isfile(fps_file)
    with open(fps_file, 'r') as f:
        fps = float(f.readline().strip())
    assert fps > 0, 'Expected fps to be larger than 0. Instead got fps={}'.format(fps)
    return fps

def get_sequence_or_none(dirpath: str) -> Union[None, Sequence]:
    """
    Returns the video or none if not found.

    Args:
        dirpath: (str): write your description
    """
    fps_file = get_fps_file(dirpath)
    if fps_file:
        # Must be a sequence (either ImageSequence or VideoSequence)
        fps = fps_from_file(fps_file)
        imgs_dir = get_imgs_directory(dirpath)
        if imgs_dir:
            return ImageSequence(imgs_dir, fps)
        video_file = get_video_file(dirpath)
        assert video_file is not None
        return VideoSequence(video_file, fps)
    # Can be VideoSequence if there is a video file. But have to use fps from meta data.
    video_file = get_video_file(dirpath)
    if video_file is not None:
        return VideoSequence(video_file)
    return None


