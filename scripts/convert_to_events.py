import argparse
import glob
import cv2
import os
import numpy as np
from os.path import join, dirname
import torch
import tqdm
from utils import image_iterator_from_folder, H5Writer, Events

import sys
sys.path.append(join(dirname(__file__), "..", "esim_torch"))
print(sys.path)
import esim_torch
from pathlib import Path


def process_dir(timestamps_file_path,
                images_directory,
                event_file_path,
                args):
    timestamps_s = np.genfromtxt(timestamps_file_path, dtype="int64")
    timestamps_s = torch.from_numpy(timestamps_s).cuda()
    esim = esim_torch.EventSimulator_torch(contrast_threshold_neg=args.contrast_threshold_neg,
                                           contrast_threshold_pos=args.contrast_threshold_pos,
                                           refractory_period_ns=args.refractory_period_ns)

    outfile = Path(event_file_path)
    h5writer = H5Writer(outfile)

    for i, image in image_iterator_from_folder(images_directory, enum=True):
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        # generate events
        timestamp_ns = (timestamps_s[i]* 1e9)
        events = esim.forward(log_image, timestamp_ns)

        if events is None:
            continue

        events = {k: v.cpu().numpy() for k, v in events.items()}

        # cast event data to specific types
        events['x'] = events['x'].astype("uint16")
        events['y'] = events['y'].astype("uint16")
        events['p'] = events['p'].astype("int8")
        events['t'] = events['t'].astype("uint64")
        events = Events(**events)
        h5writer.add_data(events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualize Trajectory""")
    parser.add_argument("-cp", "--contrast_threshold_pos", type=float, default=0.1)
    parser.add_argument("-cn", "--contrast_threshold_neg", type=float, default=0.1)
    parser.add_argument("-r", "--refractory_period_ns", type=int, default=0)
    parser.add_argument("-d", "--input_directory", default="")
    parser.add_argument("-t", "--timestamps", default="timestamps.txt")
    parser.add_argument("-e", "--events", default="events.h5")

    args = parser.parse_args()
    args.linlog = True

    for path, subdirs, files in os.walk(args.input_directory):
        if args.timestamps in files:
            assert len(subdirs) == 1
            timestamps_file_path = join(path, args.timestamps)
            images_directory = join(path, subdirs[0])
            event_file_path = join(path, args.events)
            process_dir(timestamps_file_path,
                        images_directory,
                        event_file_path, 
                        args)



