import argparse
from operator import sub
import os
import esim_torch
import numpy as np
import glob
import cv2
import tqdm
import torch


def is_valid_dir(subdirs, files):
    return len(subdirs) == 1 and len(files) == 1 and "timestamps.txt" in files and "imgs" in subdirs


def process_dir(outdir, indir, args):
    print(f"Processing folder {indir}... Generating events in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # constructor
    esim = esim_torch.ESIM(args.contrast_threshold_negative,
                           args.contrast_threshold_positive,
                           args.refractory_period_ns)

    timestamps = np.genfromtxt(os.path.join(indir, "timestamps.txt"), dtype="float64")
    timestamps_ns = (timestamps * 1e9).astype("int64")
    timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

    image_files = sorted(glob.glob(os.path.join(indir, "imgs", "*.png")))
    
    pbar = tqdm.tqdm(total=len(image_files)-1)
    num_events = 0

    counter = 0
    for image_file, timestamp_ns in zip(image_files, timestamps_ns):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        sub_events = esim.forward(log_image, timestamp_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None:
            continue

        sub_events = {k: v.cpu() for k, v in sub_events.items()}    
        num_events += len(sub_events['t'])
 
        # do something with the events
        np.savez(os.path.join(outdir, "%010d.npz" % counter), **sub_events)
        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--contrast_threshold_negative", "-cn", type=float, default=0.2)
    parser.add_argument("--contrast_threshold_positive", "-cp", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", "-rp", type=int, default=0)
    parser.add_argument("--input_dir", "-i", default="", required=True)
    parser.add_argument("--output_dir", "-o", default="", required=True)
    args = parser.parse_args()


    print(f"Generating events with cn={args.contrast_threshold_negative}, cp={args.contrast_threshold_positive} and rp={args.refractory_period_ns}")

    for path, subdirs, files in os.walk(args.input_dir):
        if is_valid_dir(subdirs, files):
            output_folder = os.path.join(args.output_dir, os.path.relpath(path, args.input_dir))

            process_dir(output_folder, path, args)
