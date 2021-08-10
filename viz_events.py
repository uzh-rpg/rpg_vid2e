import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--shape", nargs=2, default=[256, 320])
    args = parser.parse_args()

    event_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    
    fig, ax = plt.subplots()
    events = np.load(event_files[0])
    img = render(shape=args.shape, **events)
    handle = plt.imshow(img)
    plt.show(block=False)
    plt.pause(0.002)

    for f in event_files[1:]:
        events = np.load(f)
        img = render(shape=args.shape, **events)
        handle.set_data(img)
        plt.pause(0.002)


