import torch
import matplotlib.pyplot as plt
import numpy as np

import esim_torch


def increasing_sin_wave(t):
    return (400 * np.sin((t-t[0])*20*np.pi)*(t-t[0])+150).astype("uint8").reshape((-1,1,1))

if __name__ == "__main__":
    c = 0.2
    refractory_period_ns = 5e6
    esim = esim_torch.ESIM(contrast_threshold_neg=c,
                           contrast_threshold_pos=c,
                           refractory_period_ns=refractory_period_ns)

    print("Loading images")
    timestamps_s = np.genfromtxt("../esim_py/tests/data/images/timestamps.txt")
    images = increasing_sin_wave(timestamps_s)
    timestamps_ns = (timestamps_s * 1e9).astype("int64")
    log_images = np.log(images.astype("float32") / 255 + 1e-4)

    # generate torch tensors
    print("Loading data to GPU")
    device = "cuda:0"
    log_images = torch.from_numpy(log_images).to(device)
    timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

    # generate events with GPU support
    print("Generating events")
    events = esim.forward(log_images, timestamps_ns)

    # render events
    image = images[0]

    print("Plotting")
    event_timestamps = events['t']
    event_polarities = events['p']
    i0 = log_images[0].cpu().numpy().ravel()

    fig, ax = plt.subplots(ncols=2)
    timestamps_ns = timestamps_ns.cpu().numpy()
    log_images = log_images.cpu().numpy().ravel()
    ax[0].plot(timestamps_ns, log_images)
    ax[0].plot(timestamps_ns, images.ravel())
    ax[0].set_ylim([np.log(1e-1),np.log(1 + 1e-4)])
    ax[0].set_ylabel("Log Intensity")
    ax[0].set_xlabel("Time [ns]")
    ax[1].set_ylabel("Time since last event [ns]")
    ax[1].set_xlabel("Timestamp of event [ns]")
    ax[1].set_xlim([0,3e8])

    for i in range(-10,3):
        ax[0].plot([0,timestamps_ns[-1]], [i0+i*c, i0+i*c], c='g')

    event_timestamps = event_timestamps.cpu().numpy()
    for i, (t, p) in enumerate(zip(event_timestamps, event_polarities)):
        color = "r" if p == -1 else "b"
        ax[0].plot([t, t], [-3, 0], c=color)
        if i > 0:
            ax[1].scatter([t], [t-event_timestamps[i-1]], c=color)

    ax[1].plot([0,3e8], [refractory_period_ns, refractory_period_ns])

    plt.show()

