# esym\_py

This package exposes python bindings for ESIM which can be used within a training loop. 
To test out if the installation was successful you can run

```bash
python tests/test.py
```

which should print a message if completed sucessfully. 

The currently supported functions are listed in the example below:
```python
import esim_py

# constructor
esim = esim_py.EventSimulator(
    contrast_threshold_pos,  # contrast thesholds for positive 
    contrast_threshold_neg,  # and negative events
    refractory_period,  # minimum waiting period (in sec) before a pixel can trigger a new event
    log_eps,  # epsilon that is used to numerical stability within the logarithm
    use_log,  # wether or not to use log intensity
    )

# setter, useful within a training loop
esim.setParameters(contrast_threshold_pos, contrast_threshold_neg, refractory_period, log_eps, use_log)

# generate events from a sequence of images
events_from_images = esim.generateFromFolder(
    path_to_image_folder, # absolute path to folder that stores images in numbered order
    path_to_timestamps    # absolute path to timestamps file containing one timestamp (in secs) for each 
)

# generate events from a video
events_from_video = esim.generateFromVideo(
    path_to_video_file,   # absolute path to video storing images
    path_to_timestamps    # absolute path to timestamps file
)

# generate events from list of images and timestamps
events_list_of_images = esim.generateFromStampedImageSequence(
    list_of_image_files,   # list of absolute paths to images
    list_of_timestamps     # list of timestamps in ascending order
)

```
The example script `tests/plot_virtual_events.py` plots virtual events that are generated from images in `tests/data/images` with varying positive and negative contrast thresholds. To call it you need some additional pip packages:

```bash
pip install numpy matplotlib
python tests/plot_virtual_events.py
```
