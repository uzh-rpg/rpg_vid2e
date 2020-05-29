# Adaptive Upsampling

## Generate Upsampled Video or Image Sequences
You can use our example directory to experiment
```bash
device=cpu
# device=cuda:0
python upsample.py --input_dir=example/original --output_dir=example/upsampled --device=$device

```
The **expected input structure** is as follows:
```
input_dir
├── seq0
│   ├── fps.txt
│   └── imgs
│       ├── 00000001.png
│       ├── 00000002.png
│       ├── 00000003.png
│       └── ....png
├── seq1
│   └── video.mp4
└── dirname_does_not_matter
    ├── fps.txt
    └── filename_does_not_matter.mov

```
- The number of sequences (subfolders of the input directory) is unlimited.
- The `fps.txt` file
    - must specify the frames per second in the first line. The rest of the file should be empty (see example directory).
    - is required for sequences (such as seq0) with image files.
    - is **optional** for sequences with a video file. In case of a missing `fps.txt` file, the frames per second will be inferred from the metadata of the video file.

The **resulting output structure** is as follows:
```
output_dir
├── seq0
│   ├── imgs
│   │   ├── 00000001.png
│   │   ├── 00000002.png
│   │   ├── 00000003.png
│   │   └── ....png
│   └── timestamps.txt
├── seq1
│   ├── imgs
│   │   ├── 00000001.png
│   │   ├── 00000002.png
│   │   ├── 00000003.png
│   │   └── ....png
│   └── timestamps.txt
└── dirname_does_not_matter
    ├── imgs
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   ├── 00000003.png
    │   └── ....png
    └── timestamps.txt
```
The resulting image directories can later be used to generate events. The `timestamps.txt` file contains the timestamp of each image in seconds.


## Remarks
- Use a GPU device whenever possible to speed up the upsampling procedure.
- The upsampling will increase the storage requirements significantly. Try a small sample first to get an impression.
- Downsample (height and width) your images and video to save storage space and processing time.
- Why store the upsampling result in images:
    - Images support random access from a dataloader. A video file, for example, can typically only be accessed sequentally when we try to avoid loading the whole video into RAM.
    - Same sequence can be accessed by multiple processes (e.g. PyTorch num\_workers > 1).
    - Well established C++ interface to load images. This is useful to generate events on the fly (needed for contrast threshold randomization) in C++ code without loading data in Python first.
  If there is a need to store the resulting sequences in a different format, raise an issue (feature request) on this GitHub repository.
- Be aware that upsampling videos might fail due to a [bug in scikit-video](https://github.com/scikit-video/scikit-video/issues/60)

### Generating Video Files from Images
If you want to convert an ordered sequence of images (here png files) into video format you can use the following command (you may have to deactivate the current conda environment):
```bash
frame_rate=25
img_dirpath="example/original/seq0/imgs"
img_suffix=".png"
output_file="video.mp4"
ffmpeg -framerate $frame_rate -pattern_type glob -i "$img_dirpath/*$img_suffix" -c:v libx265 -x265-params lossless=1 $output_file
```

### Generating Images from a Video File
If you want to convert a video file to a sequence of images:
```bash
input_file="video.mp4"
output_dirpath="your_path_to_specify"
ffmpeg -i $input_file "$output_dirpath/%08d.png"
```
