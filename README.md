# Video to Events: Recycling Video Datasets for Event Cameras

<p align="center">
  <a href="https://youtu.be/uX6XknBGg0w">
    <img src="http://rpg.ifi.uzh.ch/data/VID2E/thumb.png" alt="Video to Events" width="600"/>
  </a>
</p>

This repository contains code that implements 
video to events conversion as described in Gehrig et al. CVPR'20 and the used dataset. The paper can be found [here](http://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf)

If you use this code in an academic context, please cite the following work:

[Daniel Gehrig](https://danielgehrig18.github.io/), [Mathias Gehrig](https://magehrig.github.io/), [Javier Hidalgo-Carrió](https://jhidalgocarrio.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "Video to Events: Recycling Video Datasets for Event Cameras", The Conference on Computer Vision and Pattern Recognition (CVPR), 2020

```bibtex
@InProceedings{Gehrig_2020_CVPR,
  author = {Daniel Gehrig and Mathias Gehrig and Javier Hidalgo-Carri\'o and Davide Scaramuzza},
  title = {Video to Events: Recycling Video Datasets for Event Cameras},
  booktitle = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  month = {June},
  year = {2020}
}
```
## News
* We now support frame interpolation done by [FILM](https://github.com/google-research/frame-interpolation).
* We release a web app and interactive demo which generates events and converts your webcam to events. Try it out [here](web_app/README.md).
* We now also release new python bindings for esim with GPU support.
Details are [here](esim_torch/README.md)

## Web App and Interactive Demo
Try out our the interactive demo and webcam support [here](web_app/README.md). 

## Dataset
The synthetic N-Caltech101 dataset, as well as video sequences used for event conversion can be found [here](http://rpg.ifi.uzh.ch/data/VID2E/ncaltech_syn_images.zip). For each sample of each class it contains events in the form `class/image_%04d.npz` and images in the form `class/image_%05d/images/image_%05d.png`, as well as the corresponding timestamps of the images in `class/image_%04d/timestamps.txt`.

## Installation
Clone the repo *recursively with submodules*

```bash
git clone git@github.com:uzh-rpg/rpg_vid2e.git --recursive
```

## Installation
First download the [FILM](https://github.com/google-research/frame-interpolation) checkpoint, and move it to the current root
```bash
    wget https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip -O /tmp/temp.zip
    unzip /tmp/temp.zip -d rpg_vid2e/
    rm -rf /tmp/temp.zip
```

make sure to install the following
    * [Anaconda Python 3.9](https://www.anaconda.com/products/individual)
    * [CUDA Toolkit 11.2.1](https://developer.nvidia.com/cuda-11.2.1-download-archive)
    * [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-download)

```bash
conda create --name vid2e python=3.9
conda activate vid2e
pip install -r rpg_vid2e/requirements.txt
conda install -y -c conda-forge pybind11 matplotlib
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Build the python bindings for ESIM

```bash
pip install rpg_vid2e/esim_py/
```

Build the python bindings with GPU support with 

```bash
pip install rpg_vid2e/esim_torch/
```

## Adaptive Upsampling
*This package provides code for adaptive upsampling with frame interpolation based on [Super-SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)*

Consult the [README](upsampling/README.md) for detailed instructions and examples.

## esim\_py
*This package exposes python bindings for [ESIM](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) which can be used within a training loop.*

For detailed instructions and example consult the [README](esim_py/README.md)

## esim\_torch
*This package exposes python bindings for [ESIM](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) with GPU support.*

For detailed instructions and example consult the [README](esim_torch/README.md)

## Example
To run an example, first upsample the example videos 

```bash
device=cpu
# device=cuda:0
python upsampling/upsample.py --input_dir=example/original --output_dir=example/upsampled --device=$device

```
This will generate upsampling/upsampled with in the `example/upsampled` folder. To generate events, use
```bash
python esim_torch/generate_events.py --input_dir=example/upsampled \
                                     --output_dir=example/events \
                                     --contrast_threshold_neg=0.2 \
                                     --contrast_threshold_pos=0.2 \
                                     --refractory_period_ns=0
```



