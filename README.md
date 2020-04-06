# Video to Events: Recycling Video Datasets for Event Cameras

<p align="center">
  <a href="https://youtu.be/uX6XknBGg0w">
    <img src="http://rpg.ifi.uzh.ch/data/VID2E/thumb.png" alt="Video to Events" width="600"/>
  </a>
</p>

This repository contains code that implements 
video to events conversion as described in Gehrig et al. CVPR'20. The paper can be found [here](http://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf)

If you use this code in an academic context, please cite the following work:

[Daniel Gehrig](https://danielgehrig18.github.io/), Mathias Gehrig, Javier Hidalgo-Carrió, [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "Video to Events: Recycling Video Datasets for Event Cameras", The Conference on Computer Vision and Pattern Recognition (CVPR), 2020

```bibtex
@InProceedings{Gehrig_2020_CVPR,
  author = {Daniel Gehrig and Mathias Gehrig and Javier Hidalgo-Carri\'o and Davide Scaramuzza},
  title = {Video to Events: Recycling Video Datasets for Event Cameras},
  booktitle = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  month = {June},
  year = {2020}
}
```

## Installation
Clone the repo *recursively with submodules*

```bash
git clone git@github.com:uzh-rpg/rpg_vid2e.git --recursive
```

## Installation with [Anaconda](https://www.anaconda.com/distribution/)
Adapt the CUDA toolkit version according to your setup. 

```bash
cuda_version=10.1

conda create -y -n vid2e python=3.7
conda activate vid2e
conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
conda install -y -c conda-forge opencv tqm scikit-video eigen boost boost-cpp pybind11
```

Build the python bindings for ESIM

```bash
cd esim_py
pip install .
```

## Adaptive Upsampling
*This package provides code for adaptive upsampling with frame interpolation based on [Super-SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)*

Consult the [README](upsampling/README.md) for detailed instructions and examples.

## esim\_py
*This package exposes python bindings for [ESIM](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) which can be used within a training loop.*

For detailed instructions and example consult the [README](esim_py/README.md)
