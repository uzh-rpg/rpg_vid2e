import streamlit as st
import numpy as np
import torch
from stqdm import stqdm
import skvideo.io
#from esim_torch import esim_torch
#import esim_torch
#from esim-cuda
import os
import numba
import cv2
from fractions import Fraction
from typing import Union
from pathlib import Path
from io import StringIO
#from .esim_torch import EventSimulator_torch
#from esim_torch import esim_torch
# from esim_torch.esim_torch import EventSimulator_torch
#import esim_torch
import glob
import h5py
from utils.events import Events

import sys
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../esim_torch"))
from esim_torch import EventSimulator_torch

st.sidebar.title("Menu")
add_selectbox = st.sidebar.selectbox(
    "ESIM Playground Options",
    ("Offline Video Generator", "Live Webcam")
)
if add_selectbox == "Offline Video Generator":

  st.title("ESIM Web App Playground")
  st.markdown('The goal of this project is to make ESIM and VID2E available to researchers who do not possess a real event camera. For this we want to deploy VID2E and ESIM as an interactive web app. The app should be easy to use and have the following functional requirements:')
  st.markdown('- Generation of events from video through dragging and dropping a video into thebrowser. The resulting events should be downloadable as raw events and rendered video.')
  st.markdown('- Incorporate video interpolation via an existing video interpolation method (e.g. Super SloMo) before event generation with ESIM.')
  st.markdown('- Visualization and inspection of the event stream in the browser.')
  st.markdown('- The event generation should be configurable by changing ESIM parameters.')
  st.subheader("Upload")
  video_file = st.file_uploader("Upload the video file for which you would like the events generation", type=([".webm", ".mp4", ".m4p", ".m4v", ".avi", ".avchd", ".ogg", ".mov", ".ogv", ".vob", ".f4v", ".mkv", ".svi", ".m2v", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".amv", ".wmv", ".flv", ".mts", ".m2ts", ".ts", ".qt", ".3gp", ".3g2", ".f4p", ".f4a", ".f4b"]))

  st.subheader('ESIM Settings')
  ct_options = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
  st.subheader('Positive Contrast Threshold')
  ct_p = st.select_slider("Choose the +ve contrast threshold", options=ct_options)
  st.subheader('Negative Contrast Threshold')
  ct_n = st.select_slider("Choose the -ve contrast threshold", options=ct_options)
  window_size_options = [24, 30, 60, 80, 100, 120]
  st.subheader('Window Size (Number of Frames)')
  window_size = st.select_slider("Choose the window size or number of frames for event generation", options=window_size_options)
  st.subheader('Upsampling')
  upsampling = st.checkbox('Yes', True)
  st.subheader('Output Format')
  format_name = ['HDF5', 'NPZ', 'Rendered Video']
  format = st.radio('Select the output format for the generated events', format_name)

  class Sequence:
    #def __init__(self):
      #normalize = transforms.Normalize(mean=mean, std=std)
      #self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __iter__(self):
      return self

    def __next__(self):
      raise NotImplementedError

    def __len__(self):
      raise NotImplementedError