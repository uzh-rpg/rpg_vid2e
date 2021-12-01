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

  class VideoSequence(Sequence):
    def __init__(self, video_filepath: str, fps: float=None):
      #super().__init__()
      self.metadata = skvideo.io.ffprobe(os.path.abspath(video_filepath))
      st.write(video_filepath)
      #st.write(type(metadata))
      #st.write(metadata.keys)
      self.fps = fps
      if self.fps is None:
        self.fps = float(Fraction(self.metadata['video']['@avg_frame_rate']))
        assert self.fps > 0, 'Could not retrieve fps from video metadata. fps: {}'.format(self.fps)
        #print('Using video metadata: Got fps of {} frames/sec'.format(self.fps))

      # Length is number of frames - 1 (because we return pairs).
      self.len = int(self.metadata['video']['@nb_frames']) - 1
      self.videogen = skvideo.io.vreader(os.path.abspath(video_filepath))
      self.last_frame = None

    def __next__(self):
      for idx, frame in enumerate(self.videogen):
        #h_orig, w_orig, _ = frame.shape
        #w, h = w_orig//32*32, h_orig//32*32

        #left = (w_orig - w)//2
        #upper = (h_orig - h)//2
        #right = left + w
        #lower = upper + h
        #frame = frame[upper:lower, left:right]
        #assert frame.shape[:2] == (h, w)
        #frame = self.transform(frame)

        if self.last_frame is None:
          self.last_frame = frame
          continue
        #last_frame_copy = self.last_frame.detach().clone()
        self.last_frame = frame
        #imgs = [last_frame_copy, frame]
        #times_sec = [(idx - 1)/self.fps, idx/self.fps]
        img = frame
        time_sec = idx/self.fps
        #yield imgs, times_sec
        yield img, time_sec

    def __len__(self):
      return self.len

  #writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

  def is_video_file(filepath: str) -> bool:
      return Path(filepath).suffix.lower() in {'.webm', '.mp4', '.m4p', '.m4v', '.avi', '.avchd', '.ogg', '.mov', '.ogv', '.vob', '.f4v', '.mkv', '.svi', '.m2v', '.mpg', '.mp2', '.mpeg', '.mpe', '.mpv', '.amv', '.wmv', '.flv', '.mts', '.m2ts', '.ts', '.qt', '.3gp', '.3g2', '.f4p', '.f4a', '.f4b'}

  def get_video_file_path(dirpath: str) -> Union[None, str]:
      filenames = [f for f in os.listdir(dirpath) if is_video_file(f)]
      if len(filenames) == 0:
          return None
      assert len(filenames) == 1
      filepath = os.path.join(dirpath, filenames[0])
      return filepath

  def save_to_npz(target_path,  data: dict):
    assert os.path.exists(target_path)
    np.savez(os.path.join(target_path, "events.npz"), **data)

  def save_to_h5(target_path, data: dict):
      assert os.path.exists(target_path)

      filter_id = 32001  # Blosc

      compression_level = 1  # {0, ..., 9}
      shuffle = 2  # {0: none, 1: byte, 2: bit}
      # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
      # define BLOSC_BLOSCLZ   0
      # define BLOSC_LZ4       1
      # define BLOSC_LZ4HC     2
      # define BLOSC_SNAPPY    3
      # define BLOSC_ZLIB      4
      # define BLOSC_ZSTD      5
      compressor_type = 5
      compression_opts = (0, 0, 0, 0, compression_level, shuffle, compressor_type)

      with h5py.File(str(target_path+"events.h5"), 'w') as h5f:
          ev_group = 'events'
          #h5f.create_dataset('{}/p'.format(ev_group), data=data['events'][:, 3], compression=filter_id, compression_opts=compression_opts, chunks=True)
          h5f.create_dataset('{}/p'.format(ev_group), data=data['p'], compression=filter_id, compression_opts=compression_opts, chunks=True)
          #h5f.create_dataset('{}/t'.format(ev_group), data=data['events'][:, 2], compression=filter_id, compression_opts=compression_opts, chunks=True)
          h5f.create_dataset('{}/t'.format(ev_group), data=data['t'], compression=filter_id, compression_opts=compression_opts, chunks=True)
          #h5f.create_dataset('{}/x'.format(ev_group), data=data['events'][:, 0], compression=filter_id, compression_opts=compression_opts, chunks=True)
          h5f.create_dataset('{}/x'.format(ev_group), data=data['x'], compression=filter_id, compression_opts=compression_opts, chunks=True)
          #h5f.create_dataset('{}/y'.format(ev_group), data=data['events'][:, 1], compression=filter_id, compression_opts=compression_opts, chunks=True)
          h5f.create_dataset('{}/y'.format(ev_group), data=data['y'], compression=filter_id, compression_opts=compression_opts, chunks=True)
          btn = st.download_button(label="Download Events", data=h5f, file_name="events.h5")
      st.write("Completed")

  # (add the njit decorator here)
  def event_processor(event_data: dict, output_frame):
    for x,y,t,p in subevents.items():
      if p == 1: #positive events
        output_frame[x,y,0] = 255
        output_frame[x,y,1] = 0
        output_frame[x,y,2] = 0
      elif p == -1: #negative events
        output_frame[x,y,0] = 0
        output_frame[x,y,1] = 0
        output_frame[x,y,2] = 255
      else:
        continue
    return output_frame

  def print_inventory(dct):
      print("Items held:")
      for item, amount in dct.items():  # dct.iteritems() in Python 2
          print("{} ({})".format(item, amount))

  def process_dir(outdir, indir, args):
      print(f"Processing folder {indir}... Generating events in {outdir}")
      os.makedirs(outdir, exist_ok=True)

      # constructor
      esim = EventSimulator_torch(args["contrast_threshold_negative"],
                                            args["contrast_threshold_positive"],
                                            args["refractory_period_ns"])

      timestamps = np.genfromtxt(os.path.join(indir, "video_upload/timestamps.txt"), dtype="float64")
      timestamps_ns = (timestamps * 1e9).astype("int64")
      timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

      image_files = sorted(glob.glob(os.path.join(indir, "video_upload/imgs", "*.png")))
      
      # pbar = tqdm.tqdm(total=len(image_files)-1)
      # num_events = 0

      # counter = 0
      # for image_file, timestamp_ns in zip(image_files, timestamps_ns):
      #     image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
      #     log_image = np.log(image.astype("float32") / 255 + 1e-5)
      #     log_image = torch.from_numpy(log_image).cuda()

      #     sub_events = esim.forward(log_image, timestamp_ns)

      #     # for the first image, no events are generated, so this needs to be skipped
      #     if sub_events is None:
      #         continue

      #     sub_events = {k: v.cpu() for k, v in sub_events.items()}    
      #     num_events += len(sub_events['t'])
  
      #     # do something with the events
      #     np.savez(os.path.join(outdir, "%010d.npz" % counter), **sub_events)
      #     pbar.set_description(f"Num events generated: {num_events}")
      #     pbar.update(1)
      #     counter += 1

      images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
      log_images = np.log(images.astype("float32") / 255 + 1e-4)
      log_images = torch.from_numpy(log_images).cuda()
      
      # generate events with GPU support
      print("Generating events")
      generated_events = esim.forward(log_images, timestamps_ns)
      #events = esim.forward(log_image, timestamp_ns)
      #print_inventory(generated_events)
      generated_events = {k: v.cpu() for k, v in generated_events.items()}
      print_inventory(generated_events)
      #save_to_h5(args["output_dir"], generated_events)
      save_to_npz(args["output_dir"], generated_events)
      

  args = {
      "contrast_threshold_negative": 0.2,
      "contrast_threshold_positive": 0.2,
      "refractory_period_ns": 0,
      "input_dir": "new_data/upsampled/",
      "output_dir": "new_data/events/"
  }


  generate_button = st.button("Generate Events")
  if generate_button:

      #Step 1:
      if video_file is not None:
        #read the video file
        #video_object = VideoSequence(video_file.getvalue())
        #saving the uploaded file on the server
        save_path = "new_data/original/video_upload/"
        completeName = os.path.join(save_path, video_file.name)
        with open(completeName, "wb") as file:
          file.write(video_file.getvalue())
        st.success("Uploaded video file saved on the server")

      #Step 2:
      if upsampling:
        print("inside upsampling")
        os.system("python upsampling/upsample.py --input_dir=new_data/original/ --output_dir=new_data/upsampled --device=cuda:0")

      #Step 3: Event Generation
      process_dir(args["output_dir"], args["input_dir"], args)

      #Step 4: Download Option
      with open(os.path.join(args["output_dir"], "events.npz"), "rb") as file:
        btn = st.download_button(label="Download Events", data=file, file_name="events.npz")
        if btn:
          st.markdown(':beer::beer::beer::beer::beer::beer::beer::beer::beer::beer::beer:')



elif add_selectbox == "Live Webcam":

  st.sidebar.subheader('ESIM Settings')
  ct_options = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
  st.sidebar.subheader('Positive Contrast Threshold')
  ct_p = st.sidebar.select_slider("Choose the +ve contrast threshold", options=ct_options)
  st.sidebar.subheader('Negative Contrast Threshold')
  ct_n = st.sidebar.select_slider("Choose the -ve contrast threshold", options=ct_options)
  window_size_options = [24, 30, 60, 80, 100, 120]
  st.sidebar.subheader('Window Size (Number of Frames)')
  window_size = st.sidebar.select_slider("Choose the window size or number of frames for event generation", options=window_size_options)


  esim = EventSimulator_torch(0.5, 0.5, 0)
  FRAME_WINDOW = st.image([])
  cam = cv2.VideoCapture(0)
  
  frame_count = 0
  #frames = np.empty(shape=(7,))
  #timestamps = np.empty(shape=(7,))
  frame_log = []
  timestamps_ns = []
  window_size = 10
  while True:
    # esim = EventSimulator_torch(0.4, 0.4, 0)
    if frame_count <= window_size:
      ret, frame = cam.read()
      if not ret:
        continue
      if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #FRAME_WINDOW.image(frame)
        #st.write(cam.get(cv2.CAP_PROP_POS_MSEC))
        #st.markdown(f"FPS: {cam.get(cv2.CAP_PROP_POS_MSEC)}")

        frame_log_new = np.log(frame.astype("float32") / 255 + 1e-5)
        #frame_log = torch.from_numpy(frame_log).cuda()
        #frame_log.append(frame_log_new)
        frame_log.append(frame_log_new)

        timestamp_curr = cam.get(cv2.CAP_PROP_POS_MSEC)
        timestamps_ns_new = (timestamp_curr * 1e9)
        #timestamps_ns_new = np.array(timestamps_ns, dtype="int64")
        #print(type(timestamp_curr))  
        #timestamps_ns = torch.from_numpy(timestamps_ns).cuda()
        # timestamps_ns.append(timestamps_ns_new)
        timestamps_ns.append(timestamps_ns_new)

        print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count += 1

    else:
      #Event Generation
      # esim = EventSimulator_torch(0.2, 0.2, 0)

      #render events as frames
      frame_log_torch_tensor = torch.from_numpy(np.array(frame_log)).cuda()
      timestamps_ns_torch_tensor = torch.from_numpy(np.array(timestamps_ns, dtype="int64")).cuda()
      generated_events = esim.forward(frame_log_torch_tensor, timestamps_ns_torch_tensor)
      generated_events = {k: v.cpu() for k, v in generated_events.items()}
      generated_events = np.stack([generated_events['x'], generated_events['y'], generated_events['t'], generated_events['p']], -1)
      generated_events = Events((int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))), generated_events)
      event_rendering = generated_events.render()
      print(event_rendering.shape)
      FRAME_WINDOW.image(event_rendering)

      frame_count = 0
      frame_log.clear()
      timestamps_ns.clear()