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
import tqdm
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
    path = os.path.join(target_path, "events.npz")
    np.savez(path, **data)
    return path

  def save_to_video(target_path, shape, data: dict):
      # just convert to 30 fps, non-overlapping windows
      fps = 30
      tmin, tmax = data['t'][[0,-1]]
      t0 = np.arange(tmin, tmax, 1e6/fps)
      t1, t0 = t0[1:], t0[:-1]
      idx0 = np.searchsorted(data['t'], t0)
      idx1 = np.searchsorted(data['t'], t1)

      path = os.path.join(target_path, "outputvideo.mp4")
      writer = skvideo.io.FFmpegWriter(path)

      red = np.array([255, 0, 0], dtype="uint8")
      blue = np.array([0,0,255], dtype="uint8")

      pbar = tqdm.tqdm(total=len(idx0))
      for i0, i1 in zip(idx0, idx1):
        sub_data = {k: v[i0:i1].cpu().numpy().astype("int32") for k, v in data.items()}
        frame = np.full(shape=shape + (3,), fill_value=255, dtype="uint8")
        event_processor(sub_data['x'], sub_data['y'], sub_data['p'], red, blue, frame)  
        writer.writeFrame(frame)
        pbar.update(1)
      writer.close()

      return path

  def save_to_h5(target_path, data: dict):
      assert os.path.exists(target_path)
      path = str(target_path+"events.h5")
      with h5py.File(path, 'w') as h5f:
          for k, v in data.items():
            h5f.create_dataset(k, data=v)
      return path

  @numba.jit(nopython=True)
  def event_processor(x, y, p, red, blue, output_frame):
    for x_,y_,p_ in zip(x, y, p):
        if p_ == 1:
          output_frame[y_,x_] = blue
        else:
          output_frame[y_,x_] = red
          
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

      shape = images.shape[1:]

      log_images = np.log(images.astype("float32") / 255 + 1e-4)
      log_images = torch.from_numpy(log_images).cuda()
      
      # generate events with GPU support
      print("Generating events")
      generated_events = esim.forward(log_images, timestamps_ns)
      #events = esim.forward(log_image, timestamp_ns)
      #print_inventory(generated_events)
      generated_events = {k: v.cpu() for k, v in generated_events.items()}
      generated_events['t'] = (generated_events['t'] / 1e3).long()
      print_inventory(generated_events)

      if args['format'] == "HDF5":
          return save_to_h5(args["output_dir"], generated_events)
      elif args['format'] == "NPZ":
          return save_to_npz(args["output_dir"], generated_events)
      elif args['format'] == "Rendered Video":
          return save_to_video(args["output_dir"], shape, generated_events)
      else:
          raise ValueError

      

  args = {
      "contrast_threshold_negative": float(ct_n),
      "contrast_threshold_positive": float(ct_p),
      "refractory_period_ns": 0,
      "input_dir": "data/upsampled/",
      "output_dir": "data/events/",
      "format": format
  }


  generate_button = st.button("Generate Events")
  if generate_button:

      #Step 1:
      if video_file is not None:
        #read the video file
        #video_object = VideoSequence(video_file.getvalue())
        #saving the uploaded file on the server
        save_path = "data/original/video_upload/"
        completeName = os.path.join(save_path, video_file.name)
        with open(completeName, "wb") as file:
          file.write(video_file.getvalue())
        st.success("Uploaded video file saved on the server")

      #Step 2:
      if upsampling:
        print("inside upsampling")
        os.system("python ../upsampling/upsample.py --input_dir=data/original/ --output_dir=data/upsampled --device=cuda:0")

      #Step 3: Event Generation
      file_path = process_dir(args["output_dir"], args["input_dir"], args)

      #Step 4: Download Option
      with open(file_path, "rb") as file:
        btn = st.download_button(label="Download Events", data=file, file_name=os.path.basename(file_path))
        if btn:
          st.markdown(':beer::beer::beer::beer::beer::beer::beer::beer::beer::beer::beer:')



elif add_selectbox == "Live Webcam":
  st.title("ESIM Web App Playground")
# st.markdown(':camera::movie_camera::camera::movie_camera::camera::movie_camera::camera::movie_camera::camera::movie_camera:')
  st.sidebar.subheader('ESIM Settings')
  ct_options = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
  st.sidebar.subheader('Positive Contrast Threshold')
  ct_p = st.sidebar.select_slider("Choose the +ve contrast threshold", options=ct_options)
  st.sidebar.subheader('Negative Contrast Threshold')
  ct_n = st.sidebar.select_slider("Choose the -ve contrast threshold", options=ct_options)
  window_size_options = [24, 30, 60, 80, 100, 120]
  st.sidebar.subheader('Window Size (Number of Frames)')
  window_size = st.sidebar.select_slider("Choose the window size or number of frames for event generation", options=window_size_options)


  esim = EventSimulator_torch(float(ct_n), float(ct_p), 0)
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

        # print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    # print(event_rendering.shape)
      FRAME_WINDOW.image(event_rendering)

      frame_count = 0
      frame_log.clear()
      timestamps_ns.clear()