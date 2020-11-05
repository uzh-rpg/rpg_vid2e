import os
import shutil
from typing import List
import urllib
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from . import Sequence
from .const import mean, std, imgs_dirname
from .model import UNet, backWarp
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str, device: str):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            input_dir: (str): write your description
            output_dir: (str): write your description
            device: (todo): write your description
        """
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir

        self.device = torch.device(device)

        self._load_net_from_checkpoint()

        negmean= [x * -1 for x in mean]
        self.negmean = self._move_to_device(torch.Tensor([x * -1 for x in mean]).view(3, 1, 1), self.device)
        revNormalize = transforms.Normalize(mean=negmean, std=std)
        self.TP = transforms.Compose([revNormalize])

    def _load_net_from_checkpoint(self):
        """
        Loads the flow from file.

        Args:
            self: (todo): write your description
        """
        ckpt_file = 'checkpoint/SuperSloMo.ckpt'

        if not os.path.isfile(ckpt_file):
            print('Downloading SuperSlowMo checkpoint to {} ...'.format(ckpt_file))
            g = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/VID2E/SuperSloMo.ckpt')
            with open(ckpt_file, 'w+b') as ckpt:
                ckpt.write(g.read())
            print('Done with downloading!')
        assert os.path.isfile(ckpt_file)

        self.flowComp = UNet(6, 4)
        self._move_to_device(self.flowComp, self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = UNet(20, 5)
        self._move_to_device(self.ArbTimeFlowIntrp, self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.flowBackWarp_dict = dict()

        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.ArbTimeFlowIntrp.load_state_dict(checkpoint['state_dictAT'])
        self.flowComp.load_state_dict(checkpoint['state_dictFC'])

    def get_flowBackWarp_module(self, width: int, height: int):
        """
        Return flowBack flow

        Args:
            self: (todo): write your description
            width: (int): write your description
            height: (int): write your description
        """
        module = self.flowBackWarp_dict.get((width, height))
        if module is None:
            module  = backWarp(width, height, self.device)
            self._move_to_device(module, self.device)
            self.flowBackWarp_dict[(width, height)] = module
        assert module is not None
        return module

    def upsample(self):
        """
        Upsample all files to disk.

        Args:
            self: (todo): write your description
        """
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue
            sequence_counter += 1
            print('Processing sequence number {}'.format(sequence_counter))
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        """
        Todo : param sequence and sequence.

        Args:
            self: (todo): write your description
            sequence: (todo): write your description
            dest_imgs_dir: (str): write your description
            dest_timestamps_filepath: (str): write your description
        """
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        idx = 0
        for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
            img_pair = self._move_to_device(img_pair, self.device)
            I0 = torch.unsqueeze(img_pair[0], dim=0)
            I1 = torch.unsqueeze(img_pair[1], dim=0)
            t0 = time_pair[0]
            t1 = time_pair[1]

            total_frames = []
            timestamps = []

            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            total_frames.append(self.TP(I0[0]))
            timestamps.append(t0)

            self._upsample_adaptive(I0, I1, t0, t1, F_0_1, F_1_0, total_frames, timestamps)

            sorted_indices = np.argsort(timestamps)

            total_frames = torch.stack([total_frames[j] for j in sorted_indices])
            timestamps = [timestamps[i] for i in sorted_indices]
            total_frames = self._to_numpy_image(total_frames)

            timestamps_list += timestamps
            for frame in total_frames:
                self._write_img(frame, idx, dest_imgs_dir)
                idx += 1
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
        """
        Prepares the output_dir to dest_dir.

        Args:
            self: (todo): write your description
            src_dir: (str): write your description
            dest_dir: (str): write your description
        """
        # Copy directory structure.
        def ignore_files(directory, files):
            """
            Check if a directory recursively recursively.

            Args:
                directory: (str): write your description
                files: (list): write your description
            """
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]
        shutil.copytree(src_dir, dest_dir, ignore=ignore_files)

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        """
        Write an image to disk.

        Args:
            img: (array): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            idx: (int): write your description
            imgs_dir: (str): write your description
        """
        assert os.path.isdir(imgs_dir)
        path = os.path.join(imgs_dir, "%08d.png" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        """
        Write timestamps to timestamps file.

        Args:
            timestamps: (todo): write your description
            timestamps_filename: (str): write your description
        """
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])

    @staticmethod
    def _to_numpy_image(img: torch.Tensor):
        """
        Convert numpy array

        Args:
            img: (array): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
        """
        img = np.clip(255 * img.cpu().numpy(), 0, 255).astype(np.uint8)
        img = np.transpose(img, (0, 2, 3, 1))
        return img

    def _upsample_adaptive(self,
                           I0: torch.Tensor,
                           I1: torch.Tensor,
                           time0: torch.Tensor,
                           time1: torch.Tensor,
                           F_0_1: torch.Tensor,
                           F_1_0: torch.Tensor,
                           total_frames: List[torch.Tensor],
                           timestamps: List[float]):
        """
        Takes the mean of the sensor.

        Args:
            self: (todo): write your description
            I0: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            I1: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            time0: (float): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            time1: (float): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            F_0_1: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            F_1_0: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            total_frames: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
            timestamps: (int): write your description
        """
        B, _, _, _ = F_0_1.shape

        flow_mag_0_1_max, _ = F_0_1.pow(2).sum(1).pow(.5).view(B,-1).max(-1)
        flow_mag_1_0_max, _ = F_1_0.pow(2).sum(1).pow(.5).view(B,-1).max(-1)

        flow_mag_max, _ = torch.stack([flow_mag_0_1_max, flow_mag_1_0_max]).max(0)
        flow_mag_max = torch.ceil(flow_mag_max).int()

        for i in range(B):
            for intermediateIndex in range(1, flow_mag_max[i].item()):
                t = float(intermediateIndex) / flow_mag_max[i].item()
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                height, width = I0.shape[-2:]
                flow_back_warp = self.get_flowBackWarp_module(width, height)
                g_I0_F_t_0 = flow_back_warp(I0, F_t_0)
                g_I1_F_t_1 = flow_back_warp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = flow_back_warp(I0, F_t_0_f)
                g_I1_F_t_1_f = flow_back_warp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                Ft_p_norm = Ft_p[i] - self.negmean

                total_frames += [Ft_p_norm]
                timestamps += [(time0 + t * (time1 - time0))]

    @classmethod
    def _move_to_device(
            cls,
            _input,
            device: torch.device,
            dtype: torch.dtype = None):
        """
        Move input to device.

        Args:
            cls: (todo): write your description
            _input: (todo): write your description
            device: (str): write your description
            torch: (todo): write your description
            device: (str): write your description
            dtype: (todo): write your description
            torch: (todo): write your description
            dtype: (todo): write your description
        """
        if not torch.cuda.is_available() and not device == torch.device('cpu'):
            warnings.warn("CUDA not available! Input remains on CPU!", Warning)

        if isinstance(_input, torch.nn.Module):
            # Performs in-place modification of the module but we still return for convenience.
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, torch.Tensor):
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, list):
            return [cls._move_to_device(v, device=device, dtype=dtype) for v in _input]
        warnings.warn("Instance type '{}' not supported! Input remains on current device!".format(type(_input)), Warning)
        return _input
