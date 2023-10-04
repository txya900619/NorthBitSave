import math
import os
from glob import glob

import numpy as np
import rootutils
import scipy
import torch
import torch.nn.functional as F
import yuvio
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.bitsave_module import BitSaveLitModule


def ycbcr420_to_444(y, uv, order=1):
    """
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    """
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    """
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    """
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h // 2, 2, w // 2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h // 2, 2, w // 2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0.0, 1.0)
    uv = np.clip(uv, 0.0, 1.0)

    return y, uv


def img_pad(x, w_pad, h_pad, w_odd_pad, h_odd_pad):
    """Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset.

    For the evaluation on the UHDM, you can also try the commented lines where the mean values are
    calculated from UHDM training set, yielding similar performance.
    """
    x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
    # x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.5165)
    # x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4952)
    # x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4695)
    y = torch.cat([x1, x2, x3], dim=1)

    return y


CKPT = "/mnt/md1/user_wayne/NorthBitSave/logs/train/runs/2023-09-22_14-41-54/checkpoints/epoch_016.ckpt"

if __name__ == "__main__":
    video_folder = "/mnt/md1/user_wayne/Corpora/UVG/"
    save_folder = "./results/UVG/"
    video_paths = glob(os.path.join(video_folder, "*.yuv"))

    model = BitSaveLitModule.load_from_checkpoint(CKPT)
    model.eval().cuda()

    for video_path in tqdm(video_paths):
        # for video_path in tqdm(["/mnt/md1/user_wayne/Corpora/UVG/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"]):
        filename = os.path.basename(video_path)
        save_path = os.path.join(save_folder, filename)

        reader = yuvio.get_reader(video_path, 1920, 1080, "yuv420p")
        writer = yuvio.get_writer(save_path, 1920, 1080, "yuv420p")

        for yuv_frame in reader:
            with torch.no_grad():
                y, u, v = yuv_frame
                y = np.expand_dims(y / 255.0, axis=0)
                # u = np.expand_dims(u / 255., axis=0)
                # v = np.expand_dims(v / 255., axis=0)
                y = torch.from_numpy(y).unsqueeze(0).permute(0, 1, 3, 2).float().cuda()
                # yuv = ycbcr420_to_444(y, np.concatenate((u, v), axis=0))
                # yuv = torch.from_numpy(yuv).unsqueeze(0).permute(0, 1, 3, 2).float().cuda()
                # b, c, h, w = yuv.size()
                # w_pad = (math.ceil(w/32)*32 - w) // 2
                # h_pad = (math.ceil(h/32)*32 - h) // 2
                # w_odd_pad = w_pad
                # h_odd_pad = h_pad
                # if w % 2 == 1:
                #     w_odd_pad += 1
                # if h % 2 == 1:
                #     h_odd_pad += 1
                # yuv = img_pad(yuv, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
                # yuv =

                # yuv, _, _ = model(yuv)
                # if h_pad != 0:
                #     yuv = yuv[:, :, h_pad:-h_odd_pad, :]
                # if w_pad != 0:
                #     yuv = yuv[:, :, :, w_pad:-w_odd_pad]
                # yuv = yuv.permute(0, 1, 3, 2).squeeze(0)
                # yuv = yuv.cpu().numpy()
                # y, uv = ycbcr444_to_420(yuv)

                y = model(y)
                y = y.permute(0, 1, 3, 2).squeeze(0).squeeze(0).cpu().numpy()
                y = (y * 255.0).astype(np.uint8)

                # y = (np.squeeze(y, axis=0) * 255.).astype(np.uint8)
                # uv = (uv * 255.).astype(np.uint8)
                # u, v = np.vsplit(uv, [1])
                # u = np.squeeze(u, axis=0)
                # v = np.squeeze(v, axis=0)

                frame = yuvio.frame((y, u, v), "yuv420p")

                writer.write(frame)
