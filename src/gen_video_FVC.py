import os
from glob import glob

import numpy as np
import rootutils
import torch
import yuvio
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.with_FVC_module import WithFVCLitModule

CKPT = "/mnt/md1/user_wayne/NorthBitSave/logs/train/runs/2023-10-02_17-33-20/checkpoints/epoch_026.ckpt"

if __name__ == "__main__":
    video_folder = "/mnt/md1/user_wayne/Corpora/UVG/"
    save_folder = "./results/UVG/"
    video_paths = glob(os.path.join(video_folder, "*.yuv"))

    model = WithFVCLitModule.load_from_checkpoint(CKPT)
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

                y = model.net(y)
                y = y.permute(0, 1, 3, 2).squeeze(0).squeeze(0).cpu().numpy()
                y = (y * 255.0).astype(np.uint8)

                # y = (np.squeeze(y, axis=0) * 255.).astype(np.uint8)
                # uv = (uv * 255.).astype(np.uint8)
                # u, v = np.vsplit(uv, [1])
                # u = np.squeeze(u, axis=0)
                # v = np.squeeze(v, axis=0)

                frame = yuvio.frame((y, u, v), "yuv420p")

                writer.write(frame)
