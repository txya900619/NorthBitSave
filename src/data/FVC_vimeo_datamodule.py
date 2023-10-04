import os
import random
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.transforms import rgb_to_ycbcr


def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(
        combined,
        (
            0,
            max(size[1], image_shape[2]) - image_shape[2],
            0,
            max(size[0], image_shape[1]) - image_shape[1],
        ),
    )
    freesize0 = np.random.randint(0, max(size[0], image_shape[1]) - size[0] + 1)
    freesize1 = np.random.randint(0, max(size[1], image_shape[2]) - size[1] + 1)
    combined_crop = combined_pad[
        :, freesize0 : freesize0 + size[0], freesize1 : freesize1 + size[1]
    ]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])


def random_flip(images, labels):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and np.random.randint(0, 2) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and np.random.randint(0, 2) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels


class VimeoDataset(Dataset):
    def __init__(self, vimeo_folder: str, vimeo_txt_name: str, im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(vimeo_folder, vimeo_txt_name)

        self.im_height = im_height
        self.im_width = im_width

    def get_vimeo(self, vimeo_folder, vimeo_txt_name):
        vimeo_sequnence_folder = os.path.join(vimeo_folder, "sequences")
        vimeo_txt_path = os.path.join(vimeo_folder, vimeo_txt_name)
        with open(vimeo_txt_path) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(vimeo_sequnence_folder, line.rstrip())
            for i in range(3, 8):
                fns_train_input += [f"{y}/im{i}.png"]
                fns_train_ref += [f"{y}/im{i-2}.png"]

        return fns_train_input, fns_train_ref

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(
            input_image, ref_image, [self.im_height, self.im_width]
        )
        input_image, ref_image = random_flip(input_image, ref_image)

        input_yuv = rgb_to_ycbcr(input_image)
        ref_yuv = rgb_to_ycbcr(ref_image)

        return input_yuv[:1], input_yuv[1:3], ref_yuv[:1], ref_yuv[1:3], input_image

    def __len__(self) -> int:
        return len(self.image_input_list)


class FVCVimeoDataModule(LightningDataModule):
    def __init__(
        self,
        vimeo_dir: str,
        train_val_split: Tuple[float, float] = [0.9, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.vimeo_dir = vimeo_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            data_train_val = VimeoDataset(self.vimeo_dir, "sep_trainlist.txt")
            self.data_train, self.data_val = random_split(data_train_val, self.train_val_split)
            self.data_test = VimeoDataset(self.vimeo_dir, "sep_testlist.txt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = VimeoDataset()
