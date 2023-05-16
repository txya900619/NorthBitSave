import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.io import ImageReadMode, read_image

from src.data.components.data_aug import (
    GaussianBlur,
    GaussianNoise,
    JpegCompression,
    Resize,
    ShotNoise,
)
from src.data.components.transforms import rgb_to_ycbcr420


class YUV420ImageDataset(Dataset):
    """Dataset for images, output is YUV420 128x128."""

    def __init__(self, image_path_list: List[str]):
        self.image_path_list = image_path_list
        self.input_transforms = transforms.Compose(
            GaussianBlur(),
            Resize((128 // 2, 128 // 2)),
            transforms.RandomChoice([GaussianNoise(), ShotNoise()]),
            JpegCompression(),
            Resize((128, 128)),
            JpegCompression(),
        )
        self.answer_transforms = transforms.RandomAdjustSharpness(sharpness_factor=2, p=1)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        croped_image = transforms.RandomCrop((128, 128))(image)

        input_image = self.input_transforms(croped_image)
        answer_image = self.answer_transforms(croped_image)

        input_y, input_uv = rgb_to_ycbcr420(input_image)
        answer_y, answer_uv = rgb_to_ycbcr420(answer_image)

        return input_y, input_uv, answer_y, answer_uv

    def __len__(self):
        return len(self.image_path_list)


class CombineYUV420DataModule(LightningDataModule):
    """Combine DIV2K and Flickr2K datasets."""

    def __init__(
        self,
        DIV2K_dir: str,
        flickr2K_dir: str,
        train_val_test_split: List[float, float, float] = [0.8, 0.1, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.DIV2K_dir = DIV2K_dir
        self.flicker2K_dir = flickr2K_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        seed = 42
        if not self.data_train and not self.data_val and not self.data_test:
            DIV2K_images = glob(os.path.join(self.DIV2K_dir, "/*/*.png"))
            flickr2K_images = glob(os.path.join(self.flicker2K_dir, "/*.png"))

            DIV2K_train_dataset, DIV2K_val_dataset, DIV2K_test_dataset = random_split(
                YUV420ImageDataset(DIV2K_images),
                lengths=[len(DIV2K_images) * split for split in self.train_val_test_split],
                generator=torch.Generator().manual_seed(seed),
            )
            flickr2K_train_dataset, flickr2K_val_dataset, flickr2K_test_dataset = random_split(
                YUV420ImageDataset(flickr2K_images),
                lengths=[len(flickr2K_images) * split for split in self.train_val_test_split],
                generator=torch.Generator().manual_seed(seed),
            )

            self.data_train = ConcatDataset(DIV2K_train_dataset, flickr2K_train_dataset)
            self.data_val = ConcatDataset(DIV2K_val_dataset, flickr2K_val_dataset)
            self.data_test = ConcatDataset(DIV2K_test_dataset, flickr2K_test_dataset)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = CombineYUV420DataModule()
