import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image

from src.data.components.data_aug import (
    GaussianBlur,
    GaussianNoise,
    JpegCompression,
    Resize,
    ShotNoise,
)
from src.data.components.transforms import rgb_to_ycbcr420


def random_split_list(
    list: List[Any], train_val_test_split: Tuple[float, float, float] = [0.8, 0.1, 0.1]
):
    np.random.shuffle(list)
    train_list = list[: int(len(list) * train_val_test_split[0])]
    val_list = list[
        int(len(list) * train_val_test_split[0]) : int(
            len(list) * (train_val_test_split[0] + train_val_test_split[1])
        )
    ]
    test_list = list[int(len(list) * (train_val_test_split[0] + train_val_test_split[1])) :]
    return train_list, val_list, test_list


class YUV420ImageDataset(Dataset):
    """Dataset for images, output is YUV420 256x256."""

    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        self.input_image_folder = os.path.join(self.image_folder, "input")
        self.answer_image_folder = os.path.join(self.image_folder, "answer")

        self.input_image_path_list = glob(os.path.join(self.input_image_folder, "*.png"))
        self.answer_image_path_list = glob(os.path.join(self.answer_image_folder, "*.png"))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        to_float = transforms.ConvertImageDtype(torch.float)

        input_image_path = self.input_image_path_list[index]
        answer_image_path = self.answer_image_path_list[index]

        input_image = to_float(read_image(input_image_path, mode=ImageReadMode.RGB))
        answer_image = to_float(read_image(answer_image_path, mode=ImageReadMode.RGB))

        input_y, input_uv = rgb_to_ycbcr420(input_image)
        answer_y, answer_uv = rgb_to_ycbcr420(answer_image)

        return input_y, input_uv, answer_y, answer_uv

    def __len__(self) -> int:
        return len(self.input_image_path_list)


class CombineYUV420DataModule(LightningDataModule):
    """Combine DIV2K and Flickr2K datasets."""

    def __init__(
        self,
        DIV2K_dir: str,
        flickr2K_dir: str,
        train_val_test_split: Tuple[float, float, float] = [0.8, 0.1, 0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sharpness_factor: float = 2,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.DIV2K_dir = DIV2K_dir
        self.flicker2K_dir = flickr2K_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sharpness_factor = sharpness_factor

        self.image_folder = {}
        self.image_folder["train"] = os.path.join("data", "tmp", "train")
        self.image_folder["val"] = os.path.join("data", "tmp", "val")
        self.image_folder["test"] = os.path.join("data", "tmp", "test")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        if (
            os.path.exists(self.image_folder["train"])
            and os.path.exists(self.image_folder["val"])
            and os.path.exists(self.image_folder["test"])
        ):
            return

        os.makedirs(os.path.join(self.image_folder["train"], "input"))
        os.makedirs(os.path.join(self.image_folder["train"], "answer"))
        os.makedirs(os.path.join(self.image_folder["val"], "input"))
        os.makedirs(os.path.join(self.image_folder["val"], "answer"))
        os.makedirs(os.path.join(self.image_folder["test"], "input"))
        os.makedirs(os.path.join(self.image_folder["test"], "answer"))

        input_transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomApply(
                    [
                        transforms.RandomChoice(
                            [
                                GaussianBlur(),
                                Resize((int(256 * 0.8), int(256 * 0.8))),
                                GaussianNoise(),
                                ShotNoise(),
                                transforms.Compose(
                                    [
                                        transforms.ConvertImageDtype(torch.uint8),
                                        JpegCompression(),
                                        transforms.ConvertImageDtype(torch.float),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    p=0.5,
                ),
                Resize((256, 256)),
                transforms.RandomApply(
                    [
                        transforms.ConvertImageDtype(torch.uint8),
                        JpegCompression(),
                        transforms.ConvertImageDtype(torch.float),
                    ],
                    p=0.5,
                ),
            ]
        )
        answer_transforms = transforms.Compose(
            [
                transforms.RandomAdjustSharpness(sharpness_factor=self.sharpness_factor, p=1),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        DIV2K_images = glob(os.path.join(self.DIV2K_dir, "*/*.png"))
        flickr2K_images = glob(os.path.join(self.flicker2K_dir, "*/*.png"))

        DIV2K_train_images, DIV2K_val_images, DIV2K_test_images = random_split_list(
            DIV2K_images, self.train_val_test_split
        )
        (
            flickr2K_train_images,
            flickr2K_val_images,
            flickr2K_test_images,
        ) = random_split_list(flickr2K_images, self.train_val_test_split)

        images = {}
        images["train"] = DIV2K_train_images + flickr2K_train_images
        images["val"] = DIV2K_val_images + flickr2K_val_images
        images["test"] = DIV2K_test_images + flickr2K_test_images

        for split in ["train", "val", "test"]:
            for image_path in images[split]:
                image = read_image(image_path, mode=ImageReadMode.RGB)
                image_file_name = os.path.splitext(os.path.basename(image_path))[0]
                for i in range(57):
                    croped_image = transforms.RandomCrop((256, 256))(image)

                    input_image = input_transforms(croped_image)
                    answer_image = answer_transforms(croped_image)

                    input_image_path = os.path.join(
                        self.image_folder[split], "input", f"{image_file_name}_{i}.png"
                    )
                    answer_image_path = os.path.join(
                        self.image_folder[split], "answer", f"{image_file_name}_{i}.png"
                    )

                    save_image(input_image, input_image_path)
                    save_image(answer_image, answer_image_path)

    def setup(self, stage: str) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = YUV420ImageDataset(self.image_folder["train"])
            self.data_val = YUV420ImageDataset(self.image_folder["val"])
            self.data_test = YUV420ImageDataset(self.image_folder["test"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
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
    _ = CombineYUV420DataModule()
