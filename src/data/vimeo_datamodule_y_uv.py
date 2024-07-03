import os
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import ImageReadMode, read_image


class VimeoDataset(Dataset):
    def __init__(self, vimeo_folder: str, vimeo_txt_name: str):
        self.image_input_list = self.get_vimeo(vimeo_folder, vimeo_txt_name)

    def get_vimeo(self, vimeo_folder, vimeo_txt_name):
        vimeo_sequnence_folder = os.path.join(vimeo_folder, "sequences")
        vimeo_txt_path = os.path.join(vimeo_folder, vimeo_txt_name)
        with open(vimeo_txt_path) as f:
            data = f.readlines()

        fns_train_input = []

        for n, line in enumerate(data, 1):
            y = os.path.join(vimeo_sequnence_folder, line.rstrip())
            for i in range(1, 8):
                fns_train_input += [f"{y}/im{i}.png"]

        return fns_train_input

    def __getitem__(self, index):
        return read_image(self.image_input_list[index], mode=ImageReadMode.RGB)

    def __len__(self) -> int:
        return len(self.image_input_list)


class VimeoDataModule(LightningDataModule):
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
