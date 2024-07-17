from src.lut_parser import lut_1d_properties

import torch
from torch.utils import data


def dataset_from_1d_lut(lut: lut_1d_properties) -> data.Dataset:
    x = torch.tensor(lut.get_x_axis())
    y = torch.tensor(lut.contents[:, 0], dtype=torch.float)
    return data.TensorDataset(x, y)


class pixel_dataset(data.Dataset):
    def __init__(self, pixels):
        # pixels is ndarray of dimension (n_images, pixels_per_image, channels_per_image)
        self.pixels = torch.permute(
            torch.tensor(pixels), (1, 0, 2)
        )  # shape (pixels_per_image, n_images, channels_per_image)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.pixels[idx]
