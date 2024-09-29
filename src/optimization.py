from typing import Optional, Tuple
from collections import defaultdict

from src.models import model_selector, gain_table
from src.datasets import dataset_from_1d_lut, pixel_dataset
from src.lut_parser import lut_1d_properties
from src.loss_functions import (
    log_lin_image_error,
    negative_linear_values_penalty,
    reconstruction_error,
)

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np


def derive_exp_function_gd_lut(
    lut: lut_1d_properties,
    model_name: str,
    epochs: int = 100,
    lr=1e-3,
    use_scheduler=True,
    initial_parameters_fn=None,
) -> nn.Module:
    # torch.autograd.set_detect_anomaly(True)
    model = model_selector(model_name, initial_parameters_fn)
    dl = data.DataLoader(dataset_from_1d_lut(lut), batch_size=lut.size)
    loss_fn = nn.L1Loss(reduction="mean")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(
        optim, lambda x: (0.5 if x % 3000 == 0 else 1.0)
    )
    errors = []
    losses = []
    model.train()
    with tqdm(total=epochs) as bar:
        for e in range(epochs):
            for x, y in dl:
                optim.zero_grad()
                y_pred = model(x)
                pred_mask = y_pred > 0.0
                loss = torch.log(
                    loss_fn(
                        torch.log(y_pred[pred_mask] + 0.5),
                        torch.log(y[pred_mask] + 0.5),
                    )
                )
                error = loss_fn(y, y_pred).detach()
                loss.backward()
                optim.step()
            if use_scheduler:
                sched.step()

            if e % 10 == 0:
                bar.update(10)
                bar.set_postfix(
                    loss=float(loss),
                    error=float(error),
                    params=model.get_log_parameters(),
                    lr=sched.get_last_lr(),
                )
                errors.append(float(error))
                losses.append(float(loss))

    plt.figure()
    plt.xlim(0, epochs)
    plt.ylim(min(errors[-1], losses[-1]), max(errors[0], losses[0]))
    plt.plot(range(0, epochs, 10), errors)
    plt.plot(range(0, epochs, 10), losses)
    plt.show()

    return model


def derive_exp_function_gd_log_lin_images(
    log_image: np.ndarray,
    lin_image: np.ndarray,
    black_point: float,
    model_name: str,
    epochs: int = 20,
    lr: float = 1e-3,
    use_scheduler: bool = True,
    initial_parameters_fn: Optional[str] = None,
    batch_size: int = 10000,
) -> nn.Module:
    device = get_device()

    model = model_selector(model_name, initial_parameters_fn)
    assert (log_image.shape == lin_image.shape) and (len(log_image.shape) == 2)
    ds = data.TensorDataset(torch.tensor(log_image), torch.tensor(lin_image))
    dl = data.DataLoader(ds, shuffle=True, batch_size=batch_size)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(
        optim, lambda x: (0.5 if x % 20 == 0 else 1.0)
    )
    errors = []
    losses = []
    model.train()
    model = model.to(device)
    black_point_t = torch.tensor(black_point, dtype=torch.float32, device=device)

    for e in range(epochs):
        print(f"Training epoch {e}")
        with tqdm(total=len(ds)) as bar:
            for batch_num, (log_pixels, lin_pixels) in enumerate(dl):
                log_pixels = log_pixels.to(device)
                lin_pixels = lin_pixels.to(device)
                # log_pixels and lin_pixels of shape (batch_size, 3)
                batch_size = log_pixels.shape[0]

                optim.zero_grad()
                lin_pred = model(log_pixels)
                log_pred = model.reverse(lin_pixels)

                # Ideally all of y_pred would be equal
                error = log_lin_image_error(log_pixels, log_pred)
                loss = torch.tensor(0.0, device=device)
                loss += error
                # loss += log_lin_image_error(
                #     torch.log10(torch.max(lin_pixels, torch.tensor(1e-8))),
                #     torch.log10(torch.max(lin_pred, torch.tensor(1e-8))),
                # )
                # loss += negative_linear_values_penalty(lin_pred)
                # loss += 0.1 * model.loss(black_point_t, None)

                loss.backward()
                optim.step()

                bar.update(batch_size)
                bar.set_postfix(
                    loss=float(loss),
                    error=float(error),
                    params=model.get_log_parameters(),
                    lr=sched.get_last_lr(),
                )
                errors.append(float(error))
                losses.append(float(loss))

            if use_scheduler:
                sched.step()

    plt.figure()
    plt.xlim(0, len(errors))
    plt.plot(range(len(errors)), errors)
    plt.plot(range(len(losses)), losses)
    plt.show()
    return model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def derive_exp_function_gd(
    images: np.ndarray,
    white_point: float,
    black_point: float,
    model_name: str,
    epochs: int = 20,
    lr: float = 1e-3,
    use_scheduler: bool = True,
    exposures: Optional[torch.Tensor] = None,
    fixed_exposures: bool = False,
    initial_parameters_fn: Optional[str] = None,
    batch_size: int = 10000,
    restart_optimizer: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    n_images = images.shape[0]
    device = get_device()

    # torch.autograd.set_detect_anomaly(True)
    model = model_selector(model_name, initial_parameters_fn)
    gains = gain_table(n_images, device, exposures, fixed_exposures).to(device)
    ds = pixel_dataset(np.clip(images, black_point, white_point))
    dl = data.DataLoader(ds, shuffle=True, batch_size=batch_size)

    optim = torch.optim.Adam(list(model.parameters()) + list(gains.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(
        optim, lambda x: (0.5 if x % 20 == 0 else 1.0)
    )
    errors = []
    losses = []
    model.train()
    model = model.to(device=device)
    white_point_t = torch.tensor(white_point, dtype=torch.float32, device=device)
    black_point_t = torch.tensor(black_point, dtype=torch.float32, device=device)

    for e in range(epochs):
        if e % 3 == 0 and restart_optimizer:
            print("resetting optimizer state")
            optim = torch.optim.Adam(
                list(model.parameters()) + list(gains.parameters()), lr=lr
            )
        print(f"Training epoch {e}")
        with tqdm(total=len(ds)) as bar:
            for batch_num, pixels in enumerate(dl):
                # pixels is of shape (batch_size, n_images, n_channels=3), each l0 row has a fixed h,w pixel coordinate.
                batch_size = pixels.shape[0]
                pixels = pixels.to(device)

                optim.zero_grad()
                lin_images = model(pixels)

                # All lin images matched to image `ref_image_num` is lin_images_matrix[:, :, ref_image_num, :]
                # lin_images_matrix[:, a, b, :] represents the transformation of image (a) when converted to the exposure of (b).
                lin_images_matrix = torch.stack(
                    [
                        lin_images
                        * gains(torch.arange(0, n_images, device=device), neutral_idx)
                        for neutral_idx in torch.arange(0, n_images, device=device)
                    ],
                    dim=2,
                )  # shape (batch_size, n_images, n_images, n_channels)
                y_pred = lin_images_matrix
                reconstructed_image = model.reverse(
                    y_pred.type(torch.float32).to(device)
                )

                # Ideally all of y_pred would be equal
                error = reconstruction_error(
                    reconstructed_image,
                    pixels[:, :, :].unsqueeze(1),
                    sample_mask=(reconstructed_image < white_point_t)
                    & (reconstructed_image > black_point_t)
                    & (y_pred > torch.tensor(0.0, device=device))
                    & (pixels.unsqueeze(1) < white_point_t),
                    device=device,
                )
                loss = torch.tensor(0.0, device=device)
                loss += error
                loss += 0.1 * negative_linear_values_penalty(y_pred)
                loss += 0.1 * model.loss(black_point_t, white_point_t)

                loss.backward()
                optim.step()

                bar.update(batch_size)
                bar.set_postfix(
                    loss=float(loss.cpu()),
                    error=float(error.cpu()),
                    params=model.cpu().get_log_parameters(),
                    lr=sched.get_last_lr(),
                )
                errors.append(float(error.cpu()))
                losses.append(float(loss.cpu()))

            if use_scheduler:
                sched.step()
    model.to(device=torch.device("cpu"))

    plt.figure()
    plt.xlim(0, len(errors))
    plt.plot(range(len(errors)), errors)
    plt.plot(range(len(losses)), losses)
    plt.show()

    return gains, model
