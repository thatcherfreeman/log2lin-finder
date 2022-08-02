import torch
from torch import nn
from torch.utils import data
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt
import argparse
import sys
from typing import Dict, List, Tuple
from lut_parser import lut_1d_properties



# y = ((10^((x - d) / c) - b) / a);
# y = ((10^((x/c) - (d/c)) - b) / a);
# y = ((10^(x/c))/(a * 10^(d/c)) - b/a);
# y = (((10^(1/c))^x)/(a * 10^(d/c)) - b/a)

# base = 10^1/c
# offset = -b/a
# scale = 1 / (a * 10^(d/c))

# y = scale * base^x + offset
# x = log((y - offset) / scale) / log(base)

# x = log((y + (b/a)) * (a * 10^(d/c))) / log(base)
# x = log10((y + (b/a)) * (a * 10^(d/c))) / log10(10^(1/c))
# x = log10((y + (b/a)) * (a * 10^(d/c))) / ((1/c)log10(10))
# x = c log10((y + (b/a)) * (a * 10^(d/c)))
# x = log10(((y + (b/a)) * (a * 10^(d/c)))^c)
# x = log10(((y + (b/a))^c * (a^c * 10^(d))))
# x = log10((((ay + b)))^c * (10^(d))))
# x = log10((((ay + b)))^c) + log10(10^(d))))
# x = c * log10(ay + b) + d





# x = (c * _log10f(a * y + b) + d)
# x = (c * _log10f(a * y + b) + log10(10^d))
# x = (_log10f((a * y + b)^c) + log10(10^d))
# x = _log10f((a * y + b)^c * (10^(1/c))^(cd))
# x = _log10f((a * y + b)^c * (base)^(cd))
# x = _log10f((a * y + ba/a)^c * (base)^(cd))
# x = _log10f((a * (y - offset))^c * (base)^(cd))
# x = _log10f((a * (y - offset))^c * (base)^(cd))



# y = ((10^((x - d) / c) - b) / a);
# x = (c * _log10f(a * y + b) + d)
# x = (c * _log10f(a * y + b) + log10f(10^d))
# x = (_log10f((a * y + b)^c) + log10f(10^d))
# x = (_log10f(10^d * (a * y + b)^c))
# x = (c * _log10f(10^(d/c) * (a * y + b)))
# x = (c * _log10f(10^(d/c) * (a * (y + (b / a)))))
# x = (c * _log10f(g * (y + (b / a)))))
# x = _log10f((g * (y + (b / a)))^c)
# x = _log10f((g^c * (y + (b / a)))^c)


# y = scale * base^x + offset
# log((y - offset) / scale) / log(base) = x

# if x < cut:
#   y = slope * x + intercept
# else:
#   y = scale * base^x + offset
#
# intercept = (base^cut + offset) - slope * cut


# y_cut = slope * cut + intercept
# if y < y_cut:
#   x = (y - intercept) / slope
# else:
#   x = log((y - offset) / scale) / log(base)

@dataclass
class exp_parameters_simplified:
    base: float
    offset: float
    scale: float
    slope: float
    intercept: float
    cut: float
    temperature: float = 1.

    def __str__(self):
        return f"base={self.base:0.3f} offset={self.offset:0.3f} scale={self.scale:0.3f} slope={self.slope:0.3f} intercept={self.intercept:0.3f} cut={self.cut:0.3f} temperature={self.temperature:0.3f}"

    def log_curve_to_str(self):
        output = f"""
const float base = {self.base};
const float offset = {self.offset};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};


const float y_cut = slope * cut + intercept
if (y < y_cut) {{
    return = (y - intercept) / slope;
}} else {{
    return _log10f((y - offset) / scale) / _log10f(base);
}}
        """
        return output

    def exp_curve_to_str(self):
        output = f"""
const float base = {self.base};
const float offset = {self.offset};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};

f (t < cut) {{
    return slope * x + intercept;
}} else {{
    return _powf(base, x) * scale + offset;
}}
"""
        return output


@dataclass
class exp_parameters:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    cut: float
    temperature: float = 1.

    def error(self, other) -> float:
        diffs = [self.b / self.a - other.b / other.a, self.c - other.c, self.d - other.d, self.e - other.e, self.f - other.f, self.cut - other.cut]
        return sum([abs(x) for x in diffs])

    def log_curve_to_str(self):
        output = f"""
const float a = {self.a};
const float b = {self.b};
const float c = {self.c};
const float d = {self.d};
const float e = {self.e};
const float f = {self.f};
const float cut = {self.cut};

if (x > cut) {{
    return (c * _log10f(a * x + b) + d);
}} else {{
    return (e * x + f);
}}
        """
        return output

    def exp_curve_to_str(self):
        output = f"""
const float a = {self.a};
const float b = {self.b};
const float c = {self.c};
const float d = {self.d};
const float e = {self.e};
const float f = {self.f};
const float cut = {self.cut};

if (t > (e * cut + f)) {{
    return ((_exp10f((t - d) / c) - b) / a);
}} else {{
    return ((t - f) / e);
}}
"""
        return output


INITIAL_GUESS = exp_parameters(
    a=250.,
    b=-0.729169,
    c=0.247190,
    d=0.385537,
    e=193.235573,
    f=-0.662201,
    cut=0.004201,
    temperature=1.0,
)

INITIAL_GUESS = exp_parameters_simplified(
    # base=11183.0,
    # offset=2.9e-4,
    # scale=1.1e-4,
    # slope=.005,
    # intercept=.00343,
    # cut=0.1506,
    # temperature=1.0,

    base=0.247,
    offset=2.9e-4,
    scale=1.1e-4,
    slope=.005,
    intercept=.00343,
    cut=0.1506,
    temperature=1.0,
)

def dataset_from_1d_lut(lut: lut_1d_properties) -> data.dataset:
    x = torch.arange(0, lut.size, dtype=torch.float) * \
        (lut.domain_max[0] - lut.domain_min[0]) / \
        (lut.size - 1) + \
        lut.domain_min[0]
    y = torch.tensor(lut.contents[:, 0], dtype=torch.float)
    return data.TensorDataset(x, y)


class pixel_dataset(data.Dataset):
    def __init__(self, pixels):
        # pixels is ndarray of dimension (n_images, pixels_per_image, channels_per_image)
        self.pixels = torch.permute(torch.tensor(pixels), (1,0,2)) # shape (pixels_per_image, n_images, channels_per_image)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.pixels[idx]


class gain_table(nn.Module):
    def __init__(self, num_images: int, frozen_image_idx: int, exposures=None):
        super(gain_table, self).__init__()
        # Store one gain value per image.
        self.table = nn.Embedding(num_images, 1)
        self.frozen_idx = frozen_image_idx
        # let the table learn the different exposures of the images from a neutral starting point
        self.table.weight.data.fill_(0.0)
        if exposures is not None:
            self.table.weight.data = exposures
            # self.table.requires_grad_(False)

    def forward(self, x):
        # check that x is not equal to the frozen one, look up the others in the table.
        table_result = self.table(x) # (n, 1)
        table_result[x == self.frozen_idx] *= 0.0
        return torch.pow(2.0, table_result)

    def get_gains(self):
        return self.table.weight.detach().numpy()


class exp_function_simplified(nn.Module):
    def __init__(self, parameters: exp_parameters_simplified):
        super(exp_function_simplified, self).__init__()
        self.base = nn.parameter.Parameter(torch.tensor(parameters.base))
        self.offset = nn.parameter.Parameter(torch.tensor(parameters.offset))
        self.scale = nn.parameter.Parameter(torch.tensor(parameters.scale))
        self.slope = nn.parameter.Parameter(torch.tensor(parameters.slope))
        self.intercept = nn.parameter.Parameter(torch.tensor(parameters.intercept))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))
        self.temperature = nn.parameter.Parameter(torch.tensor(parameters.temperature), requires_grad=True)

    def compute_intermediate_values(self):
        # f = self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a)
        # self.f = nn.parameter.Parameter(f, requires_grad=False)

        # base = torch.pow(10.0, 1/self.c)
        # offset = -self.b / self.a
        # scale = 1 / (self.a * torch.pow(10.0, self.d / self.c))
        # slope = 1 / self.e
        # intercept = -self.f / self.e
        # cut = self.e * cut + f
        base = torch.pow(10.0, 1.0/self.base)
        offset = self.offset
        scale = self.scale
        intercept = self.intercept
        cut = self.cut
        slope = torch.abs(scale * torch.pow(base, cut) + offset - intercept) / torch.abs(cut)
        temperature = self.temperature

        # self.cut = nn.parameter.Parameter(cut, requires_grad=False)
        return base, offset, scale, slope, intercept, cut, temperature

    def forward(self, t):
        base, offset, scale, slope, intercept, cut, temperature = self.compute_intermediate_values()
        if self.training:
            # float between 0 and 1
            interp = torch.sigmoid((t - cut) / torch.exp(temperature))
        else:
            # either 0 or 1
            interp = (t > cut).float()

        pow_value = scale * torch.pow(base, t) + offset
        lin_value = slope * t + intercept
        output = interp * pow_value + (1 - interp) * lin_value
        # output = pow_value
        # output = lin_value
        output = torch.clamp(output, min=1e-6)
        return output

    def reverse(self, y):
        base, offset, scale, slope, intercept, cut, temperature = self.compute_intermediate_values()
        y_cut = slope * cut + intercept
        if self.training:
            # float between 0 and 1
            interp = torch.sigmoid((y - y_cut) / temperature)
        else:
            # either 0 or 1
            interp = (y > y_cut).float()

        log_value = torch.log(torch.clamp((y - offset) / scale, min=1e-6)) / torch.log(base)
        lin_value = (y - intercept) / slope
        output = interp * log_value + (1 - interp) * lin_value
        # output = log_value
        # output = lin_value
        output = torch.clamp(output, 1e-6, 1.0)
        return output

    def get_log_parameters(self) -> exp_parameters_simplified:
        base, offset, scale, slope, intercept, cut, temperature = self.compute_intermediate_values()
        return exp_parameters_simplified(
            base = float(base),
            offset = float(offset),
            scale = float(scale),
            slope = float(slope),
            intercept = float(intercept),
            cut = float(cut),
            temperature = float(temperature),
        )



class exp_function(nn.Module):
    def __init__(self, parameters: exp_parameters):
        super(exp_function, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor(parameters.a))
        self.b = nn.parameter.Parameter(torch.tensor(parameters.b))
        self.c = nn.parameter.Parameter(torch.tensor(parameters.c))
        self.d = nn.parameter.Parameter(torch.tensor(parameters.d))
        self.e = nn.parameter.Parameter(torch.tensor(parameters.e))
        self.f = nn.parameter.Parameter(torch.tensor(parameters.f))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))
        self.temperature = nn.parameter.Parameter(torch.tensor(parameters.temperature), requires_grad=True)

    def compute_intermediate_values(self):
        cut = self.cut
        f = self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a)
        # f = self.f
        self.f = nn.parameter.Parameter(f, requires_grad=False)
        return cut, f

    def forward(self, t):
        # self.d = nn.parameter.Parameter(1. - self.c * torch.log10(self.a + self.b), requires_grad=False)
        # (t > e * cut + f) ? (pow(10, (t - d) / c) - b) / a: (t - f) / e
        # self.f = self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a)
        cut, f = self.compute_intermediate_values()

        if self.training:
            # float between 0 and 1
            interp = torch.sigmoid((t - (self.e * cut + f)) / self.temperature)
        else:
            # either 0 or 1
            interp = (t > (self.e * cut + self.f)).float()

        pow_value = (torch.pow(10., (t - self.d) / self.c) - self.b) / self.a
        lin_value = (t - f) / self.e
        output = interp * pow_value + (1 - interp) * lin_value
        output = pow_value
        output = torch.clamp(output, min=1e-6) #, max=100)
        return output

    def reverse(self, x):
        cut, f = self.compute_intermediate_values()

        if self.training:
            interp = torch.sigmoid((x - cut) / self.temperature)
        else:
            interp = (x > cut).float()
        log_value = self.c * torch.log10(torch.clamp(self.a * x + self.b, min=1e-6)) + self.d
        lin_value = self.e * x + f
        output = interp * log_value + (1 - interp) * lin_value
        output = log_value
        output = torch.clamp(output, 1e-6, 1.0)
        return output

    def get_log_parameters(self) -> exp_parameters:
        return exp_parameters(
            a = float(self.a),
            b = float(self.b),
            c = float(self.c),
            d = float(self.d),
            e = float(self.e),
            f = float(self.f),
            cut = float(self.cut),
            temperature = float(self.temperature)
        )


def percent_error_masked(y, sample_mask):
    # y is of shape (batch_size, n_images, 3), need to compare the pixel values for all images in this set.
    num_valid_entries = torch.sum(sample_mask, axis=1, keepdim=True) # now shape (batch_size, 1, 3)
    y_target = y
    y_target[~sample_mask] = 0.0

    # Compute mean among not-clipped pixels for this x,y coordinate.
    target_pixel_value = torch.sum(y_target, axis=1, keepdim=True) / num_valid_entries # shape (batch_size, 1, 3)
    delta = torch.abs(y - target_pixel_value) / target_pixel_value
    return torch.mean(delta[sample_mask])


def err_fn(y, sample_mask):
    # y is of shape (batch_size, n_images, 3), need to compare the pixel values for all images in this set.
    num_valid_entries = torch.sum(sample_mask, axis=1, keepdim=True) # now shape (batch_size, 1, 3)
    y_target = y
    y_target[~sample_mask] = 0.0

    # Compute mean among not-clipped pixels for this x,y coordinate.
    target_pixel_value = torch.sum(y_target, axis=1, keepdim=True) / num_valid_entries # shape (batch_size, 1, 3)
    delta = torch.abs(torch.log(y_target+1) - torch.log(target_pixel_value+1))
    return torch.mean(delta[sample_mask])

def percent_error_target(y, target_idx, sample_mask):
    y_target = y[:, [target_idx], :] # shape (batch_size, 1, 3)
    answer_mask = torch.ones_like(y, dtype=bool)
    answer_mask[:, [target_idx], :] = 0
    sample_mask = sample_mask & answer_mask
    delta = torch.abs(y - y_target) / y_target
    return torch.mean(delta[sample_mask])

def error_fn_2(y, target_idx, sample_mask):
    y = torch.log(torch.max(y, torch.tensor(1e-8)))
    y_target = y[:, [target_idx], :] # shape (batch_size, 1, 3)

    answer_mask = torch.ones_like(y, dtype=bool)
    answer_mask[:, [target_idx], :] = 0
    sample_mask = sample_mask & answer_mask
    delta = torch.abs(y - y_target)
    return torch.mean(delta[sample_mask])

def reconstruction_error(y, y_original, target_idx, sample_mask):
    # Assume y of shape (batch_size, n_images, 3)
    # assume y_original of same shape
    # assume target_idx points to the one with gain 1.0
    # Assume y and y_original are both the log encoded images, just take MSE.
    answer_mask = torch.ones_like(y, dtype=bool)
    answer_mask[:, [target_idx], :] = 0
    sample_mask = sample_mask & answer_mask & ~torch.isnan(y)

    y_target = y_original[:, [target_idx], :] # (batch_size, 1, 3), just the target image
    delta = torch.abs(y - y_target)
    return torch.mean(delta[sample_mask])

def derive_exp_function_gd_lut(lut: lut_1d_properties, epochs: int = 100, lr=1e-3, use_scheduler=True) -> nn.Module:
    # torch.autograd.set_detect_anomaly(True)
    model = exp_function_simplified(INITIAL_GUESS)
    dl = data.DataLoader(dataset_from_1d_lut(lut), batch_size=lut.size)
    loss_fn = nn.L1Loss(reduction='mean')
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda x: (0.5 if x % 3000 == 0 else 1.0))
    errors = []
    losses = []
    model.train()
    model.eval()
    with tqdm(total=epochs) as bar:
        for e in range(epochs):
            for x, y in dl:
                optim.zero_grad()
                y_pred = model(x)
                loss = torch.log(loss_fn(torch.log(y_pred + 0.5), torch.log(y + 0.5)))

                error = loss_fn(y, y_pred).detach()
                loss.backward()
                optim.step()
            if use_scheduler:
                sched.step()

            if e % 10 == 0:
                bar.update(10)
                bar.set_postfix(loss=float(loss), error=float(error), params=model.get_log_parameters(), lr=sched.get_last_lr())
                errors.append(float(error))
                losses.append(float(loss))


    plt.figure()
    plt.xlim(0, epochs)
    plt.ylim(min(errors[-1], losses[-1]), max(errors[0], losses[0]))
    plt.plot(range(0, epochs, 10), errors)
    plt.plot(range(0, epochs, 10), losses)
    plt.show()

    return model


def derive_exp_function_gd(
    images: np.ndarray,
    ref_image_num: int,
    white_point: float,
    epochs: int = 20,
    lr=1e-3,
    use_scheduler=True,
    exposures=None,
) -> Tuple[nn.Module, nn.Module]:

    # torch.autograd.set_detect_anomaly(True)
    model = exp_function_simplified(INITIAL_GUESS)
    gains = gain_table(images.shape[0], ref_image_num, exposures)
    ds = pixel_dataset(images)
    dl = data.DataLoader(ds, shuffle=True, batch_size=10000)

    optim = torch.optim.Adam(list(model.parameters()) + list(gains.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda x: (0.5 if x % 3000 == 0 else 1.0))
    errors = []
    losses = []
    model.train()
    model.eval()
    for e in range(epochs):
        print(f"Training epoch {e}")
        with tqdm(total=len(ds)) as bar:
            for batch_num, pixels in enumerate(dl):
                # pixels is of shape (batch_size, n_images, n_channels=3), each l0 row has a fixed h,w pixel coordinate.
                batch_size = pixels.shape[0]

                optim.zero_grad()
                img_gains = gains(torch.arange(0, pixels.shape[1])).unsqueeze(0) # (1, n_images, 1)
                sample_mask = pixels < white_point
                y_pred = model(pixels) * img_gains
                reconstructed_image = model.reverse(y_pred)


                # Ideally all of y_pred would be equal
                # loss = percent_error_masked(y_pred, sample_mask)
                # loss = percent_error_target(y_pred, ref_image_num, sample_mask)
                loss = reconstruction_error(reconstructed_image, pixels, ref_image_num, sample_mask)
                loss += (torch.mean(y_pred[:, ref_image_num, :]) - 0.18)**2
                loss.backward()
                optim.step()

                # error = err_fn(y_pred, sample_mask).detach()
                # error = error_fn_2(y_pred, ref_image_num, sample_mask).detach()
                error = reconstruction_error(reconstructed_image, pixels, ref_image_num, sample_mask)

                bar.update(batch_size)
                bar.set_postfix(loss=float(loss), error=float(error), params=model.get_log_parameters(), lr=sched.get_last_lr())
                errors.append(float(error))
                losses.append(float(loss))

            if use_scheduler:
                sched.step()

        # model.temperature -= 0.3

    plt.figure()
    plt.xlim(0, len(errors))
    plt.plot(range(len(errors)), errors)
    plt.plot(range(len(losses)), losses)
    plt.show()

    return gains, model






def plot_data(x, y):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x, y)




if __name__ == "__main__":

    # Display log2lin model's output curve vs original LUT
    ds = dataset_from_1d_lut(lut)
    x, y = ds.tensors
    model = exp_function(params)
    model.eval()
    y_pred = model(x).detach().numpy()
    model.train()
    y_pred_interp = model(x).detach().numpy()
    plt.figure()
    plot_data(x, y)
    plot_data(x, y_pred)
    plot_data(x, y_pred_interp)
    plt.show()

    # Apply lin2log curve to LUT, expect straight line.
    log_model = log_function(model.get_log_parameters())
    log_model.eval()
    x_restored = log_model(y).detach().numpy()
    plt.figure()
    plot_data(x, x)
    plot_data(x, x_restored)
    plt.show()



