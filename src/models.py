import csv
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple, Type, ClassVar

from src.loss_functions import *

from matplotlib import pyplot as plt  # type:ignore
import numpy as np
import torch
from torch import nn


@dataclass
class exp_parameters_simplified:
    # Based on Arri LogC3
    base: float
    offset: float
    scale: float
    slope: float
    intercept: float
    cut: float

    def __str__(self):
        return f"base={self.base:0.3f} offset={self.offset:0.3f} scale={self.scale:0.3f} slope={self.slope:0.3f} intercept={self.intercept:0.3f} cut={self.cut:0.3f}"

    def log_curve_to_str(self):
        output = f"""
const float base = {self.base};
const float offset = {self.offset};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};


const float y_cut = slope * cut + intercept;
if (y < y_cut) {{
    return (y - intercept) / slope;
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

if (x < cut) {{
    return slope * x + intercept;
}} else {{
    return _powf(base, x) * scale + offset;
}}
"""
        return output

    def to_csv(self):
        output = f"""name,value
base,{self.base}
offset,{self.offset}
scale,{self.scale}
slope,{self.slope}
intercept,{self.intercept}
cut,{self.cut}
"""
        return output

    @staticmethod
    def from_csv(csv_fn: str) -> "exp_parameters_simplified":
        parsed_dict = {}
        with open(csv_fn, "r") as f:
            dict_reader = csv.DictReader(f)
            for d in dict_reader:
                parsed_dict[d["name"]] = float(d["value"])
        return exp_parameters_simplified(
            base=parsed_dict["base"],
            offset=parsed_dict["offset"],
            scale=parsed_dict["scale"],
            slope=parsed_dict["slope"],
            intercept=parsed_dict["intercept"],
            cut=parsed_dict["cut"],
        )


EXP_INITIAL_GUESS = exp_parameters_simplified(
    base=0.376,
    offset=2.9e-4,
    scale=1.1e-4,
    slope=0.005,
    intercept=0.00343,
    cut=0.1506,
)


class gain_table(nn.Module):
    def __init__(self, num_images: int, device: torch.device, exposures: Optional[torch.Tensor]=None, fixed_exposures: bool=False):
        super(gain_table, self).__init__()
        # Store one gain value per image.
        self.table = nn.Embedding(num_images, 1, device=device)
        # let the table learn the different exposures of the images from a neutral starting point
        self.table.weight.data.fill_(0.0)
        self.num_images = torch.tensor(num_images, dtype=torch.int32)
        if exposures is not None:
            self.table.weight.data = exposures.to(device)
            self.table.requires_grad_(not fixed_exposures)

    def forward(self, x, neutral_idx):
        # check that x is not equal to the frozen one, look up the others in the table.
        if type(neutral_idx) != torch.Tensor:
            neutral_idx = torch.tensor(neutral_idx, dtype=torch.int32)
        neutral_gain = self.table(neutral_idx)
        table_result = self.table(x) - neutral_gain  # (n, 1)
        return torch.pow(torch.tensor(2.0, dtype=torch.float32), table_result).type(
            torch.float32
        )

    def forward_matrix(self):
        return torch.cat(
            [
                self.forward(torch.arange(0, self.num_images), neutral)
                for neutral in torch.arange(0, self.num_images)
            ],
            axis=1,
        )

    def get_gains(self, neutral_idx):
        return (
            torch.log2(self.forward(torch.arange(0, self.num_images), neutral_idx))
            .detach()
            .numpy()
        )


class exp_function_simplified(nn.Module):
    def __init__(self, parameters: exp_parameters_simplified):
        super(exp_function_simplified, self).__init__()
        self.base = nn.parameter.Parameter(torch.log10(torch.tensor(parameters.base)))
        self.offset = nn.parameter.Parameter(torch.tensor(parameters.offset))
        self.scale = nn.parameter.Parameter(torch.tensor(parameters.scale))
        self.slope = nn.parameter.Parameter(torch.tensor(parameters.slope))
        self.intercept = nn.parameter.Parameter(torch.tensor(parameters.intercept))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def compute_intermediate_values(self):
        base = torch.pow(torch.tensor(10.0), self.base)
        offset = self.offset
        scale = self.scale
        intercept = self.intercept
        cut = self.cut
        # slope = torch.abs(scale * torch.pow(base, cut) + offset - intercept) / torch.abs(cut)
        slope = (scale * torch.pow(base, cut) + offset - intercept) / torch.abs(cut)

        # self.cut = nn.parameter.Parameter(cut, requires_grad=False)
        return base, offset, scale, slope, intercept, cut

    def forward(self, t):
        base, offset, scale, slope, intercept, cut = self.compute_intermediate_values()
        interp = (t > cut).float()
        pow_value = scale * torch.pow(base, t) + offset
        lin_value = slope * t + intercept
        output = interp * pow_value + (1 - interp) * lin_value
        # output = torch.clamp(output, min=1e-6)
        return output

    def reverse(self, y):
        base, offset, scale, slope, intercept, cut = self.compute_intermediate_values()
        y_cut = slope * cut + intercept
        interp = (y > y_cut).float()
        log_value = torch.log(torch.clamp((y - offset) / scale, min=1e-7)) / torch.log(
            base
        )
        lin_value = (y - intercept) / slope
        output = interp * log_value + (1 - interp) * lin_value
        # output = torch.clamp(output, 1e-6, 1.0)
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def get_log_parameters(self) -> exp_parameters_simplified:
        base, offset, scale, slope, intercept, cut = self.compute_intermediate_values()
        return exp_parameters_simplified(
            base=float(base),
            offset=float(offset),
            scale=float(scale),
            slope=float(slope),
            intercept=float(intercept),
            cut=float(cut),
        )

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
        mid_gray: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = low_cut_penalty(black_point, self)
        loss += high_intercept_penalty(self)
        # loss += 0.1 * smoothness_penalty(model)
        if mid_gray is not None:
            loss += torch.pow(
                self.forward(mid_gray) - torch.tensor(0.18), torch.tensor(2.0)
            )
        return loss


@dataclass
class legacy_exp_function_parameters:
    # Based on ACESlog
    x_shift: float
    y_shift: float
    scale: float
    cut: float
    slope: float
    intercept: float

    def __str__(self):
        return f"x_shift={self.x_shift:0.3f} y_shift={self.y_shift:0.3f} scale={self.scale:0.3f} slope={self.slope:0.3f} intercept={self.intercept:0.3f} cut={self.cut:0.3f}"

    def log_curve_to_str(self):
        output = f"""
const float x_shift = {self.x_shift};
const float y_shift = {self.y_shift};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};
float tmp;
if (x < cut) {{
    tmp = (x - intercept) / slope;
}} else {{
    tmp = x;
}}
return (_log2f(tmp - x_shift) - y_shift) / scale;
"""
        return output

    def exp_curve_to_str(self):
        output = f"""
const float x_shift = {self.x_shift};
const float y_shift = {self.y_shift};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};

float tmp = _powf(2.0, x * scale + y_shift) + x_shift;
if (tmp < cut) {{
    return tmp * slope + intercept;
}} else {{
    return tmp;
}}
"""
        return output

    def to_csv(self):
        output = f"""name,value
x_shift,{self.x_shift}
y_shift,{self.y_shift}
scale,{self.scale}
slope,{self.slope}
intercept,{self.intercept}
cut,{self.cut}
"""
        return output

    @staticmethod
    def from_csv(csv_fn: str) -> "legacy_exp_function_parameters":
        parsed_dict = {}
        with open(csv_fn, "r") as f:
            dict_reader = csv.DictReader(f)
            for d in dict_reader:
                parsed_dict[d["name"]] = float(d["value"])
        return legacy_exp_function_parameters(
            x_shift=parsed_dict["x_shift"],
            y_shift=parsed_dict["y_shift"],
            scale=parsed_dict["scale"],
            slope=parsed_dict["slope"],
            intercept=parsed_dict["intercept"],
            cut=parsed_dict["cut"],
        )


LEGACY_EXP_INTIAL_GUESS = legacy_exp_function_parameters(
    x_shift=-3.1,
    y_shift=0.2,
    scale=6.0,
    cut=0.18,
    slope=2.1,
    intercept=-0.5,
)


class legacy_exp_function(nn.Module):
    """
    Based on the format of the old ACESlog from back in the day.
    https://github.com/ampas/aces-dev/blob/f65f8f66fbf9165c6317d9536007f2b8ae3899fb/transforms/ctl/acesLog/aces_to_acesLog16i.ctl
    """

    def __init__(self, parameters: legacy_exp_function_parameters):
        super(legacy_exp_function, self).__init__()
        self.x_shift = nn.parameter.Parameter(torch.tensor(parameters.x_shift))
        self.y_shift = nn.parameter.Parameter(torch.tensor(parameters.y_shift))
        self.scale = nn.parameter.Parameter(torch.tensor(parameters.scale))
        self.slope = nn.parameter.Parameter(torch.tensor(parameters.slope))
        self.intercept = nn.parameter.Parameter(torch.tensor(parameters.intercept))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def compute_intermediate_values(self):
        x_shift = self.x_shift
        y_shift = self.y_shift
        scale = self.scale
        intercept = self.intercept
        cut = self.cut
        # cut = cut * slope + intercept
        # Solve for slope
        slope = (self.cut - self.intercept) / torch.abs(self.cut)
        return x_shift, y_shift, scale, slope, intercept, cut

    def forward(self, t):
        # log to lin
        (
            x_shift,
            y_shift,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        pow_value = torch.pow(2.0, (t * scale + y_shift)) + x_shift
        interp = (pow_value > cut).float()
        lin_value = pow_value * slope + intercept
        output = interp * pow_value + (1 - interp) * lin_value
        return output

    def reverse(self, y):
        (
            x_shift,
            y_shift,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        lin_value = (y - intercept) / slope
        interp = (y > cut).float()
        tmp = interp * y + (1.0 - interp) * lin_value
        output = (torch.log2(torch.clamp(tmp - x_shift, 1e-7)) - y_shift) / scale
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
        mid_gray: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (
            x_shift,
            y_shift,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        loss = torch.abs(
            torch.minimum(cut - black_point, torch.zeros_like(black_point))
        )
        # loss += torch.maximum(intercept, torch.zeros_like(intercept))**2
        if mid_gray is not None:
            loss += torch.pow(
                self.forward(mid_gray) - torch.tensor(0.18), torch.tensor(2.0)
            )
        return loss

    def get_log_parameters(self) -> legacy_exp_function_parameters:
        (
            x_shift,
            y_shift,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        return legacy_exp_function_parameters(
            x_shift=float(x_shift),
            y_shift=float(y_shift),
            scale=float(scale),
            slope=float(slope),
            intercept=float(intercept),
            cut=float(cut),
        )


@dataclass
class gamma_function_parameters:
    # Based on BT1886 EOTF
    # Strictly speaking, the EOTF function looks more like the log2lin function `exp_curve_to_str` below. This is because the regression works better when the linearization function is simple, but actually the two forms are basically equivalent.
    gamma: float
    offset: float
    scale: float
    cut: float
    slope: float
    intercept: float

    def __str__(self):
        return f"gamma={self.gamma:0.3f} offset={self.offset:0.3f} scale={self.scale:0.3f} slope={self.slope:0.3f} intercept={self.intercept:0.3f} cut={self.cut:0.3f}"

    def log_curve_to_str(self):
        # lin to log
        output = f"""
const float gamma = {self.gamma};
const float offset = {self.offset};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};
const float y_cut = cut * slope + intercept;

if (x < y_cut) {{
    return (x - intercept) / slope;
}} else {{
    return powf((x - offset) / scale, 1.0 / gamma);
}}
"""
        return output

    def exp_curve_to_str(self):
        # log to lin
        output = f"""
const float gamma = {self.gamma};
const float offset = {self.offset};
const float scale = {self.scale};
const float slope = {self.slope};
const float intercept = {self.intercept};
const float cut = {self.cut};

if (x < cut) {{
    return tmp * slope + intercept;
}} else {{
    return scale * powf(x, gamma) + offset;
}}
"""
        return output

    def to_csv(self):
        output = f"""name,value
gamma,{self.gamma}
offset,{self.offset}
scale,{self.scale}
slope,{self.slope}
intercept,{self.intercept}
cut,{self.cut}
"""
        return output

    @staticmethod
    def from_csv(csv_fn: str) -> "gamma_function_parameters":
        parsed_dict = {}
        with open(csv_fn, "r") as f:
            dict_reader = csv.DictReader(f)
            for d in dict_reader:
                parsed_dict[d["name"]] = float(d["value"])
        return gamma_function_parameters(
            gamma=parsed_dict["gamma"],
            offset=parsed_dict["offset"],
            scale=parsed_dict["scale"],
            slope=parsed_dict["slope"],
            intercept=parsed_dict["intercept"],
            cut=parsed_dict["cut"],
        )


GAMMA_INTIAL_GUESS = gamma_function_parameters(
    gamma=0.45,
    offset=-0.099,
    scale=1.099,
    cut=0.018,
    slope=4.5,
    intercept=0.0,
)


class gamma_function(nn.Module):
    """
    Based on the format of the old ACESlog from back in the day.
    https://github.com/ampas/aces-dev/blob/f65f8f66fbf9165c6317d9536007f2b8ae3899fb/transforms/ctl/acesLog/aces_to_acesLog16i.ctl
    """

    def __init__(self, parameters: gamma_function_parameters):
        super(gamma_function, self).__init__()
        self.gamma = nn.parameter.Parameter(torch.log10(torch.tensor(parameters.gamma)))
        self.offset = nn.parameter.Parameter(torch.tensor(parameters.offset))
        self.scale = nn.parameter.Parameter(torch.tensor(parameters.scale))
        self.slope = nn.parameter.Parameter(torch.tensor(parameters.slope))
        self.intercept = nn.parameter.Parameter(torch.tensor(parameters.intercept))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def compute_intermediate_values(self):
        gamma = torch.pow(torch.tensor(10.0), self.gamma)
        offset = self.offset
        scale = self.scale
        intercept = self.intercept
        cut = self.cut
        # cut = cut * slope + intercept
        # Solve for slope
        slope = (self.cut - self.intercept) / torch.abs(self.cut)
        return gamma, offset, scale, slope, intercept, cut

    def forward(self, t):
        # log to lin
        (
            gamma,
            offset,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        pow_value = torch.pow(torch.clamp(t, min=0.0), gamma) * scale + offset
        interp = (t > cut).float()
        lin_value = t * slope + intercept
        output = interp * pow_value + (1 - interp) * lin_value
        return output

    def reverse(self, y):
        (
            gamma,
            offset,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        pow_value = torch.pow(torch.clamp((y - offset) / scale, min=0.0), 1.0 / gamma)
        lin_value = (y - intercept) / slope
        y_cut = slope * cut + intercept
        interp = (y > y_cut).float()
        output = interp * pow_value + (1 - interp) * lin_value
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
        mid_gray: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (
            gamma,
            offset,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        loss = torch.abs(
            torch.minimum(cut - black_point, torch.zeros_like(black_point))
        )
        if mid_gray is not None:
            loss += torch.pow(
                self.forward(mid_gray) - torch.tensor(0.18), torch.tensor(2.0)
            )
        return loss

    def get_log_parameters(self) -> gamma_function_parameters:
        (
            gamma,
            offset,
            scale,
            slope,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        return gamma_function_parameters(
            gamma=float(gamma),
            offset=float(offset),
            scale=float(scale),
            slope=float(slope),
            intercept=float(intercept),
            cut=float(cut),
        )


parameters_type = Union[
    exp_parameters_simplified, legacy_exp_function_parameters, gamma_function_parameters
]
model_type = Union[
    exp_function_simplified, legacy_exp_function, gamma_function
]

MODEL_DICT = {
    "exp_function": (
        exp_function_simplified,
        exp_parameters_simplified,
        EXP_INITIAL_GUESS,
    ),
    "legacy": (
        legacy_exp_function,
        legacy_exp_function_parameters,
        LEGACY_EXP_INTIAL_GUESS,
    ),
    "gamma": (
        gamma_function,
        gamma_function_parameters,
        GAMMA_INTIAL_GUESS,
    ),
}


def model_selector(model_name: str, initial_guess_fn: Optional[str]) -> model_type:
    model_class, parameter_class, initial_guess = MODEL_DICT[model_name]
    if initial_guess_fn is not None:
        initial_parameters = parameter_class.from_csv(initial_guess_fn)
    else:
        initial_parameters = initial_guess
    model = model_class(initial_parameters)
    return model


def low_cut_penalty(black_point, model: exp_function_simplified):
    _, _, _, _, _, model_cut = model.compute_intermediate_values()
    return torch.abs(
        torch.minimum(model_cut - black_point, torch.zeros_like(black_point))
    )


def high_intercept_penalty(model: exp_function_simplified):
    _, _, _, _, intercept, _ = model.compute_intermediate_values()
    return torch.maximum(intercept, torch.zeros_like(intercept)) ** 2


def smoothness_penalty(model: exp_function_simplified):
    base, offset, scale, slope, intercept, cut = model.compute_intermediate_values()
    # y_cut = slope * cut + intercept
    # log_slope = slope / scale * 1.0 / (torch.clamp((y_cut - offset) / scale, min=1e-7) * torch.log(base))
    # return torch.abs(log_slope - (1.0/slope))
    # pow_slope = scale * torch.exp(torch.log(base) * cut)
    pow_slope = scale * torch.pow(base, cut) * torch.log(base)
    return (slope - pow_slope) ** 2


def plot_log_curve(model: Union[exp_function_simplified, legacy_exp_function]):
    # Plot the log curve as a sanity check
    x_values = np.linspace(start=-8, stop=8, num=4000)
    x_values_lin = 0.18 * (2**x_values)
    with torch.no_grad():
        y_values = model.reverse(torch.tensor(x_values_lin)).detach().numpy()
        cut = model.get_log_parameters().cut
    plt.plot(x_values[y_values < cut], y_values[y_values < cut], color="red")
    plt.plot(x_values[y_values >= cut], y_values[y_values >= cut], color="blue")
    plt.axvline(x=0.0)
    plt.axhline(y=0.0)
    plt.title("Log curve")
    plt.show()

    plt.plot(x_values_lin[y_values < cut], y_values[y_values < cut], color="red")
    plt.plot(x_values_lin[y_values >= cut], y_values[y_values >= cut], color="blue")
    plt.axvline(x=0.18, alpha=0.5)
    plt.axvline(x=0.0)
    plt.axhline(y=0.0)
    plt.title("lin/log curve")
    plt.show()


def plot_data(x, y):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x, y)
