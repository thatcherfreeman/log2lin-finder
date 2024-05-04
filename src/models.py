import csv
from dataclasses import dataclass, asdict
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
    mid_gray_scaling: float = 1.0

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
const float mid_gray_scaling = {self.mid_gray_scaling};

y /= mid_gray_scaling;
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
const float mid_gray_scaling = {self.mid_gray_scaling};

float out;
if (x < cut) {{
    out = slope * x + intercept;
}} else {{
    out = _powf(base, x) * scale + offset;
}}
out *= mid_gray_scaling;
return out;
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
mid_gray_scaling,{self.mid_gray_scaling}
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
            mid_gray_scaling=parsed_dict.get("mid_gray_scaling", 1.0),
        )


EXP_INITIAL_GUESS = exp_parameters_simplified(
    base=0.376,
    offset=2.9e-4,
    scale=1.1e-4,
    slope=0.005,
    intercept=0.00343,
    cut=0.1506,
    mid_gray_scaling=1.0,
)


class gain_table(nn.Module):
    def __init__(
        self,
        num_images: int,
        device: torch.device,
        exposures: Optional[torch.Tensor] = None,
        fixed_exposures: bool = False,
    ):
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
            .squeeze()
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

    def get_log_parameters(
        self, target_mid_gray: Optional[float] = None
    ) -> exp_parameters_simplified:
        mid_gray_scaling = 1.0
        if target_mid_gray is not None:
            output_mid_gray = self.forward(target_mid_gray)
            mid_gray_scaling = 0.18 / output_mid_gray
        base, offset, scale, slope, intercept, cut = self.compute_intermediate_values()
        return exp_parameters_simplified(
            base=float(base),
            offset=float(offset),
            scale=float(scale),
            slope=float(slope),
            intercept=float(intercept),
            cut=float(cut),
            mid_gray_scaling=float(mid_gray_scaling),
        )

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = low_cut_penalty(black_point, self)
        loss += high_intercept_penalty(self)
        # loss += 0.1 * smoothness_penalty(model)
        return loss


@dataclass
class legacy_exp_function_parameters:
    # Based on ACESlog
    x_shift: float
    y_shift: float
    scale: float
    cut: float
    slope: float
    slope2: float
    intercept: float
    mid_gray_scaling: float = 1.0

    def __str__(self):
        return f"x_shift={self.x_shift:0.3f} y_shift={self.y_shift:0.3f} scale={self.scale:0.3f} slope={self.slope:0.3f} slope2={self.slope2:0.3f} intercept={self.intercept:0.3f} cut={self.cut:0.3f}"

    def log_curve_to_str(self):
        output = f"""
const float x_shift = {self.x_shift};
const float y_shift = {self.y_shift};
const float scale = {self.scale};
const float slope = {self.slope};
const float slope2 = {self.slope2};
const float intercept = {self.intercept};
const float cut = {self.cut};
const float mid_gray_scaling = {self.mid_gray_scaling};
x /= mid_gray_scaling;

float tmp;
if (x / slope2 < cut) {{
    tmp = (x - intercept) / slope;
}} else {{
    tmp = x / slope2;
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
const float slope2 = {self.slope2};
const float intercept = {self.intercept};
const float cut = {self.cut};
const float mid_gray_scaling = {self.mid_gray_scaling};

float tmp = _powf(2.0, x * scale + y_shift) + x_shift;
float out;
if (tmp < cut) {{
    out = tmp * slope + intercept;
}} else {{
    out = tmp * slope2;
}}
out *= mid_gray_scaling;
return out;
"""
        return output

    def to_csv(self):
        output = f"""name,value
x_shift,{self.x_shift}
y_shift,{self.y_shift}
scale,{self.scale}
slope,{self.slope}
slope2,{self.slope2}
intercept,{self.intercept}
cut,{self.cut}
mid_gray_scaling,{self.mid_gray_scaling}
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
            slope2=parsed_dict["slope2"],
            intercept=parsed_dict["intercept"],
            cut=parsed_dict["cut"],
            mid_gray_scaling=parsed_dict.get("mid_gray_scaling", 1.0),
        )


LEGACY_EXP_INTIAL_GUESS = legacy_exp_function_parameters(
    x_shift=-3.1,
    y_shift=0.2,
    scale=6.0,
    cut=0.18,
    slope=2.1,
    slope2=1.0,
    intercept=-0.5,
    mid_gray_scaling=1.0,
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
        self.slope2 = nn.parameter.Parameter(torch.tensor(parameters.slope2))
        self.intercept = nn.parameter.Parameter(torch.tensor(parameters.intercept))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def compute_intermediate_values(self):
        x_shift = self.x_shift
        y_shift = self.y_shift
        scale = self.scale
        intercept = self.intercept
        # cut = self.cut
        slope2 = self.slope2
        # cut * slope2 = cut * slope + intercept
        # Solve for slope
        slope = self.slope
        # slope = torch.abs((self.cut * self.slope2 - self.intercept) / torch.abs(self.cut))
        cut = intercept / torch.abs(slope2 - slope)
        return x_shift, y_shift, scale, slope, slope2, intercept, cut

    def forward(self, t):
        # log to lin
        (
            x_shift,
            y_shift,
            scale,
            slope,
            slope2,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        pow_value = torch.pow(2.0, (t * scale + y_shift)) + x_shift
        interp = (pow_value > cut).float()
        lin_value = pow_value * slope + intercept
        output = interp * pow_value * slope2 + (1 - interp) * lin_value
        return output

    def reverse(self, y):
        (
            x_shift,
            y_shift,
            scale,
            slope,
            slope2,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        lin_value = (y - intercept) / slope
        interp = (y / slope2 > cut).float()
        tmp = interp * y / slope2 + (1.0 - interp) * lin_value
        output = (torch.log2(torch.clamp(tmp - x_shift, 1e-7)) - y_shift) / scale
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (
            x_shift,
            y_shift,
            scale,
            slope,
            slope2,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        loss = torch.abs(
            torch.minimum(cut - black_point, torch.zeros_like(black_point))
        )
        # loss += torch.maximum(intercept, torch.zeros_like(intercept))**2
        return loss

    def get_log_parameters(
        self, target_mid_gray: Optional[float] = None
    ) -> legacy_exp_function_parameters:
        mid_gray_scaling = 1.0
        if target_mid_gray is not None:
            output_mid_gray = self.forward(target_mid_gray)
            mid_gray_scaling = 0.18 / output_mid_gray
        (
            x_shift,
            y_shift,
            scale,
            slope,
            slope2,
            intercept,
            cut,
        ) = self.compute_intermediate_values()
        return legacy_exp_function_parameters(
            x_shift=float(x_shift),
            y_shift=float(y_shift),
            scale=float(scale),
            slope=float(slope),
            slope2=float(slope2),
            intercept=float(intercept),
            cut=float(cut),
            mid_gray_scaling=float(mid_gray_scaling),
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
    mid_gray_scaling: float = 1.0

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
const float mid_gray_scaling = {self.mid_gray_scaling};

x /= mid_gray_scaling;
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
const float mid_gray_scaling = {self.mid_gray_scaling};

float out;
if (x < cut) {{
    out = x * slope + intercept;
}} else {{
    out = scale * powf(x, gamma) + offset;
}}
out *= mid_gray_scaling;
return out;
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
mid_gray_scaling,{self.mid_gray_scaling}
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
            mid_gray_scaling=parsed_dict.get("mid_gray_scaling", 1.0),
        )


GAMMA_INTIAL_GUESS = gamma_function_parameters(
    gamma=0.45,
    offset=-0.099,
    scale=1.099,
    cut=0.018,
    slope=4.5,
    intercept=0.0,
    mid_gray_scaling=1.0,
)


class gamma_function(nn.Module):
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
        # pow(cut, gamma) * scale + offset = cut * slope + intercept
        # Solve for slope
        slope = (
            torch.pow(torch.clamp(cut, min=1e-8), gamma) * scale
            + offset
            - self.intercept
        ) / torch.abs(self.cut)
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
        pow_value = torch.pow(torch.clamp(t, min=1e-8), gamma) * scale + offset
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
        pow_value = torch.pow(torch.clamp((y - offset) / scale, min=1e-8), 1.0 / gamma)
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
        return loss

    def get_log_parameters(
        self, target_mid_gray: Optional[float] = None
    ) -> gamma_function_parameters:
        mid_gray_scaling = 1.0
        if target_mid_gray is not None:
            output_mid_gray = self.forward(target_mid_gray)
            mid_gray_scaling = 0.18 / output_mid_gray
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
            mid_gray_scaling=float(mid_gray_scaling),
        )


@dataclass
class pure_exp_parameters:
    # Pure log/power function based on gopro log.
    base: float
    offset: float
    scale: float
    mid_gray_scaling: float = 1.0

    def __str__(self):
        return (
            f"base={self.base:0.3f} offset={self.offset:0.3f} scale={self.scale:0.3f}"
        )

    def log_curve_to_str(self):
        # lin to log
        output = f"""
const float base = {self.base};
const float offset = {self.offset};
const float scale = {self.scale};
const float mid_gray_scaling = {self.mid_gray_scaling};
x /= mid_gray_scaling;

return _log10f((x / scale) - offset) / _log10f(base);
"""
        return output

    def exp_curve_to_str(self):
        # log to lin
        output = f"""
const float base = {self.base};
const float offset = {self.offset};
const float scale = {self.scale};
const float mid_gray_scaling = {self.mid_gray_scaling};

float out = scale * (_powf(base, x) + offset);
out *= mid_gray_scaling;
return out;
"""
        return output

    def to_csv(self):
        output = f"""name,value
base,{self.base}
offset,{self.offset}
scale,{self.scale}
mid_gray_scaling,{self.mid_gray_scaling}
"""
        return output

    @staticmethod
    def from_csv(csv_fn: str) -> "pure_exp_parameters":
        parsed_dict = {}
        with open(csv_fn, "r") as f:
            dict_reader = csv.DictReader(f)
            for d in dict_reader:
                parsed_dict[d["name"]] = float(d["value"])
        return pure_exp_parameters(
            base=parsed_dict["base"],
            offset=parsed_dict["offset"],
            scale=parsed_dict["scale"],
            mid_gray_scaling=parsed_dict.get("mid_gray_scaling", 1.0),
        )


PURE_EXP_INTIAL_GUESS = pure_exp_parameters(
    base=100,
    offset=-1,
    scale=0.01,
    mid_gray_scaling=1.0,
)


class pure_exp_function(nn.Module):
    def __init__(self, parameters: pure_exp_parameters):
        super(pure_exp_function, self).__init__()
        self.base = nn.parameter.Parameter(torch.log10(torch.tensor(parameters.base)))
        self.offset = nn.parameter.Parameter(torch.tensor(parameters.offset))
        self.scale = nn.parameter.Parameter(torch.tensor(parameters.scale))

    def compute_intermediate_values(self):
        base = torch.pow(torch.tensor(10.0), self.base)
        offset = self.offset
        scale = self.scale
        return base, offset, scale

    def forward(self, t):
        # log to lin
        (
            base,
            offset,
            scale,
        ) = self.compute_intermediate_values()
        pow_value = (torch.pow(base, t) + offset) * scale
        return pow_value

    def reverse(self, y):
        (
            base,
            offset,
            scale,
        ) = self.compute_intermediate_values()
        log_value = torch.log10(
            torch.clamp(((y / scale) - offset), 1e-7)
        ) / torch.log10(base)
        output = torch.clamp(log_value, 0.0, 1.0)
        return output

    def loss(
        self,
        black_point: torch.Tensor,
        white_point: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        return loss

    def get_log_parameters(
        self, target_mid_gray: Optional[float] = None
    ) -> pure_exp_parameters:
        mid_gray_scaling = 1.0
        if target_mid_gray is not None:
            output_mid_gray = self.forward(target_mid_gray)
            mid_gray_scaling = 0.18 / output_mid_gray
        (
            base,
            offset,
            scale,
        ) = self.compute_intermediate_values()
        return pure_exp_parameters(
            base=float(base),
            offset=float(offset),
            scale=float(scale),
            mid_gray_scaling=float(mid_gray_scaling),
        )


parameters_type = Union[
    exp_parameters_simplified,
    legacy_exp_function_parameters,
    gamma_function_parameters,
    pure_exp_parameters,
]
model_type = Union[
    exp_function_simplified, legacy_exp_function, gamma_function, pure_exp_function
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
    "pure_exp": (
        pure_exp_function,
        pure_exp_parameters,
        PURE_EXP_INTIAL_GUESS,
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


def plot_log_curve(model: model_type, mid_gray: Optional[float] = None):
    # Plot the log curve as a sanity check
    x_values = np.linspace(start=-8, stop=8, num=4000)
    x_values_lin = 0.18 * (2**x_values)
    with torch.no_grad():
        y_values = model.reverse(torch.tensor(x_values_lin)).detach().numpy()
        params = model.get_log_parameters(mid_gray)
        cut = 0
        if "cut" in asdict(params):
            cut = asdict(params)["cut"]
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
