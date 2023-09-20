import torch


def percent_error_masked(y, sample_mask):
    # y is of shape (batch_size, n_images, 3), need to compare the pixel values for all images in this set.
    num_valid_entries = torch.sum(
        sample_mask, axis=1, keepdim=True
    )  # now shape (batch_size, 1, 3)
    y_target = y
    y_target[~sample_mask] = 0.0

    # Compute mean among not-clipped pixels for this x,y coordinate.
    target_pixel_value = (
        torch.sum(y_target, axis=1, keepdim=True) / num_valid_entries
    )  # shape (batch_size, 1, 3)
    delta = torch.abs(y - target_pixel_value) / target_pixel_value
    return torch.mean(delta[sample_mask])


def err_fn(y, sample_mask):
    # y is of shape (batch_size, n_images, 3), need to compare the pixel values for all images in this set.
    num_valid_entries = torch.sum(
        sample_mask, axis=1, keepdim=True
    )  # now shape (batch_size, 1, 3)
    y_target = y
    y_target[~sample_mask] = 0.0

    # Compute mean among not-clipped pixels for this x,y coordinate.
    target_pixel_value = (
        torch.sum(y_target, axis=1, keepdim=True) / num_valid_entries
    )  # shape (batch_size, 1, 3)
    delta = torch.abs(torch.log(y_target + 1) - torch.log(target_pixel_value + 1))
    return torch.mean(delta[sample_mask])


def percent_error_target(y, target_idx, sample_mask):
    y_target = y[:, [target_idx], :]  # shape (batch_size, 1, 3)
    answer_mask = torch.ones_like(y, dtype=bool)
    answer_mask[:, [target_idx], :] = 0
    sample_mask = sample_mask & answer_mask
    delta = torch.abs(y - y_target) / y_target
    return torch.mean(delta[sample_mask])


def error_fn_2(y, target_idx, sample_mask):
    y = torch.log(torch.maximum(y, torch.tensor(1e-8)))
    y_target = y[:, [target_idx], :]  # shape (batch_size, 1, 3)

    answer_mask = torch.ones_like(y, dtype=bool)
    answer_mask[:, [target_idx], :] = 0
    sample_mask = sample_mask & answer_mask
    delta = torch.abs(y - y_target)
    return torch.mean(delta[sample_mask])


def reconstruction_error(y, y_original, sample_mask, device):
    # Assume y of shape (batch_size, n_images, n_images, 3)
    # assume y_original of shape (batch_size, 1, n_images, 3)
    # assume target_idx points to the one with gain 1.0
    # Assume y and y_original are both the log encoded images, just take L2 or L1 error.
    sample_mask = sample_mask & ~torch.isnan(y)
    # delta = torch.abs(y - y_original)
    delta = (y - y_original) ** torch.tensor(2.0, device=device)
    return torch.mean(delta[sample_mask])


def log_lin_image_error(lin_gt, lin_pred):
    return torch.mean(torch.abs(lin_pred - lin_gt))


def negative_linear_values_penalty(y_linear):
    sample_mask = ~torch.isnan(y_linear)
    negative = torch.minimum(
        y_linear, torch.zeros_like(y_linear)
    )  # zero out positive values.
    loss = negative ** torch.tensor(2.0)
    return torch.sum(loss[sample_mask]) / torch.numel(y_linear)


def negative_black_point_penalty(black_lin):
    loss = (
        torch.minimum(black_lin, torch.zeros_like(black_lin)) ** 2
    )  # zero if black_lin > 0
    return loss


def middle_gray_penalty(lin_img):
    # Given "properly exposed" linear image, penalize if it doesn't average at middle gray
    pos_mask = (lin_img > 1e-6) & (~torch.isnan(torch.log2(lin_img)))
    return (
        torch.mean(torch.log2(lin_img)[pos_mask]) - torch.log2(torch.tensor(0.18))
    ) ** 2
