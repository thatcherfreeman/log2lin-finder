import cv2
import os
import numpy as np
import argparse

from lut_parser import lut_1d_properties, read_1d_lut
import torch
import matplotlib.pyplot as plt

import models


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fit_bracketed_exposures(args):
    epochs = args.num_epochs

    # Read in the folder of tiff files.
    dir_path = args.dir_path
    files = [x for x in sorted(os.listdir(dir_path))]
    files = [x for x in files if x.lower().endswith('tif') or x.lower().endswith('tiff') or x.lower().endswith('exr')]
    all_images = []
    print(f"Reading files at directory {dir_path}")
    for fn in files:
        print("file: ", os.path.join(dir_path, fn))
        all_images.append(open_image(os.path.join(dir_path, fn)))
    if args.blur_amt > 0:
        all_images = [cv2.blur(img, (args.blur_amt, args.blur_amt)) for img in all_images]
    print(f"Found {len(all_images)} files.")

    # Identify which one has the most white pixels and the white point.
    all_images = np.stack(all_images, axis=0)
    print(f"Found data type {all_images.dtype}")
    if all_images.dtype not in (float, np.float32):
        print("Converting image datatype to float.")
        format_max = np.iinfo(all_images.dtype).max
        all_images = all_images.astype(np.float32) / format_max

    n, h, w, c = all_images.shape
    gray_images = np.mean(all_images, axis=3)
    white_point = np.max(gray_images)
    flattened_images = gray_images.reshape(n, h*w) # shape (n, h*w)
    white_pixels_per_image = np.count_nonzero(flattened_images >= 0.95 * white_point, axis=1)
    brightest_picture_idx = np.argmax(white_pixels_per_image)
    print(f"Picture with the most clipped pixels (> 0.9 * {white_point}): {files[brightest_picture_idx]}")

    # Remove the bright image from the dataset.
    all_images = np.concatenate([all_images[:brightest_picture_idx], all_images[brightest_picture_idx+1:]])
    files.pop(brightest_picture_idx)
    n, h, w, c = all_images.shape
    all_images = all_images.reshape(n, h*w, c)

    # identify median brightness image.
    image_brightness = np.mean(all_images.reshape(n, h*w*c), axis=1)
    image_brightness_sort_idx = np.argsort(image_brightness)
    median_image_idx = int(0.5 * len(image_brightness_sort_idx))
    all_images = all_images[image_brightness_sort_idx]
    files = np.array(files)[image_brightness_sort_idx]
    image_brightness = image_brightness[image_brightness_sort_idx]

    # Identify exposure compensation to apply to each image
    default_exposure_comp = np.array([-1.0 * i for i in range(n)]) + median_image_idx
    for exp, fn, b in zip(default_exposure_comp, files, image_brightness):
        print(f"{fn} - average brightness: {b} - initial exposure comp: {exp:0.2f}")

    # Run GD
    gains, model = models.derive_exp_function_gd(
        images=all_images,
        ref_image_num=median_image_idx,
        white_point=white_point*0.95,
        epochs=args.num_epochs,
        lr=args.learning_rate,
        use_scheduler=args.lrscheduler,
        exposures=torch.tensor(default_exposure_comp, dtype=float).unsqueeze(1),
        fixed_exposures=args.fixed_exposures,
    )

    print(gains.get_gains())
    found_parameters = model.get_log_parameters()
    print(found_parameters)
    print(found_parameters.exp_curve_to_str())
    with open(os.path.join(dir_path, "parameters.csv"), "w") as f:
        f.write(found_parameters.to_csv())

    # Try a visual comparison of two images.
    input_images = all_images.reshape(n,h,w,c)
    titles = [f"Input file {fn}" for fn in files]
    plot_images(input_images, titles)

    model.eval()
    output_images = []
    titles = []
    with torch.no_grad():
        for i, (image, fn) in enumerate(zip(input_images, files)):
            gain = gains(torch.tensor(i))
            lin_image = model(torch.tensor(image)) * gain
            log_image = model.reverse(lin_image)
            # gamma_image = (32*lin_image)**0.45
            output_images.append(log_image.detach().numpy())
            titles.append(f'file {fn} gain {float(gain)}')
    output_images = np.stack(output_images, axis=0)
    plot_images(output_images, titles)

    # Plot the median image vs the reconstructed exposure compensated images to see the accuracy
    sampled_coords = np.random.choice(input_images.shape[1], 1000), np.random.choice(input_images.shape[2], 1000)
    sampled_input = np.mean(input_images[median_image_idx, sampled_coords[0], sampled_coords[1], :], axis=1)
    for i in range(output_images.shape[0]):
        sampled_output = np.mean(output_images[i, sampled_coords[0], sampled_coords[1], :], axis=1) # (1000, 3)
        plt.scatter(sampled_input, sampled_output, label=f"Image {i}", marker='.', alpha=0.1, edgecolors=None)
    plt.plot([np.min(sampled_input), np.max(sampled_input)], [np.min(sampled_input), np.max(sampled_input)], color='black')
    plt.title("comparison of images")
    plt.legend()
    plt.show()

    # Plot the log curve as a sanity check
    x_values = np.linspace(start=-8, stop=8, num=4000)
    x_values_lin = 0.18 * (2**x_values)
    with torch.no_grad():
        y_values = model.reverse(torch.tensor(x_values_lin)).detach().numpy()
        cut = model.get_log_parameters().cut
    plt.plot(x_values[y_values < cut], y_values[y_values < cut], color='red')
    plt.plot(x_values[y_values >= cut], y_values[y_values >= cut], color='blue')
    plt.axvline(x=0.0)
    plt.axhline(y=0.0)
    plt.title("Log curve")
    plt.show()

    plt.plot(x_values_lin[y_values < cut], y_values[y_values < cut], color='red')
    plt.plot(x_values_lin[y_values >= cut], y_values[y_values >= cut], color='blue')
    plt.axvline(x=0.18, alpha=0.5)
    plt.axvline(x=0.0)
    plt.axhline(y=0.0)
    plt.title("lin/log curve")
    plt.show()



def fit_lut_file(args):
    epochs = args.num_epochs
    fn = args.lut_file
    if args.lut_file == None:
        parser.print_help()

    # Train model
    lut = read_1d_lut(fn)
    model = models.derive_exp_function_gd_lut(
        lut,
        epochs=epochs,
        lr=args.learning_rate,
        use_scheduler=args.lrscheduler,
    )
    print(model.get_log_parameters())

    # Display log2lin model's output curve vs original LUT
    ds = models.dataset_from_1d_lut(lut)
    x, y = ds.tensors

    model.eval()
    y_pred = model(x).detach().numpy()
    model.train()
    y_pred_interp = model(x).detach().numpy()
    x_np = x.numpy()
    y_np = y.numpy()
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x_np, y_np, label='ground truth lut')
    plt.plot(x_np, y_pred, label='model eval mode')
    plt.plot(x_np, y_pred_interp, label='model train mode')
    plt.legend()
    plt.show()

    # Same as above but with log scale
    plt.figure()
    plt.plot(x_np, np.log(y_np), label='ground truth lut')
    plt.plot(x_np, np.log(y_pred), label='model eval mode')
    plt.plot(x_np, np.log(y_pred) - np.log(y_np), label='Log error')
    plt.legend()
    plt.show()

    # Apply lin2log curve to LUT, expect straight line.
    model.eval()
    x_restored = model.reverse(y).detach().numpy()
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x_np, x_np, label='expected')
    plt.plot(x_np, x_restored, label='lin2log(y)')
    plt.legend()
    plt.show()



def plot_images(images, titles):
    n = len(images)
    # images of shape (n, h, w, c)
    num_cols = int(n**0.5) + 1
    num_rows = n // num_cols + 1
    f, axarr = plt.subplots(num_rows, num_cols)
    f.set_size_inches(16,9)
    for i, (image, title) in enumerate(zip(images, titles)):
        r,c = i // num_cols, i % num_cols
        axarr[r, c].imshow(image)
        axarr[r, c].set_title(title, fontsize=5)
        axarr[r, c].set_xticks([])
        axarr[r, c].set_yticks([])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        default=1,
        required=False,
        type=int,
        help='number of epochs to train for.',
    )
    parser.add_argument(
        '--dir_path',
        default=None,
        help='Specify the directory to load the images from.',
    )
    parser.add_argument(
        '--lut_file',
        default=None,
        help='Specify the 1D file to load from.',
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='Specify the gradient descent learning rate.',
        required=False,
    )
    parser.add_argument(
        '--lrscheduler',
        action='store_true',
        help='Add flag to avoid learning rate scheduler. Do this if the step size goes to zero before convergeance.',
        required=False,
    )
    parser.add_argument(
        '--fixed_exposures',
        action='store_true',
        help='Add this flag to avoid fine-tuning the one-stop increments between the estimated exposure differences.',
        required=False,
    )
    parser.add_argument(
        '--blur_amt',
        default=0,
        type=int,
        help='specifies amount to blur input images, to reduce noise.',
        required=False,
    )
    args = parser.parse_args()
    print(args)

    if args.dir_path is not None and args.lut_file is None:
        fit_bracketed_exposures(args)
    elif args.lut_file is not None and args.dir_path is None:
        fit_lut_file(args)
    else:
        print("Please specify one of --dir_path or --lut_file!")

