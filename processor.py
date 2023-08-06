import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
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

    # Convert images to float, if not float already.
    all_images = np.stack(all_images, axis=0)
    print(f"Found data type {all_images.dtype}")
    if all_images.dtype not in (float, np.float32):
        print("Converting image datatype to float.")
        format_max = np.iinfo(all_images.dtype).max
        all_images = all_images.astype(np.float32) / format_max

    # Identify which one has the most white pixels and the white point.
    n, h, w, c = all_images.shape
    gray_images = np.mean(all_images, axis=3)
    white_point = np.max(np.nanquantile(gray_images, 0.99, axis=1))
    flattened_images = gray_images.reshape(n, h*w) # shape (n, h*w)
    white_pixels_per_image = np.count_nonzero(flattened_images >= 0.95 * white_point, axis=1)
    brightest_picture_idx = np.argmax(white_pixels_per_image)
    print(f"Picture with the most clipped pixels (> 0.95 * {white_point}): {files[brightest_picture_idx]}")

    # Identify blackest image.
    darkest_picture_idx = np.argmin(np.mean(flattened_images, axis=1))
    darkest_value = np.mean(all_images[darkest_picture_idx])
    print(f"Picture with the black pixels (black point: {darkest_value}): {files[darkest_picture_idx]}")

    # Remove the bright and dark image from the dataset.
    all_images = np.delete(all_images, [brightest_picture_idx, darkest_picture_idx], axis=0)
    files = np.delete(files, [brightest_picture_idx, darkest_picture_idx], axis=0)
    n, h, w, c = all_images.shape
    all_images = all_images.reshape(n, h*w, c)

    # identify median brightness image.
    image_brightness = np.mean(all_images.reshape(n, h*w*c), axis=1)
    image_brightness_sort_idx = np.argsort(image_brightness)
    median_image_idx = int(0.5 * len(image_brightness_sort_idx))
    all_images = all_images[image_brightness_sort_idx]
    files = files[image_brightness_sort_idx]
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
        black_point=darkest_value,
        epochs=args.num_epochs,
        lr=args.learning_rate,
        use_scheduler=args.lrscheduler,
        exposures=torch.tensor(default_exposure_comp, dtype=torch.float32).unsqueeze(1),
        fixed_exposures=args.fixed_exposures,
        initial_parameters_fn=args.initial_parameters,
        batch_size=args.batch_size,
        mid_gray=args.mid_gray,
    )

    print(gains.get_gains(median_image_idx))
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
    output_lin_images = []
    titles = []
    with torch.no_grad():
        for i, (image, fn) in enumerate(zip(input_images, files)):
            gain = gains(torch.tensor(i), median_image_idx)
            lin_image = model(torch.tensor(image)) * gain
            log_image = model.reverse(lin_image)
            # gamma_image = (32*lin_image)**0.45
            output_lin_images.append(lin_image.detach().numpy())
            output_images.append(log_image.detach().numpy())
            titles.append(f'file {fn} gain {float(gain)}')
    output_images = np.stack(output_images, axis=0)
    output_lin_images = np.stack(output_lin_images, axis=0)
    plot_images(output_images, titles)

    # Plot the median image vs the reconstructed exposure compensated images to see the accuracy
    sampled_coords = np.random.choice(input_images.shape[1], 1000), np.random.choice(input_images.shape[2], 1000)
    sampled_input = np.mean(input_images[median_image_idx, sampled_coords[0], sampled_coords[1], :], axis=1)
    for i in range(output_images.shape[0]):
        sampled_output = np.mean(output_images[i, sampled_coords[0], sampled_coords[1], :], axis=1) # (1000, 3)
        plt.scatter(sampled_input, sampled_output, label=f"Image {i}", marker='.', alpha=0.1, edgecolors=None)
    plt.plot([np.min(sampled_input), np.max(sampled_input)], [np.min(sampled_input), np.max(sampled_input)], color='black')
    plt.title("comparison of log images")
    plt.legend()
    plt.show()

    models.plot_log_curve(model)


def fit_two_images(args):
    log_image = open_image(args.log_image)
    lin_image = open_image(args.lin_image)

    assert log_image.shape == lin_image.shape and len(log_image.shape) == 3
    h, w, c = log_image.shape
    flattened_log_image = np.reshape(log_image, (h*w, c))
    flattened_lin_image = np.reshape(lin_image, (h*w, c))

    model = models.derive_exp_function_gd_log_lin_images(
        log_image=flattened_log_image,
        lin_image=flattened_lin_image,
        black_point=np.min(log_image),
        epochs=args.num_epochs,
        lr=args.learning_rate,
        use_scheduler=args.lrscheduler,
        initial_parameters_fn=args.initial_parameters,
        batch_size=args.batch_size,
    )
    found_parameters = model.get_log_parameters()
    print(found_parameters)
    print(found_parameters.exp_curve_to_str())
    with open(os.path.join(os.path.dirname(args.log_image), "parameters.csv"), "w") as f:
        f.write(found_parameters.to_csv())
    models.plot_log_curve(model)

    with torch.no_grad():
        reconstructed_lin_image = model(torch.tensor(log_image)).detach().numpy()
        flattened_reconstructed_lin_image = np.reshape(reconstructed_lin_image, (h*w, c))
        reconstructed_log_image = model.reverse(torch.tensor(lin_image)).detach().numpy()
        flattened_reconstructed_log_image = np.reshape(reconstructed_log_image, (h*w, c))

    num_samples = 10000
    sampled_coords = np.random.choice(h*w, num_samples)
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.scatter(flattened_log_image[sampled_coords, i], flattened_reconstructed_log_image[sampled_coords, i], alpha=0.05, marker='.', color=color, edgecolors=None)
    min_val, max_val = np.min(flattened_log_image), np.max(flattened_log_image)
    samples = np.linspace(min_val, max_val, 50)
    plt.plot(samples, samples)
    plt.title("Comparison of log images")
    plt.show()

    for i, color in enumerate(['red', 'green', 'blue']):
        plt.scatter(flattened_lin_image[sampled_coords, i], flattened_reconstructed_lin_image[sampled_coords, i], alpha=0.05, marker='.', color=color, edgecolors=None)
    min_val, max_val = np.min(flattened_lin_image), np.max(flattened_lin_image)
    samples = np.linspace(min_val, max_val, 50)
    plt.plot(samples, samples)
    plt.title("Comparison of lin images")
    plt.xscale('log')
    plt.yscale('log')
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
        initial_parameters_fn=args.initial_parameters,
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
        type=str,
        help='Specify the directory to load the images from.',
    )
    parser.add_argument(
        '--lut_file',
        default=None,
        type=str,
        help='Specify the 1D file to load from.',
    )
    parser.add_argument(
        '--log_image',
        default=None,
        type=str,
        help='Use this alongside lin_image to indicate a target log and lin image of the same scene (same primaries)'
    )
    parser.add_argument(
        '--lin_image',
        default=None,
        type=str,
        help='Use this alongside lin_image to indicate a target log and lin image of the same scene (same primaries)'
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
    parser.add_argument(
        '--initial_parameters',
        type=str,
        default=None,
        help='Initialize the parameters with this csv.',
        required=False,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help='size of each batch',
        required=False,
    )
    parser.add_argument(
        '--mid_gray',
        type=float,
        default=None,
        help='0.0-1.0 code value corresponding to 0.18 mid gray. Ignored unless you specify --dir_path.',
        required=False,
    )
    args = parser.parse_args()
    print(args)

    if args.log_image is not None and args.lin_image is not None:
        fit_two_images(args)
    elif args.dir_path is not None and args.lut_file is None:
        fit_bracketed_exposures(args)
    elif args.lut_file is not None and args.dir_path is None:
        fit_lut_file(args)
    else:
        print("Please specify one of --dir_path or --lut_file!")

