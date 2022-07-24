import cv2
import os
import numpy as np
import argparse

import models
import torch
import matplotlib.pyplot as plt



def read_img(fn):
    image = cv2.imread(fn, -1)
    # cv2 reads in images as H,W,BGR rather than RGB, need to reorder the color channels
    image = image[:, :, ::-1]
    return image

def plot_images(images, titles):
    n = len(images)
    # images of shape (n, h, w, c)
    num_cols = int(n**0.5) + 1
    num_rows = n // num_cols + 1
    f, axarr = plt.subplots(num_rows, num_cols)
    f.set_size_inches(16,10)
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
        '--learning_rate',
        default=1e-4,
        type=float,
        help='Specify the gradient descent learning rate.',
        required=False,
    )
    parser.add_argument(
        '--no_lrscheduler',
        action='store_false',
        help='Add flag to avoid learning rate scheduler. Do this if the step size goes to zero before convergeance.',
        required=False,
    )
    args = parser.parse_args()
    print(args)
    epochs = args.num_epochs

    # Read in the folder of tiff files.
    dir_path = args.dir_path
    files = [x for x in sorted(os.listdir(dir_path))]
    files = [x for x in files if x.lower().endswith('tif') or x.lower().endswith('tiff')]
    all_images = []
    print(f"Reading files at directory {dir_path}")
    for fn in files:
        print("file: ", os.path.join(dir_path, fn))
        all_images.append(read_img(os.path.join(dir_path, fn)))
    print(f"Found {len(all_images)} files.")

    # Identify which one has the most white pixels and the white point.
    all_images = np.stack(all_images, axis=0)
    print(f"Found data type {all_images.dtype}")
    if all_images.dtype != float:
        print("Converting image datatype to float.")
        format_max = np.iinfo(all_images.dtype).max
        all_images = all_images.astype(float) / float(format_max)

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
    image_brightness = np.mean(flattened_images, axis=1)
    median_image_idx = int(np.argwhere(image_brightness == np.percentile(image_brightness, 50, interpolation='nearest')))
    print(f"Median exposure image is {median_image_idx} with filename {files[median_image_idx]}")

    # Run GD
    gains, model = models.derive_exp_function_gd(
        images=all_images,
        ref_image_num=median_image_idx,
        white_point=white_point*0.95,
        epochs=args.num_epochs,
        lr=args.learning_rate,
        use_scheduler=not args.no_lrscheduler,
        exposures=torch.tensor([-4., -3., -2., -1., 0., 1., 2., 3.,]).unsqueeze(1),
    )

    print(gains.get_gains())
    found_parameters = model.get_log_parameters()
    print(found_parameters)
    # print(found_parameters.exp_curve_to_str())


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
    plot_images(np.stack(output_images, axis=0), titles)