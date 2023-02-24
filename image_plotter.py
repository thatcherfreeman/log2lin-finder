import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt



def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'x_file',
        type=str,
        help='Image to read on the x-axis.'
    )
    parser.add_argument(
        'y_file',
        type=str,
        help='Image displayed on the y-axis'
    )
    parser.add_argument(
        '--xlog',
        action='store_true',
        help='plot x-axis as log plot',
    )
    parser.add_argument(
        '--ylog',
        action='store_true',
        help='plot y-axis as log plot',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='number of points to plot.'
    )
    parser.add_argument(
        '--swap',
        action='store_true',
        help='swap x and y axes.'
    )
    args = parser.parse_args()
    print(args)
    x_img = open_image(args.x_file)
    y_img = open_image(args.y_file)
    if args.swap:
        x_img, y_img = y_img, x_img
    print("X image shape: ", x_img.shape)
    print("Y image shape: ", y_img.shape)
    assert x_img.shape == y_img.shape
    assert len(x_img.shape) == 3
    h, w, c = x_img.shape

    if args.num_samples <= 0:
        num_samples = h * w
    else:
        num_samples = args.num_samples

    sampled_coords = np.random.choice(h, num_samples), np.random.choice(w, num_samples)

    x_samples = x_img[sampled_coords[0], sampled_coords[1]] # num_samples, c
    y_samples = y_img[sampled_coords[0], sampled_coords[1]]

    colors = ['red', 'green', 'blue']
    for i in range(c):
        if i < len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3,)
        plt.scatter(x_samples[:, i], y_samples[:, i], marker='.', alpha=0.1, color=color)
    max_val = np.max([x_img, y_img])
    min_val = np.min([x_img, y_img])
    samples = np.linspace(min_val, max_val, 50)
    plt.plot(samples, samples)

    if args.xlog:
        plt.xscale('log')
    if args.ylog:
        plt.yscale('log')
    plt.show()
    print("MSE: ", np.mean((x_img - y_img)**2))



if __name__ == "__main__":
    main()