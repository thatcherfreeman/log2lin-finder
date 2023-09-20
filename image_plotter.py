import numpy as np
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # type:ignore
import argparse
import matplotlib.pyplot as plt  # type:ignore
from src.images import open_image, convert_to_hsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x_file", type=str, help="Image to read on the x-axis.")
    parser.add_argument("y_file", type=str, help="Image displayed on the y-axis")
    parser.add_argument(
        "--xlog",
        action="store_true",
        help="plot x-axis as log plot",
    )
    parser.add_argument(
        "--ylog",
        action="store_true",
        help="plot y-axis as log plot",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="number of points to plot."
    )
    parser.add_argument("--swap", action="store_true", help="swap x and y axes.")
    parser.add_argument(
        "--mode",
        choices=["huesat", "codevalue", "satsat"],
        default="codevalue",
        help="Choose the kind of plot.",
    )
    parser.add_argument(
        "--clipping_point",
        type=float,
        default=None,
        help="Pixel values above this number will be ignored.",
    )
    parser.add_argument(
        "--linear_regression",
        action="store_true",
        help="set this flag to run a linear regression on the samples.",
    )

    args = parser.parse_args()
    print(args)
    x_img = open_image(args.x_file)
    y_img = open_image(args.y_file)
    if args.swap:
        x_img, y_img = y_img, x_img
    print("X image shape: ", x_img.shape)
    print("Y image shape: ", y_img.shape)
    if x_img.shape != y_img.shape:
        print(f"Shapes mismatched, resizing Y image to {x_img.shape}")
        y_img = cv2.resize(
            y_img, (x_img.shape[1], x_img.shape[0]), interpolation=cv2.INTER_AREA
        )
        print(y_img.shape)

    assert x_img.shape == y_img.shape
    assert len(x_img.shape) == 3
    h, w, c = x_img.shape

    if args.num_samples <= 0:
        num_samples = h * w
    else:
        num_samples = args.num_samples

    sampled_coords = np.random.choice(h, num_samples), np.random.choice(w, num_samples)

    if args.mode in ("huesat", "satsat"):
        x_img = convert_to_hsv(x_img)
        y_img = convert_to_hsv(y_img)

    x_samples = x_img[sampled_coords[0], sampled_coords[1]]  # num_samples, c
    y_samples = y_img[sampled_coords[0], sampled_coords[1]]

    sample_mask = np.ones(num_samples, dtype=bool)
    if args.clipping_point is not None:
        sample_mask = sample_mask & (np.max(x_samples, axis=1) < args.clipping_point)
        sample_mask = sample_mask & (np.max(y_samples, axis=1) < args.clipping_point)
        x_samples = x_samples[sample_mask]
        y_samples = y_samples[sample_mask]

    if args.mode == "codevalue":
        colors = ["red", "green", "blue"]
        for i in range(c):
            if i < len(colors):
                color = colors[i]
            else:
                color = np.random.rand(3)
            plt.scatter(
                x_samples[:, i], y_samples[:, i], marker=".", alpha=0.1, color=color
            )
        max_val = np.max([x_img, y_img])
        if args.clipping_point is not None:
            max_val = np.min([max_val, args.clipping_point])
        min_val = np.min([x_img, y_img])
        samples = np.linspace(min_val, max_val, 50)
        plt.plot(samples, samples)
        if args.linear_regression:
            params, _, _, _ = np.linalg.lstsq(
                np.stack(
                    [x_samples.flatten(), np.ones_like(x_samples.flatten())], axis=1
                ),
                y_samples.flatten(),
                rcond=None,
            )
            print(f"Linear Regression: {params[0]}x + {params[1]}")
            plt.plot(samples, (params[0] * samples + params[1]))

    elif args.mode == "huesat":
        fig, ax1 = plt.subplots()
        sample_mask = x_samples[:, 1] > 0.1
        x_samples = x_samples[sample_mask]
        y_samples = y_samples[sample_mask]

        plt.scatter(
            x_samples[:, 0] * 360.0,
            (y_samples[:, 0] - x_samples[:, 0]),
            color="red",
            marker=".",
            alpha=0.1,
            label="Hue",
        )
        plt.scatter(
            x_samples[:, 0] * 360.0,
            y_samples[:, 1] / x_samples[:, 1],
            color="green",
            marker=".",
            alpha=0.1,
            label="Saturation",
        )
        samples = np.linspace(0.0, 1.0, 50)
        plt.hlines(1.0, 0.0, 360.0)
        plt.hlines(0.0, 0.0, 360.0)
        plt.legend()

    elif args.mode == "satsat":
        x_samples = x_samples[:, 1]
        y_samples = y_samples[:, 1]
        plt.scatter(
            x_samples,
            y_samples,
            color="green",
            marker=".",
            alpha=0.1,
            label="Saturation",
        )
        max_val = np.max([x_img[:, 1], y_img[:, 1]])
        min_val = np.min([x_img[:, 1], y_img[:, 1]])
        min_val, max_val = 0.0, 1.0
        samples = np.linspace(min_val, max_val, 50)
        plt.plot(samples, samples)
        if args.linear_regression:
            params, _, _, _ = np.linalg.lstsq(
                np.expand_dims(x_samples.flatten(), axis=1),
                y_samples.flatten(),
                rcond=None,
            )
            print(f"Linear Regression: {params[0]}x")
            plt.plot(samples, (params[0] * samples))

    if args.xlog:
        plt.xscale("log")
    if args.ylog:
        plt.yscale("log")
    plt.show()

    print("MSE: ", np.mean((x_img - y_img) ** 2))


if __name__ == "__main__":
    main()
