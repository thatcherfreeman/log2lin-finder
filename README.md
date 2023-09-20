# Log2Lin-Finder

Improves upon my other repository `zlog-curve` by adding the ability to import a set of exposure bracketed log images. In either case, I now use a simpler log2lin function that better fits the Cut parameter.

## Running instructions
Install requirements using
```
pip install -r requirements.txt
```

## Running
Put a log2linear 1D LUT somewhere, then run:
```
python processor.py --lut_file <path to LUT>
```
This will compute the parameters of the log to linear conversion function in about 10 - 15 minutes, depending on your computer's speed.

Alternatively, if you have a directory of images (the script looks for 16-bit tiff files), you can run with:
```
python processor.py --dir_path <directory path of images>
```

It's worth noting that a set of 1080p images would have 2M pixels each, whereas a 12-bit 1D LUT would have on the order of 4096 points. As a result, the number of epochs you should use would be heavily dependent on the dataset size. When using 8 images, I train for about 100 epochs, whereas with the 1D LUT, you can train for 50k epochs.

I got good results with Canon C-Log3 Images with the following parameters:
```
python .\processor.py --dir_path "<path to images>" --num_epochs 100 --fixed_exposures --blur_amt 7
```
This took about 3 hours on my computer.

## Image preperation
* Follow the instructions in `image_capture_doc/image_capture.pdf`.
* The solver assumes that there are several images that are taken at different exposures, as well as one image that shows the white point and an image taken with the lens cap on.
* The log curve will be fit to the exposure bracket, and the white and black points will be derived from the white point image and the black point image. Put all of these images in a single directory, saved as Tiff or EXR.
* For specifically the exposure bracket images, you should average together many frames (like 10 frames) of each video clip if possible.
* Do not average together frames for the lens cap image.
* If the solver does not correclty identify the white point image, then make a monochrome EXR of just the correct white point and use that instead.

## Verifying your results
As a sanity check, the script will generate a rendering of your images in their log state, with an exposure adjustment being applied in Linear. If the log curve parameters are correct, then all images should look the same (excluding noise and clipped pixels). You can copy the log curve parameters into the Log Curve DCTL and see if you can match your images in Resolve by applying Gain after converting the images to Linear.

# Technical Details
Some camera manufacturers do not release the Log to Scene Linear transfer function for their log curves. However, they sometimes provide it in the form of a 1D LUT. The general form of a log to linear conversion is the following function:

```python
def log2linear(x):
    if (t > cut):
        return scale * pow(base, x) + offset
    else:
        return slope * x + intercept
```

And likewise, the inverse of this function is the linear2log curve:
```python
def linear2log(y):
    if (y > slope * cut + intercept):
        return log((y - offset) / scale) / log(base)
    else:
        return (y - intercept) / slope
```

There's only a matter of choosing the correct values for parameters $\theta = \{\text{scale}, \text{base}, \text{offset}, \text{slope}, \text{intercept}, \text{cut}\}$.

## Unit Tests
* Run linter with `mypy *.py`
* Reformat code with `black .`
