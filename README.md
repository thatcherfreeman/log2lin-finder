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

## Tips for taking images
1. Lock your camera off on a tripod, pointing at a scene with a variety of tones in it (IE not just a white wall or a test chart, shoot your bookshelf if you can't think of anything).
2. Make sure your camera is set to be entirely manual settings, fix the Aperture, ISO, and white balance, do not use any ND filters. All exposure adjustments from here on should be done only with shutter speed.
3. One image needs to have many clipped pixels so the software can identify the white point. This image will be removed from the dataset so it doesn't necessarily have to be the same scene as the rest of the images.
4. Without moving your camera, shorten the shutter speed in one-stop increments until you can't shorten it anymore. Hopefully you have around 8 to 12 images.
5. Make 16-bit tiffs or EXRs of each exposure and throw them in a folder somewhere, I'd recommend resizing them to 1080p.
6. Run the script with `python processor.py --dir_path <directory of images>`, this should take around a minute.
7. If the script is working, run it again with `--num_epochs X` with `X` in the range of 20-100 to get a more accurate result.

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