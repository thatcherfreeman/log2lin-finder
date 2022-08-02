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

It's worth noting that a set of 1080p images would have 2M pixels each, whereas a 12-bit 1D LUT would have on the order of 4096 points. As a result, the number of epochs you should use would be heavily dependent on the dataset size. When using 8 images, I train for about 10 epochs, whereas with the 1D LUT, you can train for 50k epochs.

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

There's only a matter of choosing the correct values for parameters $\theta = \{scale, base, offset, slope, intercept}$.