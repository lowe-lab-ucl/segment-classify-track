import os
import sys
import enum
import re

import dask
import dask.array as da

import numpy as np

from skimage import io
from typing import Union, Optional
from scipy.ndimage import median_filter

OCTOPUSLITE_FILEPATTERN = "img_channel([0-9]+)_position([0-9]+)_time([0-9]+)_z([0-9]+)"


@enum.unique
class Channels(enum.Enum):
    BRIGHTFIELD = 0
    GFP = 1
    RFP = 2
    IRFP = 3
    PHASE = 4
    WEIGHTS = 98
    MASK = 99


def remove_outliers(x):
    med_x = median_filter(x, size=2)
    mask = x > med_x
    x = x * (1-mask) + (mask*med_x)
    return x


def estimate_background(x: np.ndarray) -> np.ndarray:
    """Estimate background using a second order polynomial surface.

    Estimate the background of an image using a second-order polynomial surface
    assuming sparse signal in the image.  Essentially a massive least-squares
    fit of the image to the polynomial.

    Parameters
    ----------
    x : np.ndarray
        An input image which is to be used for estimating the background.

    Returns
    -------
    background_estimate : np.ndarray
    	A second order polynomial surface representing the estimated background
        of the image.
    """

    # set up arrays for params and the output surface
    A = np.zeros((x.shape[0]*x.shape[1], 6))
    background_estimate = np.zeros((x.shape[1], x.shape[0]))

    u, v = np.meshgrid(
        np.arange(x.shape[1], dtype=np.float32),
        np.arange(x.shape[0], dtype=np.float32),
    )
    A[:, 0] = 1.
    A[:, 1] = np.reshape(u, (x.shape[0] * x.shape[1],))
    A[:, 2] = np.reshape(v, (x.shape[0] * x.shape[1],))
    A[:, 3] = A[:, 1] * A[:, 1]
    A[:, 4] = A[:, 1] * A[:, 2]
    A[:, 5] = A[:, 2] * A[:, 2]

    # convert to a matrix
    A = np.matrix(A)

    # calculate the parameters
    k = np.linalg.inv(A.T * A) * A.T
    k = np.squeeze(np.array(np.dot(k, np.ravel(x))))

    # calculate the surface
    background_estimate = k[0] + k[1]*u + k[2]*v + k[3]*u*u + k[4]*u*v + k[5]*v*v
    return background_estimate


class DaskOctopusLiteLoader:
    """ DaskOctopusLiteLoader

    A simple class to load OctopusLite data from a directory.
    Caches data once it is loaded to prevent excessive io to
    the data server.

    Can directly address fluorescence channels using the
    `Channels` enumerator:

        Channels.BRIGHTFIELD
        Channels.GFP
        Channels.RFP
        Channels.IRFP

    Usage:
        octopus = SimpleOctopusLiteLoader('/path/to/your/data/*.tif')
        gfp = octopus[Channels.GFP]

    Parameters
    ----------
    path : str
        The path to the dataset.
    crop : tuple, optional
        An optional tuple which can be used to perform a centred crop on the data.
    remove_background : bool
        Use a estimated polynomial surface to remove uneven illumination.


    Methods
    -------
    __getitem__ : Channels, str
        Return a dask lazy array of the image data for the channel. If cropping
        has been specified, the images are also cropped to this size.


    Properties
    ----------
    shape :
        Returns the shape of the uncropped data.
    channels :
        Return the channels found in the dataset.

    """
    def __init__(
        self, path: str,
        crop: Optional[tuple] = None,
        remove_background: bool = True,
    ):
        self.path = path
        self._files = {}
        self._lazy_arrays = {}
        self._crop = crop
        self._shape = ()

        print(f'Using cropping: {crop}')

        # parse the files
        self._parse_files()

    def __contains__(self, channel):
        return channel in self.channels

    @property
    def channels(self):
        return list(self._files.keys())

    @property
    def shape(self):
        return self._shape

    def channel_name_from_index(self, channel_index: int):
        return Channels(int(channel_index))

    def __getitem__(self, channel_name: Union[str, Channels]):

        if isinstance(channel_name, str):
            channel_name = Channels[channel_name.upper()]

        if channel_name not in self.channels:
            raise ValueError(f"Channel {channel_name} not found in {self.path}")

        return self._lazy_arrays[channel_name]


    def files(self, channel_name: str) -> list:
        return self._files[Channels[channel_name.upper()]]

    def _load_and_crop(self, fn: str) -> np.ndarray:
        """Load and crop the image."""
        image = io.imread(fn)

        if self._crop is None:
            return image

        assert isinstance(self._crop, tuple)

        dims = image.ndim
        shape = image.shape
        crop = np.array(self._crop).astype(np.int64)

        # check that we don't exceed any dimensions
        assert all([crop[i] <= s for i, s in enumerate(shape)])

        # automagically build the slices for the array
        cslice = lambda d: slice(
            int((shape[d] - crop[d]) // 2),
            int((shape[d] - crop[d]) // 2 + crop[d])
        )
        crops = tuple([cslice(d) for d in range(dims)])

        cleaned = remove_outliers(image[crops])
        bg = estimate_background(cleaned.astype(np.float32))

        return cleaned.astype(np.float32) - bg.astype(np.float32)

    def _parse_files(self):
        """Parse out the files from the folder and create lazy arrays."""

        # find the files in the dataset
        files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith('.tif')
        ]

        if not files:
            raise FileNotFoundError(f"No files found in directory: {self.path}")

        # take a sample of the dataset
        sample = io.imread(files[0])
        self._shape = sample.shape if self._crop is None else self._crop

        def parse_filename(fn):
            pth, fn = os.path.split(fn)
            params = re.match(OCTOPUSLITE_FILEPATTERN, fn)
            return self.channel_name_from_index(params.group(1)), params.group(3)

        channels = {k:[] for k in Channels}

        # parse the files
        for f in files:
            channel, time = parse_filename(f)
            channels[channel].append(f)

        # sort them by time
        for channel in channels.keys():
            channels[channel].sort(key=lambda f: parse_filename(f)[1])

        # remove any channels that are empty
        self._files = {k: v for k, v in channels.items() if v}

        # now set up the lazy loaders
        for channel, files in self._files.items():
            self._lazy_arrays[channel] = [
                da.from_delayed(
                    dask.delayed(self._load_and_crop)(fn),
                    shape=self._shape,
                    dtype=np.float32, # sample.dtype
                ) for fn in files
            ]

            # concatenate them along the time axis
            self._lazy_arrays[channel] = da.stack(self._lazy_arrays[channel], axis=0)
