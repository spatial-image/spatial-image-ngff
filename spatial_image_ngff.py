"""spatial-image-ngff

Serialize and deserialize a multiscale spatial image to the
Open Microscopy Environment (OME) Zarr Next Generation File Format (NGFF)."""

__version__ = "0.1.1"

from typing import MutableMapping, Union
from pathlib import Path

import xarray as xr
import numpy as np
import zarr

from spatial_image_multiscale import MultiscaleSpatialImage

def imwrite(
    multiscale: MultiscaleSpatialImage,
    store: Union[MutableMapping, str, Path]
) -> None:
    """Write a multiscale spatial image to the store or path in the NGFF format.

    Parameters
    ----------

    multiscale : MultiscaleSpatialImage
        The multiscale spatial image to serialize.

    store : MutableMapping or str or Path
        Zarr store or path to directory in the filesystem.
    """

    for scale, image in enumerate(multiscale):
        name = image.name
        image_ds = image.to_dataset(name=name)

        image_ds.to_zarr(store,
                         mode='w',
                         group=f'{scale}',
                         compute=True)

    datasets = [ { 'path': f'{scale}/{name}' } for scale in range(len(multiscale)) ]
    with zarr.open(store) as z:
        z.attrs['multiscales'] = [{ 'version': '0.1', 'name': name, 'datasets': datasets }]

    zarr.consolidate_metadata(store)
