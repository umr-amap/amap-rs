import re
import os
import sys
import glob
import functools
import rasterio
import torch
import numpy as np
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.io import DatasetReader
from torch import Tensor
from rasterio.windows import from_bounds
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast, Union
from torchgeo.datasets.utils import disambiguate_timestamp, BoundingBox
from torchgeo.datasets import RasterDataset, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple
from torch.utils.data import Dataset
import geopandas as gpd


def intersects_with_img(roi, file_list):
    res = False
    for file in file_list:
        with rasterio.open(file) as ds :
            tf = ds.meta.copy()['transform']
            bounds = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
            if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
                res = True
                break      
    return res

def get_intersected_bboxes(root, filename_glob, gdf):
    pathname = os.path.join(root, "**", filename_glob)
    file_list = []
    for filepath in glob.iglob(pathname, recursive=True):
        file_list.append(filepath)
    return gdf.loc[[intersects_with_img(gdf['bboxes'][i], file_list) for i in gdf.index]]


def polygon_to_bbox(polygon, dataset):
    bounds = list(polygon.bounds)
    bounds[1], bounds[2] = bounds[2], bounds[1]
    # keeping temporal coordinates from raster dataset
    return BoundingBox(*bounds, dataset.index.bounds[4] , dataset.index.bounds[5])


def prepare_datasets(datasets, gdf, target_variable, buffer_size=100):
    """
    Ensures that all points in the geodataframe has a buffer and intersects with raster images.

    Args:
        datasets: RasterDataset or list of RasterDatasets.
        gdf: GeoDataFrame.
        target_variable: column name of the target variable or list of column names.
        buffer_size: size (in meters) of the buffer created around the point if geometry is a point.

    Returns:
        intersection dataset of the dataset and filtered geodataframe.
    """

    ## temporally convert to list
    if not isinstance(datasets, list):
        datasets = [datasets]

    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.to_crs(datasets[0].crs)
    gdf = gdf.drop_duplicates()

    # gdf = gdf.dropna(subset=[target_variable])
    if isinstance(target_variable, list):
        for variable in target_variable:
            gdf = gdf.dropna(subset=[variable])
    else :
        gdf = gdf.dropna(subset=[target_variable])

    # if geodataframe has points, convert to square with buffer of self.size meters
    if gdf.geom_type.unique() == "Point":
        gdf.geometry = gdf.buffer(buffer_size, cap_style = 3)

    gdf['bboxes'] = [polygon_to_bbox(gdf['geometry'][i], datasets[0]) for i in gdf.index]

    for i, dataset in enumerate(datasets):
        if i == 0:
            current_dataset = dataset
        else:
            current_dataset = current_dataset & dataset

        gdf = get_intersected_bboxes(dataset.paths, dataset.filename_glob, gdf)

    return current_dataset, gdf


class ROIDataset(RasterDataset):

    def __init__(
            self, 
            raster_dataset: RasterDataset,
            gdf: gpd.GeoDataFrame, 
            target_var, 
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            bboxes_col:str = 'bboxes',
            ):

        self.dataset = raster_dataset
        self.target_var = target_var
        self.gdf = gdf
        self.transforms = transforms
        self.bboxes_col = bboxes_col

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        """Retrieve image/mask and metadata indexed by query.

        Args:
            index: Index of sample to fetch

        Returns:
            sample of image/mask, metadata and ground truth at that index

        Raises:
            IndexError: if query is not found in the index
        """
        query = self.gdf.iloc[idx][self.bboxes_col]
        # gt = self.gdf.iloc[idx][self.target_var]

        # sample = super().__getitem__(query)
        sample = self.dataset[query]

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        if isinstance(self.target_var, list):
            for gt in self.target_var:
                sample[gt] = self.gdf.iloc[idx][gt]
        else :
            sample[self.target_var] = self.gdf.iloc[idx][self.target_var]

        return sample




class CustomLabelDataset(Dataset):
    def __init__(self, gdf, lab_column, transform=None):
        self.labels = gdf[lab_column]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        if self.transform:
            label = self.transform(label)
        return label
    
# class CustomDataset(RasterDataset):
#     filename_glob = "*.tif"
#     is_image = True
