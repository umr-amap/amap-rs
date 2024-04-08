import torch
import os
import glob
import numpy as np
from torchgeo.datasets import BoundingBox, GeoDataset
import geopandas as gpd
from typing import Iterable
from typing import Any, Dict, List
from sklearn.cluster import KMeans
import rasterio
from rasterio.crs import CRS
import collections
from heapq import nsmallest
from shapely.geometry import Polygon
from shapely.geometry import Point
import warnings


def intersects_with_img(
        roi:BoundingBox, 
        file_list:List
        ):
    """
    Check if an ROI intersects with at least one image in the list provided


    Args:
        roi (BoundingBox): Torchgeo BoundingBox.
        file_list (List): List of full path to rasters.

    Returns:
        if the roi intersects with at least one of the images (Bool).

    """
    res = False
    for file in file_list:
        with rasterio.open(file) as ds :
            tf = ds.meta.copy()['transform']
            bounds = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
            if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
                res = True
                break      
    return res

def intersects_with_torchgeo_dataset(
        roi:BoundingBox, 
        dataset:GeoDataset
        ):
    """
    Check if an ROI intersects with a torchgeo GeoDataset


    Args:
        roi (BoundingBox): Torchgeo BoundingBox.
        dataset (List): A Torchgeo GeoDataset.

    Returns:
        if the roi intersects with the dataset (Bool).

    """
    res = False

    bounds = dataset.index.bounds
    print(bounds)

    if (roi.minx>bounds[0]) & (roi.miny>bounds[2]) & (roi.maxx<bounds[1]) & (roi.maxy<bounds[3]):
        res = True

    return res


#### following functions likely obsolete with ROIDataset

def get_intersected_bboxes(
        gdf, 
        img_dir, 
        filename_glob, 
        geom_col_name = 'bboxes'
        ):
    pathname = os.path.join(img_dir, "**", filename_glob)
    file_list = []
    for filepath in glob.iglob(pathname, recursive=True):
        file_list.append(filepath)
    return gdf.loc[[intersects_with_img(gdf[geom_col_name][i], file_list) for i in gdf.index]]

def get_x_bbox(bbox):
    try:
        return bbox[0]+bbox[2]
    except:
        return 'n/a'

def correct_duplicate_roi(bboxes_batch, output, labels):
    unique_bboxes = []
    keep_ind = []
    for i in range(len(bboxes_batch)) :
        if bboxes_batch[i] not in unique_bboxes and bboxes_batch[i].maxx-bboxes_batch[i].minx>90 and bboxes_batch[i].maxy-bboxes_batch[i].miny>90:          #TODO: remove hardcoding
            unique_bboxes.append(bboxes_batch[i])
            keep_ind.append(i)
    output = output[keep_ind]
    labels = labels[keep_ind]
    return output, labels


def _list_dict_to_dict_list(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    collated = collections.defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated


def stack_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    collated: Dict[Any, Any] = _list_dict_to_dict_list(samples)
    unique_bboxes = []
    keep_ind = []
    for i in range(len(collated['bbox'])):
        bbox = collated['bbox'][i]
        if bbox not in unique_bboxes and bbox.maxx-bbox.minx>90 and bbox.maxy-bbox.miny>90 :
            unique_bboxes.append(bbox)
            keep_ind.append(i)
    for key, value in collated.items():
        if isinstance(value[0], torch.Tensor):
            if len(value)==1:
                collated[key] = torch.stack(tuple(value))
            else:
                value = np.array(value)[keep_ind]
                collated[key] = torch.stack(tuple(value))
            
    return collated


def prepare_shapefile_dataset(
        shp_path, 
        img_dir, 
        filename_glob, 
        dataset,
        target_variable='C_id',
        geom_col_name = 'bboxes',
        sort_geographicaly=False,
        buffer_size=50,
        ):

    bb = dataset.index.bounds

    def polygon_to_bbox(polygon):
        bounds = list(polygon.bounds)
        bounds[1], bounds[2] = bounds[2], bounds[1]
        return BoundingBox(*bounds, bb[4], bb[5])

    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.dropna(subset=[target_variable])
    if gdf.geom_type.unique() == "Point":
        gdf.geometry = gdf.buffer(buffer_size, cap_style = 3)

    # changes labels id so they go from 0 to N-1, with N the total number of labels. Conserves labels numerical order
    labels = np.array(gdf[target_variable])
    ordered = nsmallest(len(np.unique(labels)), np.unique(labels))
    gdf[target_variable] = [ordered.index(i) for i in labels]

    # only conserves rois which intersect with the images from the dataset
    gdf[geom_col_name] = [polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
    gdf = get_intersected_bboxes(gdf, img_dir, filename_glob)

    if sort_geographicaly:
        gdf['x_temp'] = gdf['bboxes'].apply(get_x_bbox)
        gdf = gdf.sort_values('x_temp')
    gdf = gdf.drop_duplicates()
    gdf.index = [i for i in range(len(gdf))]
    print("Nb roi : ", len(gdf))
    return gdf

def prepare_shapefile_dataset_cont(
        shp_path, 
        img_dir, 
        filename_glob, 
        dataset,
        target_variable='C_id',
        geom_col_name = 'bboxes',
        sort_column=None,
        buffer_size=50,
        normalize=False,
        ):

    bb = dataset.index.bounds

    def polygon_to_bbox(polygon):
        bounds = list(polygon.bounds)
        bounds[1], bounds[2] = bounds[2], bounds[1]
        return BoundingBox(*bounds, bb[4], bb[5])

    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    gdf = gdf.dropna(subset=[target_variable])
    if gdf.geom_type.unique() == "Point":
        gdf.geometry = gdf.buffer(buffer_size, cap_style = 3)


    # # changes labels id so they go from 0 to N-1, with N the total number of labels. Conserves labels numerical order
    # labels = np.array(gdf[target_variable])
    # ordered = nsmallest(len(np.unique(labels)), np.unique(labels))
    # gdf[target_variable] = [ordered.index(i) for i in labels]

    # only conserves rois which intersect with the images from the dataset
    gdf[geom_col_name] = [polygon_to_bbox(gdf['geometry'][i]) for i in gdf.index]
    gdf = get_intersected_bboxes(gdf, img_dir, filename_glob)

    if sort_column:
        gdf = gdf.sort_values(sort_column)

    if normalize:
        gdf[target_variable] = (gdf[target_variable] - gdf[target_variable].mean()) / gdf[target_variable].std()

    gdf.index = [i for i in range(len(gdf))]
    print("Nb roi : ", len(gdf))
    return gdf

###############################"



def crop_duplicate(og_img_path, macro_img, multiple = 224):
    with rasterio.open(og_img_path) as ds :
        shape = ds.read(1).shape
        ds.close()
    print(shape)
    print(macro_img.shape)
    if macro_img.shape[0]>shape[0] and macro_img.shape[1]>shape[1]:
        macro_img = np.delete(macro_img, [multiple+i for i in range(multiple - shape[0]%multiple)], axis=0)
        macro_img = np.delete(macro_img, [multiple*(shape[1]//multiple)+i for i in range(multiple - shape[1]%multiple)], axis=1)
    print(macro_img.shape)
    return macro_img

def array_to_geotiff(
        array:np.ndarray, 
        output_file:str, 
        top_left_corner_coords:tuple, 
        pixel_width:float, 
        pixel_height:float,
        crs:CRS,
        dtype:str='float32',
        ):
    """
    Convert a numpy array to a GeoTIFF file.
    
    Parameters:
        array (numpy.ndarray): The numpy array representing the raster.
        output_file (str): The path to save the output GeoTIFF file.
        top_left_corner_coords (tuple): Tuple containing the coordinates (x, y) of the top left corner.
        pixel_width (float): Width of a pixel in the raster.
        pixel_height (float): Height of a pixel in the raster.
        crs (rasterio.crs.CRS compatible): CRS to apply.
        dtype (str): Data type of the output geotiff.
    """
    from rasterio.transform import from_origin
    # Get the dimensions of the array
    height, width, channels = array.shape
    
    # Define the transformation matrix
    transform = from_origin(top_left_corner_coords[0], top_left_corner_coords[1], pixel_width, pixel_height)
    
    # Create the GeoTIFF file
    with rasterio.open(output_file, 'w', driver='GTiff',
                       height=height, width=width, count=channels, dtype=dtype,
                       crs=crs, transform=transform) as ds:
        ds.write(np.transpose(array, (2, 0, 1)))

def create_bbox_shp(long0, lat0, lat1, long1):
    return Polygon([[long0, lat0], [long1, lat0], [long1, lat1], [long0, lat1]])

def aggregate_overlapping_polygons_with_same_label(gdf, attribute_name, attribute_value):
    polygons = gdf.loc[(gdf[attribute_name] == attribute_value)]
    single_multi_polygon = polygons['geometry'].unary_union
    try:
        polygons = single_multi_polygon.geoms
    except:
        polygons = [single_multi_polygon]
    return polygons

def export_on_map(
        labels, 
        bboxes, 
        crs, 
        out_path,
        aggregate=False,
        ):  

    fullpathname_cluster_shp = os.path.join(os.getcwd(), out_path)

    bboxes_shp = [create_bbox_shp(bboxes[i][0], 
                                  bboxes[i][2], 
                                  bboxes[i][3], 
                                  bboxes[i][1]
                                  ) for i in range(len(bboxes))]
    labels_shp = labels
    d = {'label': labels_shp, 'geometry': bboxes_shp}
    gdf = gpd.GeoDataFrame(d, crs = crs)
    gdf = gdf[gdf['label'] != -1]

    if aggregate:
        d = {'label': [], 'geometry': []}
        for label in gdf['label'].unique():
            current_polygons = aggregate_overlapping_polygons_with_same_label(gdf, 'label', label)
            for polygon in current_polygons:
                d['label'].append(label)
                d['geometry'].append(polygon)

        gdf = gpd.GeoDataFrame(d, crs = crs)
        print(gdf)
    gdf.to_file(fullpathname_cluster_shp, driver='ESRI Shapefile')


def get_stat_by_band(tif, band_number, stat='STATISTICS_MEAN'):
    '''
    reads metadata of geotif by specifying the band number and desired statistic
    '''
    with rasterio.open(tif) as src:
        statistic = src.tags(band_number)[stat]
    src.close()
    return  statistic


def get_mean_sd_by_band(path, compute_if_needed=False, ignore_zeros=True):
    '''
    Reads metadata or computes mean and sd of each band of a geotiff.
    If the metadata is not available, mean and standard deviation can be computed via numpy.

    Parameters
    ----------
    path : str
        path to a geotiff file
    ignore_zeros : boolean
        ignore zeros when computing mean and sd via numpy

    Returns
    -------
    means : list
        list of mean values per band
    sds : list
        list of standard deviation values per band
    '''

    src = rasterio.open(path)
    means = []
    sds = []

    for band in range(1, src.count+1):
        try:
            tags = src.tags(band)
            if 'STATISTICS_MEAN' in tags and 'STATISTICS_STDDEV' in tags:
                mean = float(tags['STATISTICS_MEAN'])
                sd = float(tags['STATISTICS_STDDEV'])
                means.append(mean)
                sds.append(sd)
            else:
                raise KeyError("Statistics metadata not found.")

        except KeyError:
            if compute_if_needed:
                arr = src.read(band)
                if ignore_zeros:
                    mean = np.ma.masked_equal(arr, 0).mean()
                    sd = np.ma.masked_equal(arr, 0).std()
                else:
                    mean = np.mean(arr)
                    sd = np.std(arr)
                means.append(float(mean))
                sds.append(float(sd))
            else : 
                warnings.warn("Statistics metadata not found and computation not enabled.", UserWarning)
        except Exception as e:
            print(f"Error processing band {band}: {e}")

    src.close()
    return means, sds


def get_crs(tif):
    with rasterio.open(tif) as src:
        crs = src.crs
    return crs


def get_geo_folds(input_shp,
                nfolds=6,
                seed=42,
                output_shp=None,
                # lat_variable='y_utm',
                # long_variable='x_utm'
                ):

    gdf = gpd.read_file(input_shp)
    gdf = gdf[gdf.geometry != None]
    # print(gdf)
    # gdf.reset_index(inplace=True)

    X = []
    for row in gdf.iterrows():
        index, data = row
        X.append([data.geometry.y, data.geometry.x])

    print("===== fitting kmeans =====")
    X = np.array(X)
    kmeans = KMeans(n_clusters=nfolds, random_state=seed).fit(X.astype('double'))
    folds = kmeans.labels_
    # increment all by one to avoir fold at 0
    folds = [x+1 for x in folds]

    # check distribution
    d = {}
    for x in folds:
        d[x] = d.get(x,0) + 1
     
    # printing result
    print(f"The list frequency of elements is : {d}" )

    gdf['geo_fold'] = folds

    if output_shp:
        gdf.to_file(output_shp)

    return gdf

def change_tif_resolution(orig_raster_path, dest_raster_path, new_resolution):

    # Open the original raster
    with rasterio.open(orig_raster_path) as orig_src:
        orig_array = orig_src.read()
        # Get the current resolution
        orig_resolution = orig_src.transform[0]
        # Calculate the factor by which to change the resolution
        resolution_factor = int(orig_resolution / new_resolution)

        # Calculate the new shape of the array
        new_shape = (
            orig_array.shape[0],
            orig_array.shape[1] * resolution_factor,
            orig_array.shape[2] * resolution_factor,
        )

        # Create a new array with the desired resolution
        dest_array = np.zeros(new_shape, dtype=orig_array.dtype)

        # Iterate through the original array and assign values to the new array
        for b in range(orig_array.shape[0]):
            for i in range(orig_array.shape[1]):
                for j in range(orig_array.shape[2]):
                    dest_array[
                            b, 
                            i * resolution_factor : (i + 1) * resolution_factor, 
                            j * resolution_factor : (j + 1) * resolution_factor
                            ] = orig_array[b, i, j]

        # Get the metadata from the original raster
        dest_meta = orig_src.meta.copy()

        # Update metadata with the new resolution
        dest_meta['transform'] = rasterio.Affine(new_resolution, 0, orig_src.transform[2], 0, -new_resolution, orig_src.transform[5])
        dest_meta['width'] = dest_array.shape[2]
        dest_meta['height'] = dest_array.shape[1]

    # Write the new raster
    with rasterio.open(dest_raster_path, 'w', **dest_meta) as dest_dst:
        dest_dst.write(dest_array)


def get_random_points_on_raster(raster_path, num_points=100):
    with rasterio.open(raster_path) as src:
        # Get raster metadata
        width = src.width
        height = src.height

        # Read raster data as numpy array
        raster_array = src.read()

        # Generate random points within the extent of the raster
        xmin, ymin, xmax, ymax = src.bounds
        x_points = np.random.uniform(xmin, xmax, num_points)
        y_points = np.random.uniform(ymin, ymax, num_points)

        # Extract pixel values at random points
        pixel_values = []
        for x, y in zip(x_points, y_points):
            # Convert point coordinates to pixel coordinates
            row, col = src.index(x, y)

            # Extract pixel value
            if 0 <= row < height and 0 <= col < width:
                pixel_value = raster_array[:, row, col]  # Extract all bands
                pixel_values.append(int(pixel_value[0]))

        # Create GeoDataFrame with random points and pixel values
        geometry = [Point(x, y) for x, y in zip(x_points, y_points)]
    return gpd.GeoDataFrame({'pixel_value': pixel_values}, geometry=geometry, crs=src.crs)


def get_random_points_on_raster_template(raster_path,template_path, num_points=100):
    
    from pyproj import CRS
    from pyproj import Transformer

    with rasterio.open(template_path) as tsrc:
        txmin, tymin, txmax, tymax = tsrc.bounds
        tcrs = tsrc.crs

    with rasterio.open(raster_path) as src:
        # Get raster metadata
        width = src.width
        height = src.height

        transformer = Transformer.from_crs(tcrs, src.crs)
        xmin, ymin = transformer.transform(txmin, tymin)
        xmax, ymax = transformer.transform(txmax, tymax)

        # Read raster data as numpy array
        raster_array = src.read()

        # Generate random points within the extent of the raster
        # xmin, ymin, xmax, ymax = src.bounds
        x_points = np.random.uniform(xmin, xmax, num_points)
        y_points = np.random.uniform(ymin, ymax, num_points)

        # Extract pixel values at random points
        pixel_values = []
        for x, y in zip(x_points, y_points):
            # Convert point coordinates to pixel coordinates
            row, col = src.index(x, y)

            # Extract pixel value
            if 0 <= row < height and 0 <= col < width:
                pixel_value = raster_array[:, row, col]  # Extract all bands
                pixel_values.append(int(pixel_value[0]))

        # Create GeoDataFrame with random points and pixel values
        geometry = [Point(x, y) for x, y in zip(x_points, y_points)]
    return gpd.GeoDataFrame({'pixel_value': pixel_values}, geometry=geometry, crs=src.crs)


def get_geo_folds(gdf,
                  nfolds=6,
                  seed=42,
                  verbose=False,
                  fold_col_name = 'fold',
                  ):

    if len(gdf) != len(gdf[gdf.geometry != None]):
        print('please check that all geometry are valid')

    X = []
    for row in gdf.iterrows():
        index, data = row
        X.append([data.geometry.y, data.geometry.x])

    X = np.array(X)
    kmeans = KMeans(n_clusters=nfolds, random_state=seed).fit(X.astype('double'))
    folds = kmeans.labels_
    # increment all by one to avoid fold at 0
    folds = [x+1 for x in folds]

    if verbose:
        # check distribution
        d = {}
        for x in folds:
            d[x] = d.get(x,0) + 1
        print(f"The list frequency of elements is : {d}" )

    gdf[fold_col_name] = folds

    return gdf

def attention_stats_shp(shp_path, attention_tif_path):
    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    
    with rasterio.open(attention_tif_path) as ds :
        tf = ds.meta.copy()['transform']
        bb = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
        gdf = gdf.loc[((gdf['geometry'].bounds['maxx']>bb[0]) &(gdf['geometry'].bounds['maxy']>bb[2]) &(gdf['geometry'].bounds['minx']<bb[1]) &(gdf['geometry'].bounds['miny']<bb[3]))]
        gdf.index = [i for i in range(len(gdf))]
        gdf['bboxes'] = [polygon_to_bbox(gdf['geometry'][i]) for i in range(len(gdf))]
        im = np.transpose(ds.read(), (1,2,0))
        print("image shape : ", im.shape)
        nb_chan = im.shape[-1]
        mean = []
        std = []
        for roi in gdf['bboxes']:
            bottom_left = ds.index(roi[0], roi[3])
            top_right = ds.index(roi[1], roi[2])
            roi_attention = im[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]
            mean.append(np.array([np.mean(roi_attention[:,:,k]) for k in range(nb_chan)]))
            std.append(np.array([np.std(roi_attention[:,:,k]) for k in range(nb_chan)]))

        gdf['attention_mean'] = mean
        gdf['attention_std'] = std
            
    return gdf, np.array(mean), np.array(std)
