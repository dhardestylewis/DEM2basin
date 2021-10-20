## GeoFlood preprocessing 1m DEM data
## Author: Daniel Hardesty Lewis

## Import needed modules
import argparse

import numpy as np
import csv

import fiona
import pandas as pd
import geopandas as gpd

import utm

from osgeo import gdal

import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject#, Resampling
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
import rasterio.mask
from rasterio.merge import merge
from rasterio.features import shapes
from rasterio.crs import CRS

#from pyproj import CRS
import pyproj

import os
from pathlib import Path, PurePath
import shutil

from threading import Thread
from collections import deque
import multiprocessing
from itertools import repeat
from functools import reduce
import uuid

import pickle
import tblib.pickling_support
tblib.pickling_support.install()

import time
import logging
import traceback

import sys
import psutil
import gc
import tempfile
#from memory_profiler import profile

import glob
import re

def argparser():
    """
    Parses command-line arguments for dem2basin run as a script

    :return: ArgumentParser.parse_args object for dem2basin
    :rtype: parser.parse_args
    """
    ## Define input and output file locations

    parser = argparse.ArgumentParser()

    ## NHD catchment and flowline vector data
    parser.add_argument(
        "-n",
        "--nhd",
        type=str,
        help="NHD MR GIS files with layers labelled Flowline and Catchment"
    )
    ## WBD HUC vector data
    parser.add_argument(
        "-u",
        "--hucs",
        type=str,
        help="WBD HUC dataset"
    )
    ## Input vector data with single polygon of study area
    parser.add_argument(
        "-s",
        "--shapefile",
        type=str,
        help="Vector GIS file with single polygom of the study area"
    )
    ## Parent directory of TNRIS LIDAR projects
    parser.add_argument(
        "-r",
        "--lidar_parent_directory",
        type=str,
        help="Parent directory of LIDAR projects"
    )
    ## Distance to buffer output raster
    parser.add_argument(
        "-b",
        "--buffer",
        type=float,
        help="Optional distance to buffer the output raster"
    )
    ## TNRIS LIDAR availability vector data
    parser.add_argument(
        "-a",
        "--lidar_availability",
        type=str,
        help="TNRIS GIS vector file of available LIDAR data"
    )
    ## Optional directory for all of the outputs
    ##  (The outputs will be organized by HUC)
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Optional directory for the outputs"
    )
    ## Log file
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        help="Optional log file"
    )
    ## Overwrite existing outputs within input area
    parser.add_argument(
        "-o",
        "--overwrite",
        action='store_true',
        help="Optional flag to overwrite files found in the output directory"
    )
    ## Overwrite existing outputs within input area
    parser.add_argument(
        "-f",
        "--overwrite_flowlines",
        action='store_true',
        help="Optional flag to overwrite just the flowlines file"
    )
    ## Overwrite existing outputs within input area
    parser.add_argument(
        "-c",
        "--overwrite_catchments",
        action='store_true',
        help="Optional flag to overwrite just the catchments file"
    )
    ## Overwrite existing outputs within input area
    parser.add_argument(
        "-m",
        "--overwrite_roughnesses",
        action='store_true',
        help="Optional flag to overwrite just the roughness file"
    )
    ## Overwrite existing outputs within input area
    parser.add_argument(
        "-t",
        "--overwrite_rasters",
        action='store_true',
        help="Optional flag to overwrite just the raster file"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "-i",
        "--restart",
        type=str,
        help="Restart from existing pickle or create if missing"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--memfile",
        action='store_true',
        help="Enable RasterIO's MemoryFile"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--tempdir",
        type=str,
        help="Optional directory for temporary files"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--percent_free_mem",
        type=float,
        help="Percent memory to keep free"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--percent_free_disk",
        type=float,
        help="Percent disk usage to keep free"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--lowest_resolution",
        action='store_true',
        help="Optional flag to prefer lowest resolution in mosaics"
    )
    ## Restart from intermediate files or create intermediate files if missing
    parser.add_argument(
        "--highest_resolution",
        action='store_true',
        help="Optional flag to prefer highest resolution in mosaics (default)"
    )

    args = parser.parse_args()

    ## Check that these required input files have been defined
    if not args.shapefile:
        parser.error('-s --shapefile Input shapefile cutline not specified')
    if not args.hucs:
        parser.error('-u --hucs Input HUC shapefile not specified')
    if not args.nhd:
        parser.error('-n --nhd Input NHD geodatabase not specified')
    if not args.lidar_parent_directory:
        parser.error('-r --raster Input raster not specified')
    if not args.lidar_availability:
        parser.error('-a --lidar_availability Availability shapefile not specified')

    return(args)

def _drop_index_columns(
    dataframe,
    inplace = False
):
    """
    drops columns named 'index', 'index_left', and 'index_right'
    either to prevent issues with geopandas functions
    like geopandas.sjoin and to clean up after some geopandas functions

    :param dataframe: pandas.DataFrame possibly
        with 'index', 'index_left', or 'index_right' columns
    :type dataframe: pandas.DataFrame
    :return: pandas.DataFrame without 'index', 'index_left', 'index_right'
        columns
    :rtype: pandas.DataFrame
    """

    dataframe_without_index_columns = dataframe.drop(
        columns = [
            'index',
            'index_left',
            'index_right'
        ],
        errors = 'ignore',
        inplace = inplace
    )

    return(dataframe_without_index_columns)

def _drop_index_columns_inplace(
    dataframes
):
    """
    drop index columns inplace for multiple dataframes

    :param dataframes: sequence of pandas.DataFrames possibly
        with 'index', 'index_left', or 'index_right' columns
    :type dataframes: Sequence[pandas.DataFrame]
    :return: inplace modification of dataframes with 'index', 'index_left', 
        'index_right' columns removed
    :rtype: NoneType
    """
    ## TODO: untested
    ## TODO: look into merging with _drop_index_columns by extending
    ##     functionality

    for dataframe in dataframes:
        _drop_index_columns(dataframe,inplace=True)

def find_huc_level(
    hucs
):
    """
    returns the name of the first column attribute found named "HUC[0-9]*"

    :param hucs: vector filename, pathlib.PurePath, or pandas.DataFrame with a
        column named "HUC[0-9]*"
    :type hucs: Union[str,pathlib.PurePath,pandas.DataFrame]
    :return: string of name of first column found matching "HUC[0-9]*"
    :rtype: str
    """
    ## TODO: test pathlib extension

    hucs_input = read_file_or_gdf(hucs)

    regexp = re.compile('HUC[0-9]*')
    huc_level = list(filter(
        regexp.match,hucs_input.columns.to_list()
    ))[0]

    return(huc_level)

def set_index_to_huc(
    hucs,
    sort = True
):
    """
    find HUC attribute of shape object and sets the index of this
    shape object to that attribute

    :param hucs: vector filename, pathlib.PurePath, or pandas.DataFrame with a
        column named "HUC[0-9]*"
    :type hucs: Union[str,pathlib.PurePath,pandas.DataFrame]
    :param sort: boolean to sort or not sort the resulting
        geopandas.GeoDataFrame. Defaults to True.
    :type sort: bool
    :return: geopandas.GeoDataFrame with index set to HUC attribute
    :rtype: geopandas.GeoDataFrame
    """
    ## TODO: test pathlib extension

    huc_level = find_huc_level(hucs)

    hucs = read_file_or_gdf(hucs)

    hucs_with_huc_index = hucs.set_index(huc_level,drop=False)
    hucs_with_huc_index.index.name = 'HUC'
    hucs_with_huc_index.index = hucs_with_huc_index.index.astype('int64')

    if sort:
        hucs_with_huc_index.sort_index(inplace=True)

    return(hucs_with_huc_index)

def read_file_or_gdf(
    shape,
    **kwargs
):
    """
    enables functions to take either a filename, pathlib.PurePath object, or
    geopandas.GeoDataFrame as input

    :param shape: filename, path object, or geopandas.GeoDataFrame for more
        flexible function input
    :type shape: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
    :param **kwargs: keyword arguments for geopandas.read_file
    :return: geopandas.GeoDataFrame for shape parameter
    :rtype: geopandas.GeoDataFrame
    """

    if isinstance(shape,(str,PurePath)):
        shape_input = gpd.read_file(shape,**kwargs)
    else:
        shape_input = shape.copy()

    return(shape_input)

def get_hucs_by_shape(
    shape,
    hucs,
    hucs_layer = None,
    sort = True,
    select_utm = None,
    to_utm = True,
    drop_index_columns = True
):
    """
    finds HUCs that intersect a study area given as a vector image

    :param shape: filename, pathlib.PurePath, or geopandas.GeoDataFrame input
        vector image of area of interest
    :type shape: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
    :param hucs: filename, pathlib.PurePath, geopandas.GeoDataFrame Watershed
        Boundary Dataset (WBD) input vector image with a column named
        "HUC[0-9]*"
    :type hucs: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
    :param hucs_layer: optional name of HUC layer,
        for example "HUC12", "HUC8", "HUC6", "HUC4", etc. Defaults to None.
    :type hucs_layer: str
    :param sort: boolean to sort output by HUC ID
    :type sort: bool
    :param select_utm: integer for UTM zone to explicitly select.
        Defaults to None.
    :type select_utm: int
    :param to_utm: boolean to reproject to UTM or not. Defaults to True.
    :type to_utm: bool
    :param drop_index_columns: boolean to drop columns named 'index',
        'index_left', 'index_right' from output. Defaults to True.
    :type drop_index_columns: bool
    :return: geopandas.GeoDataFrame of HUCs that cover area of interest
    :rtype: geopandas.GeoDataFrame
    """
    ## Find the HUCs that intersect with the input polygon

    #shape_input = 'data/TX-Counties/Young/TX-County-Young.shp'
    shape_gdf = read_file_or_gdf(shape)

    shape_gdf['dissolve'] = True
    shape_gdf = shape_gdf.dissolve(by='dissolve').reset_index(drop=True)
    shape_gdf = gpd.GeoDataFrame(shape_gdf[['geometry']])

    #hucs_input = 'data/WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp'
    hucs_gdf = read_file_or_gdf(hucs,layer=hucs_layer)

    crs = find_utm(hucs_gdf,select_utm=select_utm)
    if not to_utm:
        hucs_original = hucs_gdf.copy()
    to_crs(crs,[hucs_gdf,shape_gdf])

    if drop_index_columns:
        hucs_gdf = _drop_index_columns(hucs_gdf)

    hucs_gdf = gpd.overlay(
        hucs_gdf,
        shape_gdf,
        how = 'intersection'
    )

    hucs_gdf = set_index_to_huc(hucs_gdf,sort)
    if not to_utm:
        hucs_original = set_index_to_huc(hucs_original,sort)
        hucs_gdf = hucs_original.loc[hucs_original.index.isin(hucs_gdf.index)]

    return(hucs_gdf)

def set_and_sort_index(
    dataframe,
    column,
    drop = True,
):
    """
    sets a pandas dataframe’s index to column and sorts by that column

    :param dataframe: pandas.DataFrame whose index will be set to column and
        then sorted by that new index
    :type dataframe: pandas.DataFrame
    :param column: column name which will be the new index of this
        pandas.DataFrame
    :type column: str
    :param drop: boolean whether to drop original column the index
        is being set to. Defaults to True.
    :type drop: bool
    :return: pandas.DataFrame with index set to column attribute and sorted by
        that new index
    :rtype: pandas.DataFrame
    """

    dataframe.set_index(column,inplace=True,drop=drop)
    dataframe.sort_index(inplace=True)

    return(dataframe)

def index_dataframe_by_dataframe(dataframe_left,dataframe_right):
    """
    indexes a dataframe by another dataframe

    :param dataframe_left: pandas.DataFrame of which certain index values will
        be selected
    :type dataframe_left: pandas.DataFrame
    :param dataframe_right: pandas.DataFrame whose index will be selected
        against
    :type dataframe_right: pandas.DataFrame
    :return: pandas.DataFrame after index has been selected from index values of
        other pandas.DataFrame
    :rtype: pandas.DataFrame
    """

    dataframe = dataframe_left[
        dataframe_left.index.isin(dataframe_right.index)
    ]
    return(dataframe)

def get_nhd_by_shape(
    shape,
    nhd,
    layer = None,
#    comid_only = True,
    drop_index_columns = True,
    comid_column = None,
    fix_invalid_geometries = False
):
    """
    retrieves specific NHD layer masked by another geodataframe

    :param shape: geopandas.GeoDataFrame of vector image area of interest
    :type shape: geopandas.GeoDataFrame
    :param nhd: filename, pathlib.PurePath, or geopandas.GeoDataFrame input of
        National Hydrography Dataset Medium Resolution (NHD MR) vector image
    :type nhd: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
    :param layer: name of NHD MR layer, for example 'Catchment' or 'Flowline'.
        Default is None.
    :type layer: str
    :param drop_index_columns: boolean whether to drop columns named 'index',
        'index_left', 'index_right' from the output. Default is True.
    :type drop_index_columns: bool
    :param comid_column: column name of the COMID attribute in NHD MR,
        for example 'COMID' or 'FEATUREID'. Default is None.
    :type comid_column: str
    :param fix_invalid_geometries: boolean to fix invalid geometries in output.
        Default is False.
    :type fix_invalid_geometries: bool
    :return: geopandas.GeoDataFrame of NHD MR layer that covers the area of
        interest and has been re-indexed by COMID
    :rtype: geopandas.GeoDataFrame
    """
    ## Identify flowlines of each HUC

    #nhd_file = 'data/NFIEGeo_12.gdb'
    ## Find the flowlines whose representative points are within these HUCs
    nhd_layer = read_file_or_gdf(nhd,layer=layer,mask=shape)

    if drop_index_columns:
        nhd_layer = _drop_index_columns(nhd_layer)

    if comid_column is not None:
        nhd_layer.rename(columns={comid_column:'COMID'},inplace=True)
    nhd_layer = set_and_sort_index(nhd_layer,'COMID')

    if fix_invalid_geometries:
        nhd_layer.geometry = nhd_layer.buffer(0)

    return(nhd_layer)

def get_representative_points(
    flowlines,
    hucs,
    drop_index_columns = True,
    set_index_to_comid = False
):
    """
    retrieve representative points of flowlines and assign HUCs to these points

    :param shape: geopandas.GeoDataFrame of NHD flowlines, possibly with 'COMID'
        column
    :type shape: geopandas.GeoDataFrame
    :param hucs: geopandas.GeoDataFrame of HUCs with column named 'HUC'
    :type hucs: geopandas.GeoDataFrame
    :param drop_index_columns: boolean whether to drop columns named 'index',
        'index_left', 'index_right'. Default is True.
    :type drop_index_columns: bool
    :param set_index_to_comid: boolean whether to set index to COMID column.
        Default is False.
    :type set_index_to_comid: bool
    :return: geopandas.GeoDataFrame of representative points of each flowline
    :rtype: geopandas.GeoDataFrame
    """

    flowline_representative_points = flowlines.copy()
    flowline_representative_points['geometry'] = flowlines.representative_point()
    flowline_representative_points = _drop_index_columns(
        flowline_representative_points
    )

    hucs = hucs.reset_index(drop=False)

    flowline_representative_points = gpd.sjoin(
        flowline_representative_points,
        gpd.GeoDataFrame(
            hucs[['HUC','geometry']]
        ).to_crs(flowline_representative_points.crs),
        op = 'intersects',
        how = 'inner'
    )

    if drop_index_columns:
        flowline_representative_points = _drop_index_columns(
            flowline_representative_points
        )

    if set_index_to_comid:
        flowline_representative_points = set_and_sort_index(
            flowline_representative_points,
            'COMID',
            drop = False
        )
    
    return(flowline_representative_points)

def set_roughness_by_streamorder(
    flowlines_original,
    streamorder_col = 'StreamOrde',
    roughness_col = 'Roughness'
):
    """
    assign Manning’s n roughness value by each flowline’s stream order

    :param flowlines_original: geopandas.GeoDataFrame of flowlines with a
        stream order column attribute
    :type flowlines_original: geopandas.GeoDataFrame
    :param streamorder_col: column name of stream order attribute
    :type streamorder_col: str
    :param roughness_col: column name of Manning's n roughness attribute
    :type roughness_col: str
    :return: geopandas.GeoDataFrame of flowlines with the Manning's n roughness
        column set by stream order
    :rtype: geopandas.GeoDataFrame
    """

    flowlines = flowlines_original.copy()

    flowlines.loc[ flowlines[streamorder_col] == 0 , roughness_col ] = .99
    flowlines.loc[ flowlines[streamorder_col] == 1 , roughness_col ] = .2
    flowlines.loc[ flowlines[streamorder_col] == 2 , roughness_col ] = .1
    flowlines.loc[ flowlines[streamorder_col] == 3 , roughness_col ] = .065
    flowlines.loc[ flowlines[streamorder_col] == 4 , roughness_col ] = .045
    flowlines.loc[ flowlines[streamorder_col] == 5 , roughness_col ] = .03
    flowlines.loc[ flowlines[streamorder_col] == 6 , roughness_col ] = .01
    flowlines.loc[ flowlines[streamorder_col] == 7 , roughness_col ] = .025

    return(flowlines)

def clip_dataframe_by_attribute(
    dataframe,
    dataframe_with_attribute,
    attribute = None
):
    """
    assign attribute from one geodataframe to another
    by their mutual index values

    :param dataframe: pandas.DataFrame to merge single attribute column
        from other pandas.DataFrame that has this attribute
    :type dataframe: pandas.DataFrame
    :param dataframe_with_attribute: other pandas.DataFrame that has this
        attribute
    :type dataframe_with_attribute: pandas.DataFrame
    :param attribute: attribute column name
    :type attribute: str
    :return: new pandas.DataFrame with attribute merged
    :rtype: pandas.DataFrame
    """

    ## Find the flowlines corresponding with these catchments
    ##  (Note: this line is optional.
    ##  Commenting it out will result in non-COMID-identified flowlines)
    #if comid_only==True:
    dataframe = index_dataframe_by_dataframe(
        dataframe,
        dataframe_with_attribute
    )

    ## Determine which HUCs each of the flowlines and catchments belong to
    dataframe[attribute] = dataframe_with_attribute.loc[
        dataframe.index,
        attribute
    ]

    return(dataframe)

def find_common_utm(shape_original):
    """
    determines the mode of the UTMs of the representative points of a
    geodataframe’s geometries

    :param dataframe: pandas.DataFrame to merge single attribute column
        from other pandas.DataFrame that has this attribute
    :type dataframe: pandas.DataFrame
    :param dataframe_with_attribute: other pandas.DataFrame that has this
        attribute
    :type dataframe_with_attribute: pandas.DataFrame
    :param attribute: attribute column name
    :type attribute: str
    :return: new pandas.DataFrame with attribute merged
    :rtype: pandas.DataFrame
    """
    ## Determine whether the administrative division is within a single UTM

    shape = shape_original.to_crs('epsg:4326')

    ## TODO: make this majority count an option
    ##  and bring back cross-utm error as default behaviour
    output_utm = shape.representative_point().apply(
        lambda p: utm.latlon_to_zone_number(p.y,p.x)
    ).value_counts().idxmax()

    #if output_utm.shape[0] > 1:
    #    print("ERROR: Cross-UTM input shapefile not yet supported.")
    #    sys.exit(0)

    return(output_utm)

def find_utm(
    gdf_original,
    select_utm = None
):
    """
    finds a single UTM CRS best suited for the geometries of a geodataframe
    """
    ## Buffer the catchments for each HUC

    gdf = gdf_original.reset_index()

    ## Are the catchments all within the same UTM?
    if select_utm:
        utm_output = select_utm
    else:
        utm_output = find_common_utm(gdf)

    ## Buffer the HUC catchments
    if gdf.crs.datum.name == 'World Geodetic System 1984':
        #crs = CRS(proj='utm', zone=utm_output[0], datum='WGS84')
        if utm_output == 13:
            crs = 'epsg:32613'
        elif utm_output == 14:
            crs = 'epsg:32614'
        elif utm_output == 15:
            crs = 'epsg:32615'
        else:
            print("ERROR: UTMs outside of 13-15 not yet supported.")
            if hasattr(main,'__file__'):
                sys.exit(0)
    elif (
        gdf.crs.datum.name == 'North American Datum 1983' or
        gdf.crs.datum.name == 'D_NORTH_AMERICAN_1983' or
        gdf.crs.datum.name == 'NAD83 (National Spatial Reference System 2011)' or
        gdf.crs.datum.name == 'NAD83'
    ):
        #crs = CRS(proj='utm', zone=utm_output[0], datum='NAD83')
        if utm_output == 13:
            crs = 'epsg:6342'
        elif utm_output == 14:
            crs = 'epsg:6343'
        elif utm_output == 15:
            crs = 'epsg:6344'
        else:
            print("ERROR: UTMs outside of 13-15 not yet supported.")
            if hasattr(main,'__file__'):
                sys.exit(0)
    else:
        print("ERROR: Non-WGS/NAD datum not yet supported")
        if hasattr(main,'__file__'):
            sys.exit(0)

    ## Are the buffered catchments all within the same UTM?
    #find_output_utm(gdf)

    return(crs)

def reproject_and_buffer(gdf_original,crs,meters_buffered=500.):
    """
    reprojects geodataframe to a CRS and then buffers it
    """

    gdf = gdf_original.to_crs(crs)
    gdf['geometry'] = gdf.buffer(meters_buffered)
    gdf.crs = crs

    return(gdf)

def reproject_to_utm_and_buffer(gdf_original,select_utm=None):
    """
    finds best UTM for a geodataframe, reprojects, and then buffers it
    """
    ## Reproject a GeoDataFrame to the most common UTM and then buffer it

    crs = find_utm(gdf_original,select_utm)
    gdf = reproject_and_buffer(gdf_original,crs)

    return(gdf)

def get_filelist_from_parent_directory(parent_directory):

    filelist = Path(str(parent_directory)).glob('*')

    return(filelist)

def get_data_polygon_for_raster(raster_filename):

    with rasterio.open(str(raster_filename)) as src:

        results = (
            {
                'properties' : {'raster_val' : v},
                'geometry' : s
            }
            for _, (s, v)
            in enumerate(shapes(
                src.read_masks(1),
                transform = src.transform
            ))
        )

    geoms = list(results)

    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)

    return(gpd_polygonized_raster)

def get_data_polygons_for_each_raster(raster_filelist):

    polygonized_rasters = []

    for raster_filename in raster_filelist:

        gpd_polygonized_raster = get_data_polygon_for_raster(raster_filename)

        polygonized_rasters.append(gpd_polygonized_raster)

    return(polygonized_rasters)

def find_raster_filenames(
    parent_directory,
    get_filesizes = False,
    get_crs = False,
    get_epsg = False
#    get_geographic_crs = False
):

    filetypes = ('*.img', '*.dem', '*.tif')

    fathom_filenames = []
    for filetype in filetypes:
        fathom_filenames.extend(list(
            Path(str(parent_directory)).rglob(os.path.join(
                'dem',
                filetype
            ))
        ))

    fathom_filenames_lowercased = []
    fathom_filesizes = []
    fathom_file_crs = []
    fathom_file_epsg = []
#    fathom_file_geographic_crs = []
    for fathom_filename in fathom_filenames:
        fathom_filenames_lowercased.append(
            os.path.splitext(os.path.join(*fathom_filename.parts).lower())[0]
        )
        if get_filesizes:
            fathom_filesizes.append(Path(fathom_filename).stat().st_size)
        if get_crs:
            fathom_file_crs.append(
                pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection())
            )
        if get_epsg:
            fathom_file_epsg = fathom_file_crs[-1].to_epsg()
## TODO: find correct function to get geographic CRS here
#        if get_geographic_crs:
#            fathom_file_geographic_crs = fathom_file_crs[-1].

    fathom_filenames = pd.DataFrame(
        data = {
            'filename' : fathom_filenames,
            'filename_lowercased' : fathom_filenames_lowercased,
            'filesize' : fathom_filesizes,
            'crs' : fathom_file_crs,
            'epsg' : fathom_file_epsg
#            'geographic_crs' : fathom_file_geographic_crs
        }
    )

    return(fathom_filenames)

def index_fathom_files(
#    self,
    fathom_parent_directory,
    hucs = None,
    availability_file = None,
    new_availability_file = None,
    drop_index_columns = True,
    get_filesizes = False,
    get_crs = False,
    get_epsg = False
#    get_geographic_crs = False
):
    """
    Georeference Fathom 3m raster dataset, with option to associate by HUC
    
    :param fathom_parent_directory: str or pathlib.PurePath of Fathom3m parent
        directory, expected to contain a directory named 'dem' under which the
        filenames are found
    :type fathom_parent_directory: Union[str,pathlib.PurePath]
    :param hucs: filename, pathlib.PurePath, or geopandas.GeoDataFrame of 
        HUCs. If provided, modifies output geopandas.GeoDataFrame to intersect
        availability file with HUCs, usually resulting in repeated filename rows
        for adjacent HUCs. Defaults to None.
    :type hucs: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
    :param availability_file: input filename of existing Fathom3m availability
        file. Defaults to None.
    :type availability_file: str
    :param new_availability_file: output filename of availability with found.
        Defaults to None.
    :type new_availability_file: str
    :param drop_index_columns: boolean whether to drop columns names 'index',
        'index_left', or 'index_right' from output. Defaults to True.
    :type drop_index_columns: bool
    :param get_filesizes: get the filesizes of each 
    :type get_filesizes: bool
    :return: geopandas.GeoDataFrame with filenames found in the Fathom3m parent
        directory
    :rtype: geopandas.GeoDataFrame
    """
    ## TODO: divide into:
    ##   - correcting the LIDAR availability file, and
    ##   - applying HUCs column attribute

    hucs_gdf = read_file_or_gdf(hucs)

    if availability_file is not None:
        availability = gpd.read_file(availability_file,mask=hucs_gdf)
    else:
        availability = get_data_polygons_for_each_raster()

    try:
        availability = availability[
            availability['demname'] != 'No Data Exist'
        ]
    except:
        pass

    if drop_index_columns:
        availability = _drop_index_columns(availability)

    if hucs_gdf is not None:
        availability = gpd.sjoin(
            availability,
            hucs_gdf[['HUC','geometry']].to_crs(availability.crs),
            how = 'inner',
            op = 'intersects'
        )

    find_raster_filenames(
        fathom_parent_directory,
        get_filesizes = False,
        get_crs = False,
        get_epsg = False
        #get_geographic_crs = False
    )

#    filetypes = ('*.img', '*.dem', '*.tif')
#    fathom_filenames = []
#    for filetype in filetypes:
#        fathom_filenames.extend(list(
#            Path(str(fathom_parent_directory)).rglob(os.path.join(
#                'dem',
#                filetype
#            ))
#        ))
#    fathom_filenames_lowercased = []
#    fathom_filesizes = []
#    for fathom_filename in fathom_filenames:
#        fathom_filenames_lowercased.append(
#            os.path.splitext(os.path.join(*fathom_filename.parts).lower())[0]
#        )
#        fathom_filesizes = Path(fathom_filename).stat().st_size
#
#    fathom_filenames = pd.DataFrame(
#        data = {
#            'filename' : fathom_filenames,
#            'filename_lowercased' : fathom_filenames_lowercased
#            'filesize' : fathom_filesizes
#        }
#    )

    availability['possible_filename'] = availability[['demname']].apply(
#        lambda row: os.path.join(
#            os.path.join(*Path(str(fathom_parent_directory)).parts),
#            'fathom3m',
#            'dem',
#            row[1]
#        ),
        lambda row: Path(str(fathom_parent_directory)).joinpath(*Path(
            'fathom3m',
            'dem',
            str(row[1])
        ).parts),
        axis = 1
    )
    availability['possible_filename_lowercased'] = availability[
        'possible_filename'
    ].apply(
        lambda filename: filename.lower()
    )

    availability = availability.merge(
        fathom_filenames,
        on = 'possible_filename_lowercased'
    )
    availability.drop(
        columns = ['possible_filename','possible_filename_lowercased'],
        inplace = True
    )

    availability['filename'] = availability['filename'].apply(
        lambda filename: str(filename)
    )

    if new_availability_file is not None:
        availability.to_file(new_availability_file)

    return(availability)
    
class LidarIndex():
    """
    Georeference TNRIS LIDAR 1m raster dataset
    """

    def __init__(
        self,
        hucs = None,
        lidar_availability_file = None,
        lidar_parent_directory = None
    ):

        if hucs is None:
            self.hucs = gpd.GeoDataFrame()
        else:
            self.hucs = hucs.copy()

        if lidar_availability_file is None:
            self.lidar_availability_file = ''
        else:
            self.lidar_availability_file = lidar_availability_file

        if lidar_parent_directory is None:
            self.lidar_parent_directory = ''
        else:
            self.lidar_parent_directory = lidar_parent_directory

    def index_lidar_files(
        self,
        lidar_availability_file,
        lidar_parent_directory,
        hucs = None,
        new_lidar_availability_file = None,
        drop_index_columns = True
    ):
        """
        Georeference TX Lidar 1m raster dataset, with option to associate by HUC
        
        :param lidar_parent_directory: str or pathlib.PurePath of TX Lidar
            parent directory, expected to contain a directory named 'dem' under
            which the filenames are found
        :type lidar_parent_directory: Union[str,pathlib.PurePath]
        :param hucs: filename, pathlib.PurePath, or geopandas.GeoDataFrame of 
            HUCs. If provided, modifies output geopandas.GeoDataFrame to
            intersect availability file with HUCs, usually resulting in repeated
            filename rows for adjacent HUCs.
        :type hucs: Union[str,pathlib.PurePath,geopandas.GeoDataFrame]
        :param availability_file: input filename of existing Lidar
            availability file.
        :type availability_file: str
        :param new_availability_file: output filename of availability with
            found. Defaults to None.
        :type new_availability_file: str
        :param drop_index_columns: boolean whether to drop columns names
            'index', 'index_left', or 'index_right' from output. Defaults to
            True.
        :type drop_index_columns: bool
        :return: geopandas.GeoDataFrame with filenames found in the Lidar parent
            directory
        :rtype: geopandas.GeoDataFrame
        """
        ## TODO: divide into:
        ##   - correcting the LIDAR availability file, and
        ##   - applying HUCs column attribute
    
        availability = gpd.read_file(lidar_availability_file,mask=hucs)
        availability = availability[availability['demname']!='No Data Exist']

        if drop_index_columns:
            availability = _drop_index_columns(availability)

        filetypes = ('*.img', '*.dem', '*.tif')
        lidardatafiles = []
        for filetype in filetypes:
            lidardatafiles.extend(list(
                Path(lidar_parent_directory).rglob(os.path.join(
                    '*',
                    'dem',
                    filetype
                ))
            ))
        lidardatafileslower = [
            os.path.splitext(os.path.join(*fn.parts).lower())[0]
            for fn
            in lidardatafiles
        ]
        lidardatafiles = pd.DataFrame(
            data = {
                'lidar_file': lidardatafiles,
                'pathlower': lidardatafileslower
            }
        )

        availability['path'] = availability[['dirname','demname']].apply(
            lambda row: os.path.join(
                os.path.join(*Path(lidar_parent_directory).parts),
                row[0],
                'dem',
                row[1]
            ),
            axis = 1
        )
        availability['pathlower'] = availability['path'].apply(
            lambda path: path.lower()
        )
        availability = availability.merge(lidardatafiles,on='pathlower')
        availability.drop(
            columns = ['path','pathlower'],
            inplace = True
        )
        availability['lidar_file'] = availability['lidar_file'].apply(
            lambda fn: str(fn)
        )

        if hucs is not None:
            availability = gpd.sjoin(
                availability,
                hucs[['HUC','geometry']].to_crs(availability.crs),
                how = 'inner',
                op = 'intersects'
            )

        if new_lidar_availability_file is not None:
            print(new_lidar_availability_file)
            sys.stdout.flush()
            availability.to_file(str(new_lidar_availability_file))

        return(availability)
    
class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise(self.ee.with_traceback(self.tb))

def delete_file(filename):
    """
    deletes a file in all versions of Python
    """

    try:
        Path(str(filename)).unlink(missing_ok=True)
    ## TODO: is this the correct exception for missing `missing_ok` parameter?
    except FileNotFoundError:
        Path(str(filename)).unlink()
    except:
        pass

def skip_function_if_file_exists(function,filename,skip_existing=True):
    """
    wrapper to skip a particular step in a workflow if a file already exists
    """

    filename = Path(str(filename))
    if not (
        filename.is_file() and not
        skip_existing
    ):
        delete_file(filename)
        function

def _write_geodataframe(
    geodataframe,
    filename,
    reset_index = True,
    driver = 'ESRI Shapefile',
    drop_index = False
):

    if reset_index:
        geodataframe = geodataframe.reset_index(drop=drop_index)

    geodataframe.to_file(str(filename),driver=driver)

def write_geodataframe(
    shapefile,
    filename,
    skip_existing = True,
    reset_index = True,
    driver = 'ESRI Shapefile'
):
    """
    write geodataframe to filename or concrete path
    """

    skip_function_if_file_exists(
        _write_geodataframe(
            shapefile,
            filename,
            reset_index = reset_index,
            driver = driver
        ),
        filename,
        skip_existing = skip_existing
    )

def _write_roughness_table(
    roughness_table,
    filename,
    column = None,
    sort = True
):

    if column is None:
        if 'COMID' in roughness_table.columns.unique():
            roughness_table_comids = roughness_table['COMID'].unique()
        else:
            roughness_table_comids = roughness_table.index.unique()
    else:
        roughness_table_comids = roughness_table[column].unique()

    if sort:
        roughness_table_comids = np.sort(roughness_table_comids)
    
    with open(str(filename), 'w', newline='') as roughnesses_csv:
        writer = csv.writer(roughnesses_csv)
        writer.writerow(['COMID','StreamOrde','Roughness'])
        for comid in roughness_table_comids:
            writer.writerow([
                comid,
                roughness_table.loc[comid,'StreamOrde'],
                roughness_table.loc[comid,'Roughness']
            ])

def write_roughness_table(
    roughness_table,
    filename,
    skip_existing = True,
    column = None,
    sort = True
):
    """
    write Manning’s n roughness table to CSV filename or concrete path
    """

    skip_function_if_file_exists(
        _write_roughness_table(
            roughness_table,
            filename,
            column = column,
            sort = sort
        ),
        filename,
        skip_existing = skip_existing
    )

def get_flowlines_and_representative_points_by_huc(
    hucs,
    nhd_input
):
    """
    Attributes HUCs to flowlines based on each flowline's representative point

    Assumes National Hydrography Dataset (NHD) vector image inputs for flowlines
    HUCs are assumed to be derived from the Watershed Boundary Dataset (WBD),
    and must have a column labelled ``HUC*``,
    for example ``HUC12`` or ``HUC8``

    :param hucs: HUCs geodataframe,
        with column labelled ``HUC*``, such as ``HUC12`` or ``HUC8``
    :type hucs: gpd.GeoDataFrame
    :param nhd_input: NHD Flowline vector image filename or geodataframe
    :type nhd_input: Union[str, gpd.GeoDataFrame]
    :return: Tuple of
        flowlines geodataframe and flowlines representative points geodataframe,
        both assigned a HUC attribute
    :rtype: tuple
    """

    flowlines = get_nhd_by_shape(
        hucs,
        nhd_input,
        layer = 'Flowline'
    )

    flowline_representative_points = get_representative_points(
        flowlines,
        hucs
    )

    flowlines = clip_dataframe_by_attribute(
        flowlines,
        flowline_representative_points,
        attribute = 'HUC'
    )

    flowlines = set_roughness_by_streamorder(flowlines)

    return(flowlines,flowline_representative_points)

def get_catchments_by_huc(hucs,nhd_input,flowline_representative_points):
    """
    assigns HUCs to NHD catchments
    """

    catchments = get_nhd_by_shape(
        hucs,
        nhd_input,
        layer = 'Catchment',
        comid_column = 'FEATUREID',
        fix_invalid_geometries = True
    )

    catchments = clip_dataframe_by_attribute(
        catchments,
        flowline_representative_points,
        attribute = 'HUC'
    )

    return(catchments)

def get_hucs_from_catchments(catchments):
    """
    dissolves NHD catchments into HUC equivalents
    """

    hucs = catchments.dissolve(by='HUC')
    hucs.reset_index(inplace=True)
    hucs = reproject_to_utm_and_buffer(hucs)

    return(hucs)

def to_crs(crs,geodataframes):
    """
    reprojects multiples geodataframes simultaneously
    """

    for geodataframe in geodataframes:
        geodataframe.to_crs(crs,inplace=True)

def extend_lidar_index(lidar_index,raster,vrt_filename):
    ## Check each raster's resolution in this HUC

    #if any(np.float16(i) > 1. for i in var.res):
    #    out_path = os.path.join(subdirectory, "gt1m.err")
    #    Path(out_path).touch()
    #    print('WARNING: >1m raster input for HUC: '+str(hu))
    #    sys.stdout.flush()
    #else:

    index = lidar_index[
        lidar_index['lidar_file'] == raster.name
    ].index[0]

    lidar_index.loc[index,'minimum_resolution'] = min(raster.res)
    lidar_index.loc[index,'maximum_resolution'] = max(raster.res)
    lidar_index.loc[index,'x'] = raster.res[0]
    lidar_index.loc[index,'y'] = raster.res[1]
    lidar_index.loc[index,'vrt_filename'] = vrt_filename
    #lidar_index.loc[index,'lidar_file_handler'] = raster

    return(lidar_index)

def reproject_append(
    fp,
    dst_crs,
    memoryfile_dict,
    mosaic_metadata,
    subdirectory,
    memoryfile = False
):

    src = rasterio.open(fp)
    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds
    )
    out_meta = src.meta.copy()
    out_meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    if src.meta!=out_meta:

        ## Don't do an expensive reprojection if projection already correct
        if memoryfile:
            dst = memoryfile_dict[fp].open(**out_meta)
        else:
            dst = rasterio.open(
                memoryfile_dict[fp].name,
                'w+',
                **out_meta
            )
            memoryfile_dict[fp] = dst
        for i in range(1, src.count + 1):
            reproject(
                source = rasterio.band(src, i),
                destination = rasterio.band(dst, i),
                src_transform = src.transform,
                src_crs = src.crs,
                dst_transform = dst.transform,
                dst_crs = dst.crs,
                resampling = Resampling.nearest
            )
    else:
        dst = src
        if not memoryfile:
            memoryfile_dict[fp] = dst
    mosaic_metadata = append_metadata(mosaic_metadata,dst)

    return(mosaic_metadata)

def close_rasters(raster,delete_rasters=False):

    for raster in rasters:

        try:
            raster.close()
        except:
            pass

        if delete_rasters:
            delete_file(raster.name)

def write_error_file_and_print_message(filename,message):

    Path(str(filename)).touch()
    print(message)
    sys.stdout.flush()

def build_vrt(
    filenames,
    vrt_filename,
    lowest_resolution = False,
    **kwargs
):

    filenames = [str(filename) for filename in filenames]

    if lowest_resolution:
        resolution = 'lowest'
    else:
        resolution = 'highest'

    vrt_options = gdal.BuildVRTOptions(
        resampleAlg = 'near',
        #addAlpha = True,
        resolution = resolution,
        **kwargs
        #separate=True
    )

    vrt = gdal.BuildVRT(str(vrt_filename),filenames,options=vrt_options)
    vrt = None

def build_vrts_in_lidar_index(
    lidar_index,
    vrt_filename_template,
    lowest_resolution = False
):

    filenames = lidar_index['lidar_file'].to_list()

    for filename in filenames:

        vrt_filename = (
            os.path.splitext(str(vrt_filename_template))[0] +
            '-' +
            os.path.splitext(Path(str(filename)).name)[0] +
            '.vrt'
        )

        lidar_index.loc[
            lidar_index['lidar_file'] == filename,
            'vrt_filename'
        ] = vrt_filename

        build_vrt(filename,vrt_filename)

    return(lidar_index)

def reproject_raster(
    filename,
    reprojected_filename,
    raster_mask_filename = None,
    dst_crs = None
):

    if dst_crs is not None:
        dst_crs = str(dst_crs)

    if raster_mask_filename is not None:
        crop_to_mask_file = True
    else:
        crop_to_mask_file = False

    raster = gdal.Open(str(filename))

    warp = gdal.Warp(
        str(reprojected_filename),
        raster,
        dstSRS = dst_crs,
        cutlineDSName = raster_mask_filename,
        cropToCutline = crop_to_mask_file
    )
    warp = None

def isiterable(theElement):

    try:
        iterator = iter(theElement)
    except TypeError:
        return(False)
    else:
        return(True)

    return(True)

def reproject_rasters(
    filenames,
    reprojected_filenames,
    dst_crs = None
):

    if dst_crs is not None and not isiterable(dst_crs):
        dst_crs = [dst_crs] * len(reprojected_filenames)

    for filename,reprojected_filename,dst_crs_inner in zip(
        filenames,
        reprojected_filenames,
        dst_crs
    ):

        reproject_raster(
            str(filename),
            str(reprojected_filename),
            dst_crs = dst_crs_inner
        )

def timing(f):
    ## Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) 2011 Mike Lewis
    ## https://stackoverflow.com/a/5478448/16518080
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__,
            (time2-time1)*1000.0
        ))
        return(ret)
    return(wrap)

def try_except_for_huc(function,huc_id):

    ## TODO: test and implement this function
    try:

        result = function

    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_id))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_id))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:

        print('Result for HUC: '+str(huc_id))
        sys.stdout.flush()

    finally:

        print('Reached finally clause')
        sys.stdout.flush()

    return(result)

def count_lidar_projects_in_lidar_index(lidar_index_by_huc):

    lidar_index_by_project_grouped = lidar_index_by_huc.groupby('dirname')
    lidar_index_by_project = [
        lidar_index_by_project_grouped.get_group(group_name)
        for group_name
        in lidar_index_by_project_grouped.groups
    ]
    lidar_project_tile_counts = [
        [ project['dirname'].unique()[0] , project.shape[0] ]
        for project
        in lidar_index_by_project
    ]
    lidar_projects_with_counts = pd.DataFrame(
        lidar_project_tile_counts,
        columns = ['dirname','count']
    )
    lidar_projects_with_counts.sort_values(by=['count'])
    lidar_projects_with_info_tile = lidar_index_by_huc.groupby('dirname').first()[['lidar_file']].reset_index()
    lidar_projects_with_counts = lidar_projects_with_info_tile.merge(
        lidar_projects_with_counts,
        on = ['dirname']
    )
    lidar_projects_with_counts.sort_values(
        by = ['count'],
        ascending = False,
        inplace = True
    )

    huc_prefix = Path(str(lidar_index_by_huc['HUC'].unique()[0]))
    try:
        lidar_projects_with_counts['crs'] = lidar_projects_with_counts[
            'lidar_file'
        ].apply(
            lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection())
        )
    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_prefix))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_prefix))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:
        print('Result for HUC: '+str(huc_prefix))
        sys.stdout.flush()

    finally:
        print('Reached finally clause')
        sys.stdout.flush()

    lidar_projects_with_counts['epsg'] = lidar_projects_with_counts['crs'].apply(
        lambda crs: crs.to_epsg()
    )

    return(lidar_projects_with_counts)

def build_vrts(
    filenames_repeated,
    vrts_to_composite,
    **kwargs
):

    for filenames_inner,vrts_to_composite_inner in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        build_vrt(filenames_inner,vrts_to_composite_inner,**kwargs)

def _get_mosaic_and_output_raster(
    lidar_index_by_huc,
    huc,
    output_raster_filename,
    parent_temporary_directory,
):

    huc_prefix = Path(str(huc['HUC'].unique()[0]))

    temporary_directory = Path(str(parent_temporary_directory)).joinpath(
        huc_prefix
    )

    huc_prefix = str(huc_prefix)

    if not temporary_directory.is_dir():
        temporary_directory.mkdir(parents=True, exist_ok=True)

    filenames = lidar_index_by_huc['lidar_file'].to_list()

    lidar_projects_with_counts = count_lidar_projects_in_lidar_index(
        lidar_index_by_huc
    )

    different_epsgs = list(lidar_projects_with_counts['epsg'].unique())

    vrts_to_composite = []
    dst_crs_epsgs = []
    for epsg in different_epsgs:
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            huc_prefix +
            '-' +
            str(epsg) +
            '.vrt'
        ))
        dst_crs_epsgs.append('EPSG:' + str(epsg))

    filenames_repeated = [filenames] * len(vrts_to_composite)

    ## Build a new VRT for each different CRS found
    try_except_for_huc(
        build_vrts(
            filenames_repeated,
            vrts_to_composite,
            allowProjectionDifference = True
        ),
        huc_prefix
    )

    reprojected_vrts_filenames = [
        Path(str(vrt)).parent.joinpath(
            Path(
                os.path.splitext(str(Path(str(vrt)).name))[0] +
                '-reprojected.vrt',
            )
        )
        for vrt
        in vrts_to_composite
    ]

    ## Reproject VRTs to each different CRS for this study area
    try_except_for_huc(
        reproject_rasters(
            vrts_to_composite,
            reprojected_vrts_filenames,
            dst_crs = dst_crs_epsgs
        ),
        huc_prefix
    )

    same_projection_vrts_filenames = [
        Path(str(vrt)).parent.joinpath(
            Path(
                os.path.splitext(str(Path(str(vrt)).name))[0] +
                '-' +
                str(huc.crs.to_epsg()) +
                '.vrt'
            )
        )
        for vrt
        in reprojected_vrts_filenames
    ]

    ## Reproject each of these different CRSs VRTs to the same CRS VRTs
    try_except_for_huc(
        reproject_rasters(
            reprojected_vrts_filenames,
            same_projection_vrts_filenames,
            dst_crs = huc.crs
        ),
        huc_prefix
    )

    temporary_vrt_file = temporary_directory.joinpath(
        huc_prefix + '.vrt'
    )

    ## Build VRTs from reprojected VRTs
    try_except_for_huc(
        build_vrt(
            reprojected_vrts_filenames,
            temporary_vrt_file,
            allowProjectionDifference = True
        ),
        huc_prefix
    )

    temporary_huc_file = temporary_directory.joinpath(
        huc_prefix + '.geojson'
    )

    huc.to_file(temporary_huc_file)

    ## Reproject VRTs to the same CRS and output
    try_except_for_huc(
        reproject_raster(
            str(temporary_vrt_file),
            str(output_raster_filename),
            raster_mask_filename = str(temporary_huc_file),
            dst_crs = huc.crs
        ),
        huc_prefix
    )

def _get_mosaic_and_output_raster_dev(
#def reproject_lidar_tiles_and_build_vrt_by_huc(
    lidar_index_by_huc,
    huc,
    output_raster_filename,
    parent_temporary_directory,
):

    huc_prefix = Path(str(huc['HUC'].unique()[0]))

    temporary_directory = Path(str(parent_temporary_directory)).joinpath(
        huc_prefix
    )

    if not temporary_directory.is_dir():
        temporary_directory.mkdir(parents=True, exist_ok=True)

    filenames = lidar_index_by_huc['lidar_file'].to_list()

    lidar_projects_with_counts = count_lidar_projects_in_lidar_index(
        lidar_index_by_huc
    )

    different_epsgs = lidar_projects_with_counts['epsg'].unique().to_list()

#    filenamess = []
    vrts_to_composite = []
    for epsg in different_epsgs.to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(epsg) +
            '.vrt'
        ))

    filenames_repeated = [filenames] * len(vrts_to_composite)
    result = try_except_for_huc(function,huc_prefix)
    try:
        for filenames_inner,vrts_to_composite_inner in zip(
            filenames_repeated,
            vrts_to_composite
        ):
            build_vrt(filenames_inner,vrts_to_composite_inner)
    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_prefix))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_prefix))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:
        print('Result for HUC: '+str(huc_prefix))
        sys.stdout.flush()

    finally:
        print('Reached finally clause')
        sys.stdout.flush()

    reprojected_vrts_filenames = [
        Path(str(vrt)).parent.joinpath(
            Path(
                os.path.splitext(str(Path(str(vrt)).name))[0] +
                '-reprojected.vrt',
            )
        )
        for vrt
        in vrts_to_composite
    ]

    try:
        for vrt,reprojected_vrt,epsg in zip(
            vrts_to_composite,
            reprojected_vrts_filenames,
            different_epsgs
        ):
            reproject_raster(vrt,reprojected_vrt,dst_crs='EPSG:'+str(epsg))
    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_prefix))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_prefix))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:
        print('Result for HUC: '+str(huc_prefix))
        sys.stdout.flush()

    finally:
        print('Reached finally clause')
        sys.stdout.flush()

    temporary_vrt_file = temporary_directory.joinpath(
        str(huc_prefix) + '.vrt'
    )

    try:
        build_vrt(reprojected_vrts_filenames,temporary_vrt_file)
    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_prefix))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_prefix))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:
        print('Result for HUC: '+str(huc_prefix))
        sys.stdout.flush()

    finally:
        print('Reached finally clause')
        sys.stdout.flush()

    temporary_huc_file = temporary_directory.joinpath(
        str(huc_prefix) + '.geojson'
    )

    huc.to_file(temporary_huc_file)

    try:
        reproject_raster(
            str(temporary_vrt_file),
            str(output_raster_filename),
            raster_mask_filename = str(temporary_huc_file)
        )
    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_prefix))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_prefix))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        raise

    else:
        print('Result for HUC: '+str(huc_prefix))
        sys.stdout.flush()

    finally:
        print('Reached finally clause')
        sys.stdout.flush()

def get_mosaic_dev(lidar_index,vrt_options,directory):

    filenames = lidar_index['lidar_file'].to_list()

    for filename in filenames:

        name = os.path.split(filename)[1]
        vrt_filename = os.path.join(directory,'aligned-{}'.format(name))

        with rasterio.open(filename) as src:

            with WarpedVRT(src,**vrt_options) as vrt:

                try:

                    data = vrt.read()

                except MemoryError:

                    for _, window in vrt.block_windows():
                        data = vrt.read(window=window)

                rio_shutil.copy(vrt,vrt_filename,driver='VRT')

                lidar_index = extend_lidar_index(lidar_index,src,vrt_filename)

    return(lidar_index)

def get_mosaic(
    lidar_index,
    huc_id,
    dst_crs,
    subdirectory,
    memoryfile = False,
    log = None,
    temporary_dir = None,
    temporary_dir_for_each_huc = False,
    lowest_resolution = True
):
    ## Get mosaic of DEMs for each HUC

    ## Reproject the mosaic to DEM tiles pertaining to each HUC
    dem_fps = list(lidar_index['lidar_file'])
    mosaic_metadata = pd.DataFrame({
        'lidar_file_handler' : [],
        'minimum_resolution' : [],
        'maximum_resolution' : [],
        'x' : [],
        'y' : []
    })
    memoryfile_dict = {}

    if memoryfile:
        for fp in dem_fps:
            memoryfile_dict[fp] = MemoryFile()
    else:
        if temporary_dir_for_each_huc:
            temporary_dir = Path(os.path.join(
                tempdir,
                str(huc_id)
            )).mkdir(parents=True, exist_ok=True)
        for fp in dem_fps:
            memoryfile_dict[fp] = tempfile.NamedTemporaryFile(
                dir = temporary_dir
            )
#            memoryfile_dict[fp] = os.path.join(
#                tempdir,
#                str(huc_id),
#                next(tempfile._get_candidate_names())
#            )

    for fp in dem_fps:
        try:
            print(fp)
            sys.stdout.flush()
            mosaic_metadata = reproject_append(
                fp,
                dst_crs,
                memoryfile_dict,
                mosaic_metadata,
                subdirectory,
                memoryfile = memoryfile
            )
            print("Finished getting mosaic_metadata")
            sys.stdout.flush()
        except Exception as err:
            print(
                '[EXCEPTION] Exception on HUC: ' +
                str(huc_id)
            )
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(
                exc_tb.tb_frame.f_code.co_filename
            )[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.stdout.flush()
            if log:
                logging.debug(
                    '[EXCEPTION] on HUC ' +
                    str(huc_id)
                )
            traceback.print_tb(err.__traceback__)
            pass

    if mosaic_metadata.shape[0] == 0:

        filename = os.path.join(subdirectory, "allGT1m.err")
        message = (
            'WARNING: Found no <=1m raster input data for HUC: ' +
            str(huc_id)
        )
        write_error_file_and_print_message(filename,message)

        break_hu = True
        mosaic_tuple = ()
        return(break_hu,mosaic_tuple)

    else:

        if lowest_resolution:
            mosaic_metadata.sort_values(
                by = ['minimum_resolution','maximum_resolution'],
                inplace = True
            )
        else:
            mosaic_metadata.sort_values(
                by = ['maximum_resolution','minimum_resolution'],
                inplace = True
            )

        mosaic, out_trans = merge(
            mosaic_metadata['lidar_file_handler'].to_list(),
            res = (mosaic_metadata['x'].max(),mosaic_metadata['y'].max())
        )

        close_rasters(mosaic_metadata['lidar_file_handler'])

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": 'GTiff',
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": dst_crs
        })

        close_rasters(memoryfile_dict.values(),delete_rasters=True)

        break_hu = False
        mosaic_tuple = (mosaic,out_meta)
        return(break_hu,mosaic_tuple)

def mask_raster_dev(raster,meta,mask_geometry,memoryfile=False):

    if memoryfile:
        function = raster
        parameters = [*meta]
    else:
        function = rasterio
        parameters = [raster, 'w+', *meta]

    with function.open(*parameters) as dataset:
        dataset.write(mosaic)
    with function.open(*parameters) as dataset:
        out_image, out_trans = rasterio.mask.mask(
            dataset,
            mask_geometry,
            crop = True
        )

    return(out_image,out_trans)

def output_raster(
    hu_buff,
    mosaic,
    out_meta,
    raster_filename,
    tempdir,
    huc_id,
    memoryfile = False
):
    ## Crop and output the mosaic to the buffered catchments of each HUC

    hu_buff_geom = hu_buff['geometry'].to_list()
    if memoryfile:
        with MemoryFile() as memoryfile_handler:
            out_image, out_trans = mask_raster(
                memoryfile_handler,
                out_meta,
                hu_buff_geom,
                memoryfile = True
             )
#            with memfile.open(**out_meta) as dataset:
#                dataset.write(mosaic)
#            with memfile.open(**out_meta) as dataset:
#                out_image, out_trans = rasterio.mask.mask(
#                    dataset,
#                    hu_buff_geom,
#                    crop = True
#                )
    else:
        memoryfilename = os.path.join(
            tempdir,
            str(huc_id),
            next(tempfile._get_candidate_names())
        )
        out_image, out_trans = mask_raster(memoryfilename,out_meta,hu_buff_geom)
#        with rasterio.open(memfile,'w+',**out_meta) as dataset:
#            dataset.write(mosaic)
#        with rasterio.open(memfile,'w+',**out_meta) as dataset:
#            out_image, out_trans = rasterio.mask.mask(
#                dataset,
#                hu_buff_geom,
#                crop = True
#            )

    out_meta.update({
        "height" : out_image.shape[1],
        "width" : out_image.shape[2],
        "transform" : out_trans
    })

    with rasterio.open(raster_filename,"w",**out_meta) as dst:
        dst.write(out_image)

    return(out_image)

def _get_mosaic_and_output_raster_original(
    lidar_index,
    huc_id,
    dst_crs,
    subdirectory,
    return_dict,
    raster_filename
):

    break_hu, mosaic_tuple = get_mosaic(
        lidar_index,
        huc_id,
        dst_crs,
        subdirectory
    )

    if break_hu!=True:

        ## TODO: Use for selection but not cropping
        raster = lidar_index.dissolve(by=['HUC'])
        raster.reset_index(inplace=True)
        raster = _drop_index_columns(raster)
        raster.geometry = raster.buffer(.8).buffer(-.8)

        #with rasterio.Env():
        #    results = ({
        #        'properties': {
        #            'Elevation': v
        #        },
        #        'geometry': s
        #    }
        #    for i, (s, v) in enumerate(shapes(
        #        (mosaic_tuple[0]==mosaic_tuple[1]['nodata']).astype(np.int16),
        #        mask=mosaic_tuple[0]!=mosaic_tuple[1]['nodata'],
        #        transform=mosaic_tuple[1]['transform']
        #    )))
        #geoms = list(results)
        #raster = gpd.GeoDataFrame.from_features(geoms,crs=mosaic_tuple[1]['crs'])

        hu_buff = hucs.to_crs(mosaic_tuple[1]['crs'])
        hu_buff.reset_index(inplace=True)
        hu_buff = _drop_index_columns(hu_buff)

        filename = os.path.join(subdirectory, 'hu_buff.geojson')
        write_geodataframe(hu_buff,filename,drop_index=True)

        filename = os.path.join(subdirectory, 'raster.geojson')
        write_geodataframe(raster,filename,drop_index=True)

        if len(gpd.sjoin(
            hu_buff,
            raster.to_crs(hu_buff.crs),
            op = 'within',
            how = 'inner'
        ).index) == 0:

            filename = os.path.join(
                subdirectory,
                "rasterDataDoesNotEnclose.err"
            )
            message = (
                'WARNING: <=1m raster input data does not enclose HUC: ' +
                str(huc_id)
            )
            write_error_file_and_print_message(filename,message)

        else:

            out_image = output_raster(
                hu_buff,
                mosaic_tuple[0],
                mosaic_tuple[1],
                raster_filename
            )
            return_dict[huc_id] = out_image

def get_mosaic_and_output_raster(
    lidar_index_by_huc,
    huc,
    output_raster_filename,
    parent_temporary_directory,
    skip_existing = True
):

    skip_function_if_file_exists(
        _get_mosaic_and_output_raster(
            lidar_index_by_huc,
            huc,
            output_raster_filename,
            parent_temporary_directory
        ),
        output_raster_filename,
        skip_existing = skip_existing
    )

def make_directories_and_error_files(
    output_parent_directory,
    output_filename_prefix,
    huc_id
):

    output_directory = os.path.join(
        directory,
        str(output_filename_prefix) + '-' + str(huc_id)
    )
#    print(output_directory)
#    sys.stdout.flush()
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    notimeleft_filename = os.path.join(
        output_directory,
        "jobNoTimeLeftWhileProcessing.err"
    )
    Path(notimeleft_filename).touch()

    greaterthan1m_filename = os.path.join(output_directory, "allGT1m.err")
    greaterthan1m_path = Path(greaterthan1m_filename)

    rasternotenclosed_filename = os.path.join(
        output_directory,
        "rasterDataDoesNotEnclose.err"
    )
    rasternotenclosed_path = Path(rasternotenclosed_filename)

    return(
        output_directory,
        notimeleft_filename,
        greaterthan1m_path,
        rasternotenclosed_path
    )

def make_parent_directories(filenames):
    for filename in filenames:
        Path(str(filename)).parent.mkdir(parents=True, exist_ok=True)

def prepare_and_output_geoflood_gis_inputs_by_huc(
    shape_input,
    hucs_input,
    nhd_input,
    lidar_availability_input,
    lidar_parent_directory,
    output_parent_directory,
    temporary_parent_directory,
    select_utm = None,
    new_lidar_availability_file = None,
    correct_lidar_availability_input = True,
    pickle_file = None,
    **kwargs
):
    ## TODO: include parameters to output and restart from intermediate products

    (
        hucs,
        flowlines,
        catchments,
        lidar_index
    ) = prepare_geoflood_gis_inputs(
        shape_input,
        hucs_input,
        nhd_input,
        lidar_availability_input,
        lidar_parent_directory,
        select_utm = select_utm,
        new_lidar_availability_file = None,
        correct_lidar_availability_input = True
    )

    huc_ids = get_merged_column(
        'HUC',
        [lidar_index,flowlines,catchments,hucs]
    )

    [
        hucs_by_huc,
        flowlines_by_huc,
        catchments_by_huc,
        lidar_index_by_huc
    ] = list_geodataframes_grouped_by_column(
        [hucs,flowlines,catchments,lidar_index],
        huc_ids,
        column = 'HUC'
    )

    ## TODO: Consider replacing with `make_parent_directories` elsewhere here
    output_directories_by_huc = []
    temporary_directories_by_huc = []
    for huc_id in huc_ids:

        output_directory = Path(str(output_parent_directory)).joinpath(
            Path(str(huc_id))
        )
        output_directory.mkdir(parents=True,exist_ok=True)
        output_directories_by_huc.append(
            output_directory
        )

        temporary_directory = Path(str(output_parent_directory)).joinpath(
            Path(str(huc_id))
        )
        temporary_directory.mkdir(parents=True,exist_ok=True)
        temporary_directories_by_huc.append(
            temporary_directory
        )

    if pickle_file is not None:
        pickle_multiple_objects(str(pickle_file),
            [
                output_directories_by_huc,
                flowlines_by_huc,
                catchments_by_huc,
                lidar_index_by_huc,
                hucs_by_huc,
                temporary_directories_by_huc
            ]
        )

    output_geoflood_gis_inputs_by_huc(
        output_directories_by_huc,
        flowlines_by_huc,
        catchments_by_huc,
        lidar_index_by_huc,
        hucs_by_huc,
        temporary_directories_by_huc,
        **kwargs
    )

def output_geoflood_gis_inputs_by_huc(
    output_directories_by_huc,
    flowlines_by_huc,
    catchments_by_huc,
    lidar_index_by_huc,
    hucs_by_huc,
    temporary_directories_by_huc,
    **kwargs
):

    for (
        output_directory,
        flowlines,
        catchments,
        lidar_index,
        hucs,
        temporary_directory
    ) in zip(
        output_directories_by_huc,
        flowlines_by_huc,
        catchments_by_huc,
        lidar_index_by_huc,
        hucs_by_huc,
        temporary_directories_by_huc
    ):

        output_geoflood_gis_inputs(
            output_directory,
            flowlines,
            catchments,
            lidar_index,
            hucs,
            temporary_directory,
            **kwargs
        )

def output_geoflood_gis_inputs(
    output_directory,
    flowlines,
    catchments,
    lidar_index,
    hucs,
    temporary_directory,
    exclude_raster_input = False,
    exclude_vector_inputs = False
):
    ## TODO: this is implicitly for a single HUC right now

    if not exclude_vector_inputs:

        filename = os.path.join(output_directory, 'Flowline.shp')
        write_geodataframe(flowlines,filename)
    
        filename = os.path.join(output_directory, 'Roughness.csv')
        write_roughness_table(flowlines,filename)
    
        filename = os.path.join(output_directory, 'Catchment.shp')
        write_geodataframe(catchments,filename)

    if not exclude_raster_input:

        filename = os.path.join(output_directory, 'Elevation.tif')
        get_mosaic_and_output_raster(
            lidar_index,
            hucs,
            filename,
            temporary_directory
        )

#@profile
def output_files(arguments,return_dict):
#def output(flow_key,flowshu12shape,catchshu12shape,hu12catchs,lidar_index,args,prefix,dst_crs,mem_estimates):
    ## Output catchments, flowlines, roughnesses, and rasters

    (
        huc_ids,
        flowlines,
        catchments,
        hucs,
        lidar_index,
        args,
        output_filename_prefix,
        dst_crs,
        tempdir
    ) = arguments

    try:

        (
            output_directory,
            path_notime,
            file_gt1m,
            file_enclose
        ) = make_directories_and_error_files(directory,output_filename_prefix)

        if not (file_gt1m.is_file() or file_enclose.is_file()):

            output_geoflood_gis_inputs(
                output_directory,
                flowlines,
                catchments,
                lidar_index,
                hucs,
                temporary_directory
            )

            filename = os.path.join(subdirectory, 'Flowline.shp')
            write_geodataframe(flowlines,filename)

            filename = os.path.join(subdirectory, 'Roughness.csv')
            write_roughness_table(flowlines,filename)

            filename = os.path.join(subdirectory, 'Catchment.shp')
            write_geodataframe(catchments,filename)

            filename = os.path.join(subdirectory, 'Elevation.tif')
            get_mosaic_and_output_raster(
                lidar_index,
                hucs,
                filename,
                tempdir
            )

        Path(path_notime).unlink()

    except OSError as e:
        Path(path_notime).unlink()
        out_path = os.path.join(subdirectory, "OS.err")
        Path(out_path).touch()
        with open(out_path, 'w') as f:
            f.write(str(e))
        print('[ERROR] OSError on HUC: '+str(huc_id))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        if args.log:
            logging.debug('[ERROR] OSError on HUC '+str(huc_id))

    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(huc_id))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        if args.log:
            logging.debug('[EXCEPTION] on HUC '+str(huc_id))
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(huc_id))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        if args.log:
            logging.debug('[EXCEPTION] on HUC '+str(huc_id))
        raise

    else:
        print('Result for HUC: '+str(huc_id))
        sys.stdout.flush()
        if args.log:
            logging.debug('[EXCEPTION] on HUC '+str(huc_id))

    finally:
        print('Reached finally clause')
        sys.stdout.flush()
        if args.log:
            logging.debug('[EXCEPTION] on HUC '+str(huc_id))

def crop_and_convert_large_vector_file(
    filename,
    converted_filename,
    meta = None,
    bounding_box = None
):

    with fiona.open(filename) as src:

        if meta is None:
            meta = src.meta

        with fiona.open(converted_filename, 'w', **meta) as dst:

            for f in src.filter(bbox=bounding_box):

                dst.write(f)

class TaskProcessor(Thread):
    """
    Processor class which monitors memory usage for running tasks (processes).
    Suspends execution for tasks surpassing `max_b` and completes them one
    by one, after behaving tasks have finished.
    """

    def __init__(
        self,
        n_cores,
        max_b,
        tempdir,
        percent_free_mem,
        percent_free_disk,
        return_dict,
        tasks
    ):

        super().__init__()

        self.n_cores = n_cores

        self.max_b = max_b
        self.percent_free_mem = percent_free_mem / 100.
        self.max_free_mem = self.max_b * self.percent_free_mem

        self.tempdir = tempdir
        self.max_tmp = (
            shutil.disk_usage(self.tempdir)[0] -
            shutil.disk_usage(self.tempdir)[1]
        )
        self.percent_free_disk = percent_free_disk / 100.
        self.max_free_disk = self.max_tmp * self.percent_free_disk

        self.return_dict = return_dict

        self.tasks = deque(tasks)
        self._running_tasks = []

        self.log = pd.DataFrame({
            'update/monitoring' : [],
            'n_running_tasks' : [],
            'n_cores' : [],
            'n_tasks' : [],
            'available_memory' : [],
            'max_memory' : [],
            'available_disk' : [],
            'max_disk' : [],
            'process' : [],
            'status' : []
        })

    def run(self):
        """Main-function in new thread."""
        self._update_running_tasks()
        self._monitor_running_tasks()

    def _update_running_tasks(self):
        """Start new tasks if we have less running tasks than cores."""
        gc.collect()
        while (
            len(self._running_tasks) < self.n_cores and
            len(self.tasks) > 0
        ):
            available_memory = psutil.virtual_memory().available
            available_disk = shutil.disk_usage(self.tempdir)[2]
            if (
                available_memory > self.max_free_mem and
                available_disk > self.max_free_disk
            ):
                n_running_tasks = len(self._running_tasks)
                n_tasks = len(self.tasks)
                #print('ENTERED running')
                #print(
                #    'self._running_tasks: ',
                #    [task[0] for task in self._running_tasks]
                #)
                #print('self.tasks: ',[task[0] for task in self.tasks])
                p = self.tasks.popleft()
                gc.collect()
                p[0].start()
                # for further process-management we here just need the
                # psutil.Process wrapper
                self._running_tasks.append((
                    psutil.Process(pid=p[0].pid),
                    p[1]
                ))
                status = 'started'
                log = pd.DataFrame({
                    'update/monitoring' : ['update'],
                    'n_running_tasks' : [n_running_tasks],
                    'n_cores' : [self.n_cores],
                    'n_tasks' : [n_tasks],
                    'available_memory' : [available_memory],
                    'max_memory' : [self.max_free_mem],
                    'available_disk' : [available_disk],
                    'max_disk' : [self.max_free_disk],
                    'process' : [self._running_tasks[-1][0]],
                    'status' : [status]
                })
                self.log.append(log,ignore_index=True)
                print(log.iloc[0])
                #print(f'Started process: {self._running_tasks[-1][0]}')
                sys.stdout.flush()
            else:
                break

    def _monitor_running_tasks(self):
        """
        Monitor running tasks. Replace completed tasks and suspend tasks
        which exceed the memory threshold `self.max_b`.
        """

        # loop while we have running or non-started tasks
        while self._running_tasks or self.tasks:

            #print('ENTERED monitoring')
            multiprocessing.active_children() # Join all finished processes
            # Without it, p.is_running() below on Unix would not return
            # `False` for finished processes.
            self._update_running_tasks()
            actual_tasks = self._running_tasks.copy()

            for p in actual_tasks:
                #print('monitoring running task: ',p[0])
                n_running_tasks = len(self._running_tasks)
                n_tasks = len(self.tasks)
                available_memory = psutil.virtual_memory().available
                available_disk = shutil.disk_usage(self.tempdir)[2]
                p0 = p[0]
                status = 'running'
                if not (
                    p[0].is_running() and
                    p[0].status() != psutil.STATUS_ZOMBIE
                ):  # process has finished
                    self._running_tasks.remove(p)
                    shutil.rmtree(Path(os.path.join(
                        self.tempdir,
                        str(p[1][0])
                    )),ignore_errors=True)
                    status = 'finished'
                    #print(f'Finished process: {p[0]}')
                elif not (
                    available_memory > self.max_free_mem and
                    available_disk > self.max_free_disk
                ):
                    p[0].terminate()
                    self._running_tasks.remove(p)
                    shutil.rmtree(Path(os.path.join(
                        self.tempdir,
                        str(p[1][0])
                    )),ignore_errors=True)
                    p = (
                        multiprocessing.Process(
                            target = output_files,
                            args = (p[1],self.return_dict),
                            name = uuid.uuid4().hex
                        ),
                        p[1]
                    )
                    self.tasks.append(p)
                    status = 'suspended'
                    #print(f'Suspended process: {p[0]}')
#                    except OSError as err:
#                        exc_type, exc_obj, exc_tb = sys.exc_info()
#                        fname = os.path.split(
#                            exc_tb.tb_frame.f_code.co_filename
#                        )[1]
#                        print(exc_type, fname, exc_tb.tb_lineno)
#                        sys.stdout.flush()
#                        traceback.print_tb(err.__traceback__)
#                        pass
                log = pd.DataFrame({
                    'update/monitoring' : ['monitoring'],
                    'n_running_tasks' : [n_running_tasks],
                    'n_cores' : [self.n_cores],
                    'n_tasks' : [n_tasks],
                    'available_memory' : [available_memory],
                    'max_memory' : [self.max_free_mem],
                    'available_disk' : [available_disk],
                    'max_disk' : [self.max_free_disk],
                    'process' : [p0],
                    'status' : [status]
                })
                self.log.append(log,ignore_index=True)
                print(log.iloc[0])
                sys.stdout.flush()
                        
            time.sleep(.005)

def get_merged_column(column,dataframes,sort=True):
    """
    returns the mutual elements of an identically names column in multiple dataframes
    """

    ## Ensure lists share the same HUCs
#    mutual_row_values = reduce(
#        lambda left,right: pd.merge(left,right,on=[column],how='outer'),
#        dataframes
#    )
    mutual_row_values = pd.concat(
        [dataframe[column] for dataframe in dataframes]
    ).dropna().unique()
    if sort:
        mutual_row_values = np.sort(mutual_row_values)

    return(mutual_row_values)

def get_mutual_column_elements_of_dataframes(dataframes,column='HUC'):

    setlist = []
    for dataframe in dataframes:
        setlist.append(set(dataframe[column].unique()))

    dataframes = list(set.intersection(*setlist))

    return(dataframes)

def get_dataframes_with_elements_in_column(dataframes,elements,column='HUC'):

    new_dataframes = []

    for dataframe in dataframes:
        new_dataframes.append(dataframe[dataframe[column].isin(elements)])

    return(new_dataframes)

def get_lidar_intermediate_vectors(
    shape_input,
    hucs_input,
    nhd_input,
    lidar_availability_input = None,
    lidar_parent_directory = None,
    new_lidar_availability_file = None,
    reproject = True,
    select_utm = None
):

    flowlines, catchments = get_flowlines_and_catchments_by_shape(
        shape_input,
        hucs_input,
        nhd_input,
        select_utm = select_utm
    )

    hucs = get_hucs_from_catchments(catchments)

    if (
        lidar_availability_input is not None and
        lidar_parent_directory is not None
    ):
        #lidar_index = index_lidar_files_dev(hucs)
        lidar_index_obj = LidarIndex()
        lidar_index = lidar_index_obj.index_lidar_files(
            lidar_availability_input,
            lidar_parent_directory,
            hucs = hucs
        )
    elif (
        lidar_availability_input is not None and
        lidar_parent_directory is None
    ):
        print('Need Lidar parent directory as well')
        sys.stdout.flush()
        return(ExceptionWrapper(e))
    elif (
        lidar_availability_input is None and
        lidar_parent_directory is not None
    ):
        print('Need Lidar availability file as well')
        sys.stdout.flush()
        return(ExceptionWrapper(e))

    if new_lidar_availability_file is not None:
        lidar_index.to_file(str(new_lidar_availability_file))

    return(flowlines,catchments,hucs,lidar_index)

def get_flowlines_and_catchments_by_shape(
    shape_input,
    hucs_input,
    nhd_input,
    reproject = True,
    select_utm = None
):

    hucs = get_hucs_by_shape(shape_input,hucs_input,select_utm=select_utm)

    (
        flowlines,
        flowline_representative_points
    ) = get_flowlines_and_representative_points_by_huc(hucs,nhd_input)

    catchments = get_catchments_by_huc(
        hucs,
        nhd_input,
        flowline_representative_points
    )

    flowlines = index_dataframe_by_dataframe(
        flowlines,
        catchments
    )

    if reproject:
        to_crs(catchments.crs,[flowlines,catchments])

    return(flowlines,catchments)

def sort_values(column,geodataframes):

    for geodataframe in geodataframes:
        geodataframe.sort_values(by=column,inplace=True)

def pickle_multiple_objects(pickle_file,objects):

    with open(pickle_file, 'wb') as output:
        pickle.dump(
            objects,
            output,
            pickle.HIGHEST_PROTOCOL
        )

def unpickle_multiple_objects(pickle_file):

    with open(pickle_file, 'rb') as input:
        objects = pickle.load(input)

    return(objects)

def prepare_geoflood_gis_inputs(
    shape_input,
    hucs_input,
    nhd_input,
    lidar_availability_input,
    lidar_parent_directory,
    select_utm = None,
    new_lidar_availability_file = None,
    correct_lidar_availability_input = True
):
    ## TODO: redundant method: merge with above `get_lidar_intermediate_vectors`

    hucs = get_hucs_by_shape(shape_input,hucs_input,select_utm=select_utm)

    (
        flowlines,
        flowline_representative_points
    ) = get_flowlines_and_representative_points_by_huc(hucs,nhd_input)

    catchments = get_catchments_by_huc(
        hucs,
        nhd_input,
        flowline_representative_points
    )

    flowlines = index_dataframe_by_dataframe(
        flowlines,
        catchments
    )

    hucs = get_hucs_from_catchments(catchments)

    if correct_lidar_availability_input:
        #lidar_index = index_lidar_files_dev(hucs)
        lidar_index_obj = LidarIndex()
        lidar_index = lidar_index_obj.index_lidar_files(
            lidar_availability_input,
            lidar_parent_directory,
            hucs = hucs,
            new_lidar_availability_file = new_lidar_availability_file
        )
    else:
        lidar_index = read_file_or_gdf(lidar_availability_input)

    to_crs(hucs.crs,[flowlines,catchments,lidar_index])

    return(hucs,flowlines,catchments,lidar_index)

def list_geodataframes_grouped_by_column(
    geodataframes,
    column_elements_subset,
    column = 'HUC'
):
    ## TODO: Sorting built-in here. Above functions treat sorting as an option

    lists_of_geodataframes = []

    for geodataframe in geodataframes:

        lists_of_geodataframes.append(list(dict(tuple(
            geodataframe[
                geodataframe[column].isin(column_elements_subset)
            ].sort_values(column).groupby(column)
        )).values()))

    return(lists_of_geodataframes)

def sort_lists_of_geodataframes_by_index(geodataframes,index):

    sorted_geodataframes = []

    for geodataframe in geodataframes:
        sorted_geodataframe.append([geodataframe[i] for i in index])

    return(sorted_geodataframes)

def sort_and_crop_lists_of_geodataframes_by_filesize(
    geodataframes,
    geoseries_of_filenames
):
    ## TODO: this whole function is ill-conceived:
    ##     filesize estimates should be folded into
    ##         LidarIndex() and FathomIndex() classes above, not done here
    ##     this sorting can and should happen before dividing geodataframes into
    ##         lists by HUC, not after

    ## TODO: replace `lidar_index[i]['lidar_file']`
    ##     with `geoseries_of_filenames`
    mem_estimates = {}
    for i in range(len(lidar_index)):
        mem_estimates[i] = lidar_index[i]['lidar_file'].apply(
            lambda x: Path(x).stat().st_size
        ).sum()

    mem_estimates = {
        k : v
        for k, v
        in sorted(mem_estimates.items(), key=lambda item: item[1])
    }
    gc.collect()
    mem_estimates = {
        k : v
        for k, v
        in mem_estimates.items()
        if v < psutil.virtual_memory().total
    }

    (
        huc_ids,
        hucs,
        flowlines,
        catchments,
        lidar_index
    ) = sort_geodataframes_by_index(
        [
            huc_ids,
            hucs,
            flowlines,
            catchments,
            lidar_index
        ],
        mem_estimates.keys()
    )
    ## TODO: test function replacement and then delete comments below
#    huc_ids = [huc_ids[i] for i in mem_estimates.keys()]
#    flowlines = [flowlines[i] for i in mem_estimates.keys()]
#    catchments = [catchments[i] for i in mem_estimates.keys()]
#    hucs = [hucs[i] for i in mem_estimates.keys()]
#    lidar_index = [lidar_index[i] for i in mem_estimates.keys()]

    return()

def main():

    MAX_B = psutil.virtual_memory().total - psutil.virtual_memory().used

    oldgdal_data = os.environ['GDAL_DATA']
    os.environ['GDAL_DATA'] = os.path.join(fiona.__path__[0],'gdal_data')

    global args

    args = argparser()

    ## TODO: Also a shared file, potentially causing deadlock
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG)

    no_restart_file = False
    if args.restart:
        my_file = Path(args.restart)
        if my_file.is_file():
            with open(args.restart, 'rb') as input:
                (
                    huc_ids,
                    flowlines,
                    catchments,
                    hucs,
                    lidar_index
                ) = pickle.load(input)
        else:
            no_restart_file = True

    if not args.restart or no_restart_file:

        (
            hucs,
            flowlines,
            catchments,
            lidar_index
        ) = prepare_geoflood_gis_inputs(
            args.shapefile,
            args.hucs,
            args.nhd,
            args.lidar_availability,
            args.lidar_parent_directory,
            select_utm = select_utm
        )
        
        huc_ids = get_merged_column(
            'HUC',
            [lidar_index,flowlines,catchments,hucs]
        )
        ## Ensure lists share the same HUCs
        ## TODO: test function replacement and delete these comments
#        huc_ids = np.sort(list(
#            set(lidar_index['HUC']).intersection(flowlines['HUC'])
#        ))
#        huc_ids = np.sort(list(
#            set(huc_ids).intersection(catchments['HUC'])
#        ))
#        huc_ids = np.sort(list(
#            set(huc_ids).intersection(hucs['HUC'])
#        ))
    
#        hucs.drop(
#            columns = ['index_left','index_right'],
#            errors = 'ignore',
#            inplace = True
#        )
        _drop_index_columns_inplace([hucs,flowlines,catchments,lidar_index])

        ## Divide into lists per HUC
        [
            hucs,
            flowlines,
            catchments,
            lidar_index
        ] = list_geodataframes_grouped_by_column(
            [hucs,flowlines,catchments,lidar_index],
            huc_ids,
            column = 'HUC'
        )
        ## TODO: test function replacement and delete these comments
#        flowlines = list(dict(tuple(
#            flowlines[
#                flowlines['HUC'].isin(huc_ids)
#            ].sort_values('HUC').groupby('HUC')
#        )).values())
#        catchments = list(dict(tuple(
#            catchments[
#                catchments['HUC'].isin(huc_ids)
#            ].sort_values('HUC').groupby('HUC')
#        )).values())
#        hucs = list(dict(tuple(
#            hucs[
#                hucs['HUC'].isin(huc_ids)
#            ].sort_index().groupby('HUC')
#        )).values())
#        lidar_index = list(dict(tuple(
#            lidar_index[
#                lidar_index['HUC'].isin(huc_ids)
#            ].sort_values('HUC').groupby('HUC')
#        )).values())
    
        ## Sort lists by estimated memory usage
        ## TODO: test function replacement and delete these comments
        mem_estimates = {}
        for i in range(len(lidar_index)):
            mem_estimates[i] = lidar_index[i]['lidar_file'].apply(
                lambda x: Path(x).stat().st_size
            ).sum()
        mem_estimates = {
            k : v
            for k, v
            in sorted(mem_estimates.items(), key=lambda item: item[1])
        }
        gc.collect()
        mem_estimates = {
            k : v
            for k, v
            in mem_estimates.items()
            if v < psutil.virtual_memory().total
        }
        huc_ids = [huc_ids[i] for i in mem_estimates.keys()]
        flowlines = [flowlines[i] for i in mem_estimates.keys()]
        catchments = [catchments[i] for i in mem_estimates.keys()]
        hucs = [hucs[i] for i in mem_estimates.keys()]
        lidar_index = [lidar_index[i] for i in mem_estimates.keys()]

    if args.restart and no_restart_file:
        with open(args.restart, 'wb') as output:
            pickle.dump(
                [
                    huc_ids,
                    flowlines,
                    catchments,
                    hucs,
                    lidar_index
                ],
                output,
                pickle.HIGHEST_PROTOCOL
            )

    start_time = time.time()

    output_prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
    remove_keys = []

    if (
        not args.overwrite or not (
            args.overwrite_flowlines and
            args.overwrite_catchments and
            args.overwrite_roughnesses and
            args.overwrite_rasters
        )
    ):
        for huc in huc_ids:
            subdirectory = os.path.join(args.directory, output_prefix+'-'+str(huc))
            if (
                Path(os.path.join(subdirectory,'Elevation.tif')).is_file() and
                Path(os.path.join(subdirectory,'Catchments.shp')).is_file() and
                Path(os.path.join(subdirectory,'Flowlines.shp')).is_file() and
                Path(os.path.join(subdirectory,'Roughness.csv')).is_file()
            ) or Path(os.path.join(
                subdirectory,
                'rasterDataDoesNotEnclose.err'
            )).is_file():
                remove_keys.append(huc)

    remove_keys_idcs = [huc_ids.index(key) for key in remove_keys]
    flowlines = [
        flowlines[key]
        for key
        in range(len(huc_ids))
        if key
        not in remove_keys_idcs
    ]
    catchments = [
        catchments[key]
        for key
        in range(len(huc_ids))
        if key
        not in remove_keys_idcs
    ]
    hucs = [
        hucs[key]
        for key
        in range(len(huc_ids))
        if key
        not in remove_keys_idcs
    ]
    lidar_index = [
        lidar_index[key]
        for key
        in range(len(huc_ids))
        if key
        not in remove_keys_idcs
    ]
    huc_ids = [
        huc_ids[key] 
        for key
        in range(len(huc_ids))
        if key
        not in remove_keys_idcs
    ]

    hucs['crs'] = CRS.from_proj4(init=hucs.crs.to_proj4())
    hucs['bounds'] = hucs.bounds
    hucs['height'] = hucs.height
    
    dst_crs = CRS.from_epsg(hucs.crs.to_epsg())

    multiprocessing.set_start_method('spawn')

    N_CORES = multiprocessing.cpu_count() - 1
    if args.tempdir:
        tempdir = args.tempdir
    else:
        tempdir = tempfile.gettempdir()

    ## Run the output functions for each of the HUCs
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    arguments = zip(
        huc_ids,
        flowlines,
        catchments,
        hucs,
        lidar_index,
        repeat(args),
        repeat(output_prefix),
        repeat(dst_crs),
        repeat(tempdir)
    )
    tasks = [
        (
            multiprocessing.Process(
                target = output_files,
                args = (argument,return_dict),
                name = uuid.uuid4().hex
            ),
            argument
        )
        for argument
        in arguments
    ]
    print('tasks: ',[task[0] for task in tasks])
    sys.stdout.flush()
    pool = TaskProcessor(
        n_cores = N_CORES,
        max_b = MAX_B,
        tempdir = tempdir,
        percent_free_mem = args.percent_free_mem,
        percent_free_disk = args.percent_free_disk,
        return_dict = return_dict,
        tasks = tasks
    )
    pool.start()
    pool.join()
    
    #print(return_dict)
    #for key,value in return_dict.items():
    #    print(key+':'+value)
    #    sys.stdout.flush()
    #    if isinstance(value, ExceptionWrapper):
    #        value.re_raise()
    #    else:
    #        print(value)

    print("All catchments, flowlines, roughnesses, and rasters created for each HUC")
    print("Time spent with ", N_CORES, " threads in milliseconds")
    print("-----", int((time.time()-start_time)*1000), "-----")
    sys.stdout.flush()

    os.environ['GDAL_DATA'] = oldgdal_data

if __name__ == "__main__":
    main()

