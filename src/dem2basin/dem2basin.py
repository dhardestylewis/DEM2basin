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

import os
from pathlib import Path
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

    ## Check that the required input files have been defined
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

def _drop_index_columns(shape_original):

    shape = shape_original.drop(
        columns = [
            'index',
            'index_left',
            'index_right'
        ],
        errors = 'ignore'
    )

    return(shape)

def find_huc_level(shape_original):

    regexp = re.compile('HUC[0-9]*')
    huc_level = list(filter(
        regexp.match,shape_original.columns.to_list()
    ))[0]

    return(huc_level)

def set_index_to_huc(shape_original,sort=True):

    huc_level = find_huc_level(shape_original)
    shape = shape_original.set_index(huc_level,drop=False)
    shape.index.name = 'HUC'
    shape.index = shape.index.astype('int64')
    if sort:
        shape.sort_index(inplace=True)

    return(shape)

def read_file_or_gdf(shape_input,**kwargs):

    if isinstance(shape_input,str):
        shape = gpd.read_file(shape_input,**kwargs)
    else:
        shape = shape_input.copy()

    return(shape)

def get_hucs_by_shape(
    shape_input,
    hucs_input,
    hucs_layer = None,
    sort = True,
    select_utm = None,
    to_utm = True,
    drop_index_columns = True
):
    ## Find the HUCs that intersect with the input polygon

    #shape_input = 'data/TX-Counties/Young/TX-County-Young.shp'
    shape = read_file_or_gdf(shape_input)

    shape['dissolve'] = True
    shape = shape.dissolve(by='dissolve').reset_index(drop=True)
    shape = gpd.GeoDataFrame(shape[['geometry']])

    #hucs_input = 'data/WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp'
    hucs = read_file_or_gdf(hucs_input,layer=hucs_layer)

    crs = find_utm(hucs,select_utm)
    if not to_utm:
        hucs_original = hucs.copy()
    to_crs(crs,[hucs,shape])
    #hucs.to_crs(crs,inplace=True)
    #shape.to_crs(hucs.crs,inplace=True)

    if drop_index_columns:
        hucs = _drop_index_columns(hucs)

    hucs = gpd.overlay(
        hucs,
        shape,
        how = 'intersection'
    )

    hucs = set_index_to_huc(hucs,sort)
    if not to_utm:
        hucs_original = set_index_to_huc(hucs_original,sort)
        hucs = hucs_original.loc[hucs_original.index.isin(hucs.index)]

    return(hucs)

def set_and_sort_index(
    dataframe,
    column,
    drop = True,
):

    dataframe.set_index(column,inplace=True,drop=drop)
    dataframe.sort_index(inplace=True)

    return(dataframe)

def index_dataframe_by_dataframe(dataframe_left,dataframe_right):

    dataframe = dataframe_left[
        dataframe_left.index.isin(dataframe_right.index)
    ]
    return(dataframe)

def get_nhd_by_shape(
    shape,
    nhd_input,
    layer = None,
#    comid_only = True,
    drop_index_columns = True,
    comid_column = None,
    fix_invalid_geometries = False
):
    ## Identify flowlines of each HUC

    #nhd_file = 'data/NFIEGeo_12.gdb'
    ## Find the flowlines whose representative points are within these HUCs
    geodataframe = read_file_or_gdf(nhd_input,layer=layer,mask=shape)

    if drop_index_columns:
        geodataframe = _drop_index_columns(geodataframe)

    if comid_column is not None:
        geodataframe.rename(columns={comid_column:'COMID'},inplace=True)
    geodataframe = set_and_sort_index(geodataframe,'COMID')

    if fix_invalid_geometries:
        geodataframe.geometry = geodataframe.buffer(0)

    return(geodataframe)

def get_representative_points(flowlines,hucs,drop_index_columns=True):

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

    flowlines = flowlines_original.copy()

    flowlines.loc[flowlines[streamorder_col]==0,roughness_col] = .99
    flowlines.loc[flowlines[streamorder_col]==1,roughness_col] = .2
    flowlines.loc[flowlines[streamorder_col]==2,roughness_col] = .1
    flowlines.loc[flowlines[streamorder_col]==3,roughness_col] = .065
    flowlines.loc[flowlines[streamorder_col]==4,roughness_col] = .045
    flowlines.loc[flowlines[streamorder_col]==5,roughness_col] = .03
    flowlines.loc[flowlines[streamorder_col]==6,roughness_col] = .01
    flowlines.loc[flowlines[streamorder_col]==7,roughness_col] = .025

    return(flowlines)

def clip_geodataframe_by_attribute(
    geodataframe,
    geodataframe_with_attribute,
    attribute = None
):

    ## Find the flowlines corresponding with these catchments
    ##  (Note: this line is optional.
    ##  Commenting it out will result in non-COMID-identified flowlines)
    #if comid_only==True:
    geodataframe = index_dataframe_by_dataframe(
        geodataframe,
        geodataframe_with_attribute
    )

    ## Determine which HUCs each of the flowlines and catchments belong to
    geodataframe[attribute] = geodataframe_with_attribute.loc[
        geodataframe.index,
        attribute
    ]

    return(geodataframe)

def find_common_utm(shape_original):
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

def find_utm(gdf_original,select_utm=None):
    ## Buffer the catchments for each HUC

    gdf = gdf_original.reset_index()

    ## Are the catchments all within the same UTM?
    if select_utm:
        utm_output = select_utm
    else:
        utm_output = find_common_utm(gdf)

    ## Buffer the HUC catchments
    if gdf.crs.datum.name=='World Geodetic System 1984':
        #crs = CRS(proj='utm', zone=utm_output[0], datum='WGS84')
        if utm_output==13:
            crs = 'epsg:32613'
        elif utm_output==14:
            crs = 'epsg:32614'
        elif utm_output==15:
            crs = 'epsg:32615'
        else:
            print("ERROR: UTMs outside of 13-15 not yet supported.")
            if hasattr(main,'__file__'):
                sys.exit(0)
    elif gdf.crs.datum.name=='North American Datum 1983' or gdf.crs.datum.name=='D_NORTH_AMERICAN_1983' or gdf.crs.datum.name=='NAD83 (National Spatial Reference System 2011)' or gdf.crs.datum.name=='NAD83':
        #crs = CRS(proj='utm', zone=utm_output[0], datum='NAD83')
        if utm_output==13:
            crs = 'epsg:6342'
        elif utm_output==14:
            crs = 'epsg:6343'
        elif utm_output==15:
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

    gdf = gdf_original.to_crs(crs)
    gdf['geometry'] = gdf.buffer(meters_buffered)
    gdf.crs = crs

    return(gdf)

def reproject_to_utm_and_buffer(gdf_original,select_utm=None):
    ## Reproject a GeoDataFrame to the most common UTM and then buffer it

    crs = find_utm(gdf_original,select_utm)
    gdf = reproject_and_buffer(gdf_original,crs)

    return(gdf)

class LidarIndex():
    """
    Georeference TNRIS LIDAR 1m dataset
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
        hucs,
        lidar_availability_file,
        lidar_parent_directory,
        drop_index_columns = True
    ):
    ## TODO: divide into:
    ##   - correcting the LIDAR availability file, and
    ##   - applying HUCs column attribute
    
        availability = gpd.read_file(lidar_availability_file,mask=hucs)
        availability = availability[availability['demname']!='No Data Exist']
#        availability.drop(
#            columns = ['tilename','las_size_m','laz_size_m'],
#            inplace = True
#        )

        if drop_index_columns:
            availability = _drop_index_columns(availability)
        availability = gpd.sjoin(
            availability,
            hucs[['HUC','geometry']].to_crs(availability.crs),
            how = 'inner',
            op = 'intersects'
        )
        #availability.rename(columns={'index_right':'index_shape'},inplace=True)
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
            axis=1
        )
        availability['pathlower'] = availability['path'].apply(
            lambda path: path.lower()
        )
        availability = availability.merge(lidardatafiles,on='pathlower')
        availability.drop(
#            columns = ['demname','dirname','path','pathlower'],
            columns = ['path','pathlower'],
            inplace = True
        )
        availability['lidar_file'] = availability['lidar_file'].apply(
            lambda fn: str(fn)
        )

        return(availability)
    
    def index_lidar_files_original(
        self,
        hucs,
        lidar_availability_file,
        lidar_parent_directory
    ):
        ## Identify each DEM tile file for our study area
    
        ## Find the DEM tiles that intersect with these buffered HUC catchments
        #lidar_availability = 'data/TNRIS-LIDAR-Availability-20191213.shp/TNRIS-LIDAR-Availability-20191213.shp'
        lidar_availability = gpd.read_file(lidar_availability_file)
#        lidar_availability = lidar_availability.loc[gpd.sjoin(
#            lidar_availability,
#            hucs[['HUC','geometry']].to_crs(lidar_availability.crs),
#            op = 'intersects',
#            how = 'inner'
#        ).index]
        lidar_availability = lidar_availability.loc[gpd.overlay(
            lidar_availability,
            hucs[['HUC','geometry']].to_crs(lidar_availability.crs),
            how = 'intersection'
        ).index]
        lidar_availability = lidar_availability[
            lidar_availability['demname'] != 'No Data Exist'
        ]
    
        ## Construct an exact path for each DEM tile
    
        filename_extensions = ['.dem','.img','.tif']
    
        for extension in filename_extensions:
            lidar_availability['demname'] = lidar_availability['demname'].str.replace(
                extension + '$',
                ''
            )
    
        for dirname in lidar_availability['dirname'].unique():
    
            lidar_files = []
            #raster = '/scratch/projects/tnris/tnris-lidardata'
            dirname_absolute = os.path.join(
                lidar_parent_directory,
                dirname,
                'dem'
            ) + os.sep
            for extension in filename_extensions:
                lidar_files.extend(glob.glob(dirname_absolute+'*'+extension))
            if len(lidar_files) == 0:
                break
            extensions_within_directory = set([
                os.path.splitext(os.path.basename(filename))[1]
                for filename
                in lidar_files]
            )
    
            ## If more than one vector image extension found in a DEM project,
            ##  then figure out each file's extension individually
            ## TODO: Test this against stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro
    
            dirname_absolute_pathlib = Path(dirname_absolute)
    
            resolution = np.unique(np.array([
                re.split('-|_',filename.parts[-1])[1]
                for filename
                in dirname_absolute_pathlib.rglob('*')
            ]))[0]
    
            if len(extensions_within_directory) > 1:
                for demname in lidar_availability.loc[lidar_availability['dirname']==dirname,'demname'].unique():
                    truth_dirname = lidar_availability['dirname']==dirname
                    truth_demname = lidar_availability['demname']==demname
                    truth = np.logical_and(truth_dirname,truth_demname)
                    for extension in filename_extensions:
                        lidar_file = lidar_availability.loc[
                            truth,
                            'demname'
                        ].apply(lambda x: os.path.join(
                            dirname_absolute,
                            re.split('-|_',x)[0] +
                                '-' +
                                resolution +
                                '_' +
                                re.split('-|_',x)[2] +
                                extension
                        ))
                    lidar_availability.loc[truth,'lidar_file'] = lidar_file
            ## Else do all the files in a DEM project at once
            elif len(extensions_within_directory) == 1:
                lidar_file = lidar_availability.loc[
                    lidar_availability['dirname'] == dirname,
                    'demname'
                ].apply(lambda x: os.path.join(
                    dirname_absolute,
                    re.split('-|_',x)[0] +
                        '-' +
                        resolution +
                        '_' +
                        re.split('-|_',x)[2] +
                        list(extensions_within_directory)[0]
                ))
                #lidar_file.drop_duplicates(inplace=True)
                for filename in dirname_absolute_pathlib.rglob('*'):
                    if len(lidar_file[lidar_file.str.lower()==str(filename).lower()].index)>0:
                        lidar_file.loc[
                            lidar_file[
                                lidar_file.str.lower()==filename.as_posix().lower()
                            ].index[0]
                        ] = filename.as_posix()
                lidar_file_drop = lidar_file[~lidar_file.isin([
                    filename.as_posix()
                    for filename
                    in list(dirname_absolute_pathlib.rglob('*'))
                ])].index
                lidar_availability.loc[
                    lidar_availability['dirname'] == dirname,
                    'lidar_file'
                ] = lidar_file
                lidar_availability.drop(
                    index = lidar_file_drop,
                    inplace = True,
                    errors = 'ignore'
                )
            else:
                continue
    
        lidar_availability.dropna(subset=['lidar_file'],inplace=True)
        lidar_availability.drop(
            columns = [
                'path',
                'pathlower'
            ],
            inplace = True,
            errors = 'ignore'
        )
        lidar_availability.rename(columns={'index_right':'HUC'},inplace=True)
        lidar_availability = lidar_availability.loc[
            :,
            ~lidar_availability.columns.duplicated(keep='last')
        ]
        lidar_availability = lidar_availability[
            lidar_availability['lidar_file'].apply(
                lambda fn: Path(fn).is_file()
            )
        ]
        #lidar_availability_grouped = lidar_availability.groupby('index_right')
    
        return(lidar_availability)

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise(self.ee.with_traceback(self.tb))

def delete_file(filename):

    try:
        Path(str(filename)).unlink(missing_ok=True)
    ## TODO: is this the correct exception for missing `missing_ok` parameter?
    except FileNotFoundError:
        Path(str(filename)).unlink()
    except:
        pass

def skip_function_if_file_exists(function,filename,skip_existing=True):

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

def get_flowlines_and_representative_points_by_huc(hucs,nhd_input):
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

    flowlines = clip_geodataframe_by_attribute(
        flowlines,
        flowline_representative_points,
        attribute = 'HUC'
    )

    flowlines = set_roughness_by_streamorder(flowlines)

    return(flowlines,flowline_representative_points)

def get_catchments_by_huc(hucs,nhd_input,flowline_representative_points):

    catchments = get_nhd_by_shape(
        hucs,
        nhd_input,
        layer = 'Catchment',
        comid_column = 'FEATUREID',
        fix_invalid_geometries = True
    )

    catchments = clip_geodataframe_by_attribute(
        catchments,
        flowline_representative_points,
        attribute = 'HUC'
    )

    return(catchments)

def get_hucs_from_catchments(catchments):

    hucs = catchments.dissolve(by='HUC')
    hucs.reset_index(inplace=True)
    hucs = reproject_to_utm_and_buffer(hucs)

    return(hucs)

def to_crs(crs,geodataframes):

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
        lidar_index['lidar_file']==raster.name
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

def build_vrt(filenames,vrt_filename,lowest_resolution=False):

    filenames = [str(filename) for filename in filenames]

    if lowest_resolution:
        resolution = 'lowest'
    else:
        resolution = 'highest'

    vrt_options = gdal.BuildVRTOptions(
        resampleAlg = 'near',
        #addAlpha = True,
        resolution = resolution
        #separate=True
    )

    vrt = gdal.BuildVRT(vrt_filename,filenames,options=vrt_options)
    vrt = None

def build_vrts(lidar_index,vrt_filename_template,lowest_resolution=False):

    filenames = lidar_index['lidar_file'].to_list()

    for filename in filenames:

        vrt_filename = (
            os.path.splitext(str(vrt_filename_template))[0] +
            '-' +
            os.path.splitext(Path(str(filename)).name)[0] +
            '.vrt'
        )

        lidar_index.loc[
            lidar_index['lidar_file']==filename,
            'vrt_filename'
        ] = vrt_filename

        build_vrt(filename,vrt_filename)

    return(lidar_index)

def reproject_raster(filename,reprojected_filename,dst_crs=None):

    if dst_crs is not None:
        dst_crs = str(dst_crs)

    raster = gdal.Open(str(filename))

    warp = gdal.Warp(str(reprojected_filename),raster,dstSRS=dst_crs)
    warp = None

def reproject_rasters(filenames,reprojected_filenames,dst_crs):

    for filename,reprojected_filename in zip(filenames,reprojected_filenames):

        reproject_raster(str(filename),str(reprojected_filename),dst_crs)

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

def mask_raster(raster,meta,mask_geometry,memoryfile=False):

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

def _get_mosaic_and_output_raster(
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
    lidar_index,
    huc_id,
    dst_crs,
    subdirectory,
    return_dict,
    filename,
    skip_existing = True
):

    skip_function_if_file_exists(
        _get_mosaic_and_output_raster(
            lidar_index,
            huc_id,
            dst_crs,
            subdirectory,
            return_dict,
            filename
        ),
        filename,
        skip_existing = skip_existing
    )

def make_directories_and_error_files(directory,output_prefix,huc_id):

    subdirectory = os.path.join(
        directory,
        str(output_prefix) + '-' + str(huc_id)
    )
    print(subdirectory)
    sys.stdout.flush()
    Path(subdirectory).mkdir(parents=True, exist_ok=True)

    path_notime = os.path.join(
        subdirectory,
        "jobNoTimeLeftWhileProcessing.err"
    )
    Path(path_notime).touch()

    path_gt1m = os.path.join(subdirectory, "allGT1m.err")
    file_gt1m = Path(path_gt1m)

    path_enclose = os.path.join(
        subdirectory,
        "rasterDataDoesNotEnclose.err"
    )
    file_enclose = Path(path_enclose)

    return(subdirectory,path_notime,file_gt1m,file_enclose)

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
        output_prefix,
        dst_crs,
        tempdir
    ) = arguments

    try:

        (
            subdirectory,
            path_notime,
            file_gt1m,
            file_enclose
        ) = make_directories_and_error_files(directory,output_prefix)

        if not (file_gt1m.is_file() or file_enclose.is_file()):

            #output_nhd(flowlines,catchments,huc_ids)
            filename = os.path.join(subdirectory,'Flowline.shp')
            write_geodataframe(flowlines,filename)

            filename = os.path.join(subdirectory,'Roughness.csv')
            write_roughness_table(flowlines,filename)

            filename = os.path.join(subdirectory,'Catchment.shp')
            write_geodataframe(catchments,filename)

            filename = os.path.join(subdirectory, 'Elevation.tif')
            get_mosaic_and_output_raster(filename)

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
                    flowshu12shape,
                    catchshu12shape,
                    hu12catchs,
                    lidar_index
                ) = pickle.load(input)
        else:
            no_restart_file = True

    if not args.restart or no_restart_file:

        hucs = get_hucs_by_shape(args.shapefile,args.hucs)

        (
            flowlines,
            flowline_representative_points
        ) = get_flowlines_and_representative_points_by_huc(hucs,args.nhd)

        catchments = get_catchments_by_huc(
            hucs,
            args.nhd,
            flowline_representative_points
        )

        flowlines = index_dataframe_by_dataframe(
            flowlines,
            catchments
        )

        hucs = get_hucs_from_catchments(catchments)

        #lidar_index = index_lidar_files_dev(hucs)
        lidar_index_obj = LidarIndex()
        lidar_index = lidar_index_obj.index_lidar_files(
            hucs,
            args.lidar_availability,
            args.lidar_parent_directory
        )

        to_crs(hucs.crs,[flowlines,catchments,lidar_index])

        huc_ids = get_merged_column(
            'HUC',
            [lidar_index,flowlines,catchments,hucs]
        )
        ## Ensure lists share the same HUCs
#        huc_ids = np.sort(list(
#            set(lidar_index['HUC']).intersection(flowlines['HUC'])
#        ))
#        huc_ids = np.sort(list(
#            set(huc_ids).intersection(catchments['HUC'])
#        ))
#        huc_ids = np.sort(list(
#            set(huc_ids).intersection(hucs['HUC'])
#        ))
    
        ## Divide into lists per HUC
        flowlines = list(dict(tuple(
            flowlines[
                flowlines['HUC'].isin(huc_ids)
            ].sort_values('HUC').groupby('HUC')
        )).values())
        catchments = list(dict(tuple(
            catchments[
                catchments['HUC'].isin(huc_ids)
            ].sort_values('HUC').groupby('HUC')
        )).values())
        hucs.drop(
            columns = ['index_left','index_right'],
            errors = 'ignore',
            inplace = True
        )
        hucs = list(dict(tuple(
            hucs[
                hucs['HUC'].isin(huc_ids)
            ].sort_index().groupby('HUC')
        )).values())
        lidar_index = list(dict(tuple(
            lidar_index[
                lidar_index['HUC'].isin(huc_ids)
            ].sort_values('HUC').groupby('HUC')
        )).values())
    
        ## Sort lists by estimated memory usage
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
