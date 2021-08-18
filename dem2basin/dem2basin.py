## GeoFlood preprocessing 1m DEM data
## Author: Daniel Hardesty Lewis

## Import needed modules
import argparse
import pandas as pd
import fiona
import geopandas as gpd
import utm
#from pyproj import CRS
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask
from rasterio.merge import merge
from rasterio.features import shapes
import os
from pathlib import Path
import csv
import glob
import sys
from threading import Thread
from collections import deque
import multiprocessing
import time
from itertools import repeat
import tblib.pickling_support
tblib.pickling_support.install()
import logging
import psutil
#from memory_profiler import profile
import pickle
import gc
import tempfile
import shutil
import uuid
import traceback
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

def get_hucs_by_shapefile(shapefile,hucs_file):
    ## Find the HUCs that intersect with the input polygon

    #shapefile = 'data/TX-Counties/Young/TX-County-Young.shp'
    shape = gpd.read_file(shapefile)

#    shape.drop(
#        columns = ['Shape_Leng','Shape_Area'],
#        inplace = True,
#        errors = 'ignore'
#    )
    shape.drop(
        columns = [
            'index_left',
            'index_right'
        ],
        inplace = True,
        errors = 'ignore'
    )

    #hucs_file = 'data/WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp'
    hucs = gpd.read_file(hucs_file,mask=shape)
    huc_level = list(filter(r.match,hucs.columns.to_list()))[0]
    hucs = hucs[[huc_level,'geometry']]
    huc.drop(
        columns = [
            'HUC'
        ],
        inplace = True,
        errors = 'ignore'
    )
    huc.rename(
        columns = {huc_level:'HUC'},
        inplace = True,
        errors = 'ignore'
    )

    return(hucs,huc_level)

def get_flowlines_by_huc(hucs,nhd,comid_only=True):
    ## Identify flowlines of each HUC

    #nhd = 'data/NFIEGeo_12.gdb'
    ## Find the flowlines whose representative points are within these HUCs
    flowlines = gpd.read_file(nhd,layer='Flowline',mask=hucs)

#    flowlines.drop(
#        columns = [
#            'Shape_Length',
#            'Shape_Area',
#            'AreaSqKM',
#            'index_left',
#            'index_right'
#        ],
#        inplace = True,
#        errors = 'ignore'
#    )
    flowlines.drop(
        columns = [
            'index_left',
            'index_right'
        ],
        inplace = True,
        errors = 'ignore'
    )

    flowlines.reset_index(inplace=True)
    flowlines.set_index('COMID',inplace=True)
    flowlines.sort_index(inplace=True)

    flowline_representative_points = flowlines.copy()
    flowline_representative_points['geometry'] = flowlines.representative_point()
    flowline_representative_points = gpd.sjoin(
        flowline_representative_points,
        hucs[['HUC']].to_crs(flowline_representative_points.crs),
        op = 'intersects',
        how = 'inner'
    )
    flowline_representative_points.drop(
        columns = [
            'index_right'
        ],
        inplace = True
    )

    ## Find the flowlines corresponding with these cactchments
    ##  (Note: this line is optional.
    ##  Commenting it out will result in non-COMID-identified flowlines)
    if comid_only==True:
        flowlines = flowlines[flowlines.index.isin(
            flowline_representative_points.index
        )]
    

    ## Determine which HUCs each of the flowlines and catchments belong to
    flowlines['HUC'] = flowline_representative_points.loc[
        flowlines.index,
        'HUC'
    ]

    flowlines.loc[flowlines['StreamOrde']==0,'Roughness'] = .99
    flowlines.loc[flowlines['StreamOrde']==1,'Roughness'] = .2
    flowlines.loc[flowlines['StreamOrde']==2,'Roughness'] = .1
    flowlines.loc[flowlines['StreamOrde']==3,'Roughness'] = .065
    flowlines.loc[flowlines['StreamOrde']==4,'Roughness'] = .045
    flowlines.loc[flowlines['StreamOrde']==5,'Roughness'] = .03
    flowlines.loc[flowlines['StreamOrde']==6,'Roughness'] = .01
    flowlines.loc[flowlines['StreamOrde']==7,'Roughness'] = .025

    return(flowlines,flowline_representative_points)

def get_catchments_by_huc(hucs,nhd,flowline_representative_points):
    ## Identify catchments of each HUC

    ## Find the catchments corresponding with these flowlines
    catchments = gpd.read_file(nhd,layer='Catchment',mask=hucs)
#    catchments.drop(
#        columns=[
#            'Shape_Length',
#            'Shape_Area',
#            'AreaSqKM',
#            'index_left',
#            'index_right'
#        ],
#        inplace = True,
#        errors = 'ignore'
#    )
    catchments.drop(
        columns = [
            'index_left',
            'index_right'
        ],
        inplace = True,
        errors = 'ignore'
    )

    catchments.reset_index(inplace=True)
    catchments.set_index('FEATUREID',inplace=True)
    catchments.index.rename('COMID',inplace=True)
    catchments.sort_index(inplace=True)
    catchments = catchments[catchments.index.isin(
        flowline_representative_points.index
    )]

    catchments['HUC'] = flowline_representative_points.loc[
        catchments.index,
        'HUC'
    ]

    catchments.geometry = catchments.buffer(0)

    return(catchments)

def index_flowlines_by_catchments(flowlines,catchments):

    flowlines = flowlines[
        flowlines.index.isin(catchments.index)
    ]
    return(flowlines)

def buffer_hucs(hucs,meters_buffered=500.):
    ## Buffer the catchments for each HUC

    def unique(shape):
        ## Determine whether the administrative division is within a single UTM

        shape.to_crs('epsg:4326',inplace=True)

        ## TODO: make this majority count an option
        ##  and bring back cross-utm error as default behaviour
        uniq = shape.representative_point().apply(lambda p: utm.latlon_to_zone_number(p.y,p.x)).value_counts().idxmax()

        #if uniq.shape[0] > 1:
        #    print("ERROR: Cross-UTM input shapefile not yet supported.")
        #    sys.exit(0)

        return(uniq)

    hucs.reset_index(inplace=True)

    ## Are the catchments all within the same UTM?
    uniq = unique(hucs)

    ## Buffer the HUC catchments
    if hucs.crs.datum.name=='World Geodetic System 1984':
        #crs = CRS(proj='utm', zone=uniq[0], datum='WGS84')
        if uniq==13:
            crs = 'epsg:32613'
        elif uniq==14:
            crs = 'epsg:32614'
        elif uniq==15:
            crs = 'epsg:32615'
        else:
            print("ERROR: UTMs outside of 13-15 not yet supported.")
            sys.exit(0)
    elif hucs.crs.datum.name=='North American Datum 1983' or hucs.crs.datum.name=='D_NORTH_AMERICAN_1983' or hucs.crs.datum.name=='NAD83 (National Spatial Reference System 2011)':
        #crs = CRS(proj='utm', zone=uniq[0], datum='NAD83')
        if uniq==13:
            crs = 'epsg:6342'
        elif uniq==14:
            crs = 'epsg:6343'
        elif uniq==15:
            crs = 'epsg:6344'
        else:
            print("ERROR: UTMs outside of 13-15 not yet supported.")
            sys.exit(0)
    else:
        print("ERROR: Non-WGS/NAD datum not yet supported")
        sys.exit(0)
    hucs['geometry'] = hucs.to_crs(crs).buffer(meters_buffered)
    hucs.crs = crs

    ## Are the buffered catchments all within the same UTM?
    #unique(hucs)

    return(hucs)

class LidarIndex():
    """
    Georeference TNRIS LIDAR 1m dataset
    """

    def __init__(
        self,
        hucs = None,
        lidar_availability_file = None,
        lidar_parent_directory = None,
        lidar_index = None
    ):

        if hucs is None:
            self.hucs = gpd.GeoDataFrame()
        else:
            self.hucs = hucs

        if lidar_availability_file is None:
            self.lidar_availability_file = ''
        else:
            self.lidar_availability_file = lidar_availability_file

        if lidar_parent_directory is None:
            self.lidar_parent_directory = ''
        else:
            self.lidar_parent_directory = lidar_parent_directory

        if lidar_index is None:
            self.lidar_index = gpd.GeoDataFrame()
        else:
            self.lidar_index = lidar_index

    def index_lidar_files_dev(self,hucs):
    
        availability = gpd.read_file(args.availability,mask=hucs)
        availability = availability[availability['demname']!='No Data Exist']
        availability.drop(
            columns = ['tilename','las_size_m','laz_size_m'],
            inplace = True
        )
        hucs.to_crs(availability.crs,inplace=True)
        availability = gpd.sjoin(
            availability,
            hucs[['HUC']],
            how = 'inner',
            op = 'intersects'
        )
        availability.rename(columns={'index_right':'index_shape'},inplace=True)
        types = ('*.img', '*.dem', '*.tif')
        lidardatafiles = []
        for files in types:
            lidardatafiles.extend(list(Path(args.lidar_parent_directory).rglob(os.path.join(
                '*',
                'dem',
                files
            ))))
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
                os.path.join(*Path(args.lidar_parent_directory).parts),
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
            columns = ['demname','dirname','path','pathlower'],
            inplace = True
        )
        availability['lidar_file'] = availability['lidar_file'].apply(
            lambda fn: str(fn)
        )
        return(availability)
    
    def index_lidar_files(self,hucs,lidar_availability_file,lidar_parent_directory):
        ## Identify each DEM tile file for our study area
    
        ## Find the DEM tiles that intersect with these buffered HUC catchments
        #lidar_availability = 'data/TNRIS-LIDAR-Availability-20191213.shp/TNRIS-LIDAR-Availability-20191213.shp'
        lidar_availability = gpd.read_file(lidar_availability_file)
        lidar_availability = gpd.sjoin(
            lidar_availability,
            hucs[['HUC']].to_crs(lidar_availability.crs),
            op = 'intersects',
            how = 'inner'
        )
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
            columns = ['demname','dirname','path','pathlower'],
            inplace = True,
            errors = 'ignore'
        )
        lidar_availability.rename(columns={'index_right':'HUC'},inplace=True)
        lidar_availability = lidar_availability.loc[
            :,
            ~lidar_availability.columns.duplicated(keep='last')
        ]
        lidar_availability = lidar_availability[
            lidar_availability['lidar_file'].apply(lambda fn: Path(fn).is_file())
        ]
        #lidar_availability_grouped = lidar_availability.groupby('index_right')
    
        return(lidar_availability)

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise(self.ee.with_traceback(self.tb))

#@profile
def output_files(arguments,return_dict):
#def output(flow_key,flowshu12shape,catchshu12shape,hu12catchs,avail_hu12catchs_group,args,prefix,dst_crs,mem_estimates):
    ## Output catchments, flowlines, roughnesses, and rasters

    try:

        def output_nhd(flowshu12shape,catchshu12shape,hu):
            ## For each HUC, write the flowlines, catchments, and roughnesses corresponding to it

            out_path = os.path.join(subdirectory, 'Flowlines.shp')
            my_file = Path(out_path)
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_flowlines:
                pass
            else:
                my_file.unlink(missing_ok=True)
                flowshu12shape.reset_index().to_file(str(out_path))

            out_path = os.path.join(subdirectory, 'Roughness.csv')
            my_file = Path(out_path)
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_roughnesses:
                pass
            else:
                my_file.unlink(missing_ok=True)
                with open(out_path, 'w', newline='') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow(['COMID','StreamOrde','Roughness'])
                    for comid in np.sort(flowshu12shape.index.unique()):
                        writer.writerow([comid,flowshu12shape.loc[comid,'StreamOrde'],flowshu12shape.loc[comid,'Roughness']])

            out_path = os.path.join(subdirectory, 'Catchments.shp')
            my_file = Path(out_path)
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_catchments:
                pass
            else:
                my_file.unlink(missing_ok=True)
                catchshu12shape.reset_index().to_file(str(out_path))

        def get_mosaic(avail_hu12catchs_group,hu,break_hu,dst_crs):
            ## Get mosaic of DEMs for each HUC

            def append_check(src_files_to_mosaic,var,subdirectory,hu):
                ## Check each raster's resolution in this HUC

                #if any(np.float16(i) > 1. for i in var.res):
                #    out_path = os.path.join(subdirectory, "gt1m.err")
                #    Path(out_path).touch()
                #    print('WARNING: >1m raster input for HUC: '+str(hu))
                #    sys.stdout.flush()
                #else:
                src_res_min_to_mosaic.append(min(var.res))
                src_res_max_to_mosaic.append(min(var.res))
                src_x_to_mosaic.append(var.res[0])
                src_y_to_mosaic.append(var.res[1])
                src_files_to_mosaic.append(var)

                return(src_files_to_mosaic,src_res_min_to_mosaic,src_res_max_to_mosaic,src_x_to_mosaic,src_y_to_mosaic)

            def reproject_append(
                fp,
                dst_crs,
                arguments,
                memfile,
                src_files_to_mosaic,
                src_res_min_to_mosaic,
                src_res_max_to_mosaic,
                src_x_to_mosaic,
                src_y_to_mosaic,
                subdirectory,
                hu
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
                    if arguments[5].memfile:
                        dst = memfile[fp].open(**out_meta)
                    else:
                        dst = rasterio.open(
                            memfile[fp],
                            'w+',
                            **out_meta
                        )
                        memfile[fp] = dst
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst.transform,
                            dst_crs=dst.crs,
                            resampling=Resampling.nearest
                        )
                else:
                    dst = src
                    if not arguments[5].memfile:
                        memfile[fp] = dst
                (
                    src_files_to_mosaic,
                    src_res_min_to_mosaic,
                    src_res_max_to_mosaic,
                    src_x_to_mosaic,
                    src_y_to_mosaic
                ) = append_check(
                    src_files_to_mosaic,
                    dst,
                    subdirectory,
                    hu
                )

                return(
                    src_files_to_mosaic,
                    src_res_min_to_mosaic,
                    src_res_max_to_mosaic,
                    src_x_to_mosaic,
                    src_y_to_mosaic
                )

            ## Reproject the mosaic to DEM tiles pertaining to each HUC
            dem_fps = list(avail_hu12catchs_group['lidar_file'])
            src_files_to_mosaic = []
            src_res_min_to_mosaic = []
            src_res_max_to_mosaic = []
            src_x_to_mosaic = []
            src_y_to_mosaic = []
            memfile = {}
            if arguments[5].memfile:
                for fp in dem_fps:
                    memfile[fp] = MemoryFile()
            else:
                Path(os.path.join(
                    arguments[8],
                    str(arguments[0])
                )).mkdir(parents=True, exist_ok=True)
                for fp in dem_fps:
                    memfile[fp] = os.path.join(
                        arguments[8],
                        str(arguments[0]),
                        next(tempfile._get_candidate_names())
                    )
            for fp in dem_fps:
                try:
                    (
                        src_files_to_mosaic,
                        src_res_min_to_mosaic,
                        src_res_max_to_mosaic,
                        src_x_to_mosaic,
                        src_y_to_mosaic
                    ) = reproject_append(
                        fp,
                        dst_crs,
                        arguments,
                        memfile,
                        src_files_to_mosaic,
                        src_res_min_to_mosaic,
                        src_res_max_to_mosaic,
                        src_x_to_mosaic,
                        src_y_to_mosaic,
                        subdirectory,
                        hu
                    )
                except Exception as err:
                    print(
                        '[EXCEPTION] Exception on HUC: ' +
                        str(arguments[0])
                    )
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(
                        exc_tb.tb_frame.f_code.co_filename
                    )[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    sys.stdout.flush()
                    if arguments[5].log:
                        logging.debug(
                            '[EXCEPTION] on HUC ' +
                            str(arguments[0])
                        )
                    traceback.print_tb(err.__traceback__)
                    pass

            if len(src_files_to_mosaic) == 0:

                out_path = os.path.join(subdirectory, "allGT1m.err")
                Path(out_path).touch()
                print('WARNING: Found no <=1m raster input data for HUC: '+str(hu))
                sys.stdout.flush()

                break_hu = True
                mosaic_tuple = ()
                return(break_hu,mosaic_tuple)

            else:

                src_files_to_mosaic = pd.DataFrame(data={
                    'Files':src_files_to_mosaic,
                    'min(resolution)':src_res_min_to_mosaic,
                    'max(resolution)':src_res_max_to_mosaic
                })
                if arguments[5].lowest_resolution:
                    src_files_to_mosaic.sort_values(by=['min(resolution)','max(resolution)'],inplace=True)
                else:
                    src_files_to_mosaic.sort_values(by=['max(resolution)','min(resolution)'],inplace=True)
                mosaic, out_trans = merge(list(src_files_to_mosaic['Files']),res=(max(src_x_to_mosaic),max(src_y_to_mosaic)))
                for src in src_files_to_mosaic['Files']:
                    try:
                        src.close()
                    except:
                        pass
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": 'GTiff',
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": dst_crs
                })
                for keyvalue in memfile.items():
                    try:
                        keyvalue[1].close()
                        Path(keyvalue[1].name).unlink(missing_ok=True)
                    except:
                        pass

                mosaic_tuple = (mosaic,out_meta)
                return(break_hu,mosaic_tuple)

        def output_raster(hu_buff_geom,mosaic,out_meta,path_elevation):
            ## Crop and output the mosaic to the buffered catchments of each HUC

            if arguments[5].memfile:
                with MemoryFile() as memfile:
                    with memfile.open(**out_meta) as dataset:
                        dataset.write(mosaic)
                    with memfile.open(**out_meta) as dataset:
                        out_image, out_trans = rasterio.mask.mask(dataset,hu_buff_geom,crop=True)
            else:
                memfile = os.path.join(
                    arguments[8],
                    str(arguments[0]),
                    next(tempfile._get_candidate_names())
                )
                with rasterio.open(memfile,'w+',**out_meta) as dataset:
                    dataset.write(mosaic)
                with rasterio.open(memfile,'r+',**out_meta) as dataset:
                    out_image, out_trans = rasterio.mask.mask(dataset,hu_buff_geom,crop=True)

            out_meta.update({
                "height": out_image.shape[1],
                "width":out_image.shape[2],
                "transform": out_trans
            })

            with rasterio.open(path_elevation,"w",**out_meta) as dst:
                dst.write(out_image)

            return(out_image)

        subdirectory = os.path.join(arguments[5].directory, arguments[6]+'-'+str(arguments[0]))
        print(subdirectory)
        sys.stdout.flush()
        Path(subdirectory).mkdir(parents=True, exist_ok=True)

        path_notime = os.path.join(subdirectory, "jobNoTimeLeftWhileProcessing.err")
        Path(path_notime).touch()

        path_gt1m = os.path.join(subdirectory, "allGT1m.err")
        file_gt1m = Path(path_gt1m)
        path_enclose = os.path.join(subdirectory, "rasterDataDoesNotEnclose.err")
        file_enclose = Path(path_enclose)

        if file_gt1m.is_file() or file_enclose.is_file():

            pass

        else:

            output_nhd(arguments[1],arguments[2],arguments[0])

            path_elevation = os.path.join(subdirectory, 'Elevation.tif')
            file_elevation = Path(path_elevation)
            if file_elevation.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_rasters:

                pass

            else:

                file_elevation.unlink(missing_ok=True)

                break_hu = False

                break_hu, mosaic_tuple = get_mosaic(arguments[4],arguments[0],break_hu,arguments[7])

                if break_hu!=True:

                    raster = arguments[4].dissolve(by=['HUC'])
                    raster.reset_index(inplace=True)
                    raster.drop(
                        columns = {'index','index_left','index_right'},
                        inplace = True,
                        errors = 'ignore'
                    )
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

                    hu_buff = arguments[3].to_crs(mosaic_tuple[1]['crs'])
                    hu_buff.reset_index(inplace=True)
                    hu_buff.drop(
                        columns = {'index','index_left','index_right'},
                        inplace = True,
                        errors = 'ignore'
                    )
                    hu_buff_geom = list(hu_buff['geometry'])

                    out_path = os.path.join(subdirectory, 'hu_buff.shp')
                    my_file = Path(out_path)
                    if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_flowlines:
                        pass
                    else:
                        my_file.unlink(missing_ok=True)
                        hu_buff.to_file(str(out_path))

                    out_path = os.path.join(subdirectory, 'raster.shp')
                    my_file = Path(out_path)
                    if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_flowlines:
                        pass
                    else:
                        my_file.unlink(missing_ok=True)
                        raster.to_file(str(out_path))

                    if len(gpd.sjoin(hu_buff,raster.to_crs(hu_buff.crs),op='within',how='inner').index) == 0:
                        out_path = os.path.join(subdirectory, "rasterDataDoesNotEnclose.err")
                        Path(out_path).touch()
                        print('WARNING: <=1m raster input data does not enclose HUC: '+str(arguments[0]))
                        sys.stdout.flush()
                    else:
                        out_image = output_raster(hu_buff_geom,mosaic_tuple[0],mosaic_tuple[1],path_elevation)
                        return_dict[arguments[0]] = out_image

        Path(path_notime).unlink()

    except OSError as e:
        Path(path_notime).unlink()
        out_path = os.path.join(subdirectory, "OS.err")
        Path(out_path).touch()
        with open(out_path, 'w') as f:
            f.write(str(e))
        print('[ERROR] OSError on HUC: '+str(arguments[0]))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        if arguments[5].log:
            logging.debug('[ERROR] OSError on HUC '+str(arguments[0]))

    except Exception as e:

        print('[EXCEPTION] Exception on HUC: '+str(arguments[0]))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.stdout.flush()
        if arguments[5].log:
            logging.debug('[EXCEPTION] on HUC '+str(arguments[0]))
        return(ExceptionWrapper(e))

    except:
        print('Unexpected error on HUC: '+str(arguments[0]))
        print(sys.exc_info()[0])
        sys.stdout.flush()
        if arguments[5].log:
            logging.debug('[EXCEPTION] on HUC '+str(arguments[0]))
        raise

    else:
        print('Result for HUC: '+str(arguments[0]))
        sys.stdout.flush()
        if arguments[5].log:
            logging.debug('[EXCEPTION] on HUC '+str(arguments[0]))

    finally:
        print('Reached finally clause')
        sys.stdout.flush()
        if arguments[5].log:
            logging.debug('[EXCEPTION] on HUC '+str(arguments[0]))

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
                    flows_keys,
                    flowshu12shape,
                    catchshu12shape,
                    hu12catchs,
                    avail_hu12catchs_grouped
                ) = pickle.load(input)
        else:
            no_restart_file = True

    if not args.restart or no_restart_file:

        hucs = get_hucs_by_shapefile(args.shapefile,args.hucs)
        flowlines,flowline_representative_points = get_flowlines_by_huc(
            hucs,
            args.nhd
        )
        catchments = get_catchments_by_huc(
            hucs,
            args.nhd,
            flowline_representative_points
        )
        flowlines = index_flowlines_by_catchments(
            flowlines,
            catchments
        )

        hucs = catchments.dissolve(by='HUC')
        hucs.reset_index(inplace=True)
        hucs = buffer_hucs(hucs)

        flowlines.to_crs(hucs.crs,inplace=True)
        catchments.to_crs(hucs.crs,inplace=True)

        #lidar_index = index_lidar_files_dev(hucs)
        lidar_index = index_lidar_files(
            hucs,
            args.lidar_availability,
            args.lidar_parent_directory
        )

        ## Ensure lists share the same HUCs
        flows_keys = np.sort(list(
            set(lidar_index['HUC']).intersection(flowlines['HUC'])
        ))
        flows_keys = np.sort(list(
            set(flows_keys).intersection(catchments['HUC'])
        ))
        flows_keys = np.sort(list(
            set(flows_keys).intersection(hucs['HUC'])
        ))
    
        ## Divide into lists per HUC
        flowlines = list(dict(tuple(
            flowlines[
                flowlines['HUC'].isin(flows_keys)
            ].sort_values('HUC').groupby('HUC')
        )).values())
        catchments = list(dict(tuple(
            catchments[
                catchments['HUC'].isin(flows_keys)
            ].sort_values('HUC').groupby('HUC')
        )).values())
        hucs.drop(
            columns = ['index_left','index_right'],
            errors = 'ignore',
            inplace = True
        )
        hucs = list(dict(tuple(
            hucs[
                hucs['HUC'].isin(flows_keys)
            ].sort_index().groupby('HUC')
        )).values())
        lidar_index = list(dict(tuple(
            lidar_index[
                lidar_index['HUC'].isin(flows_keys)
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
        flows_keys = [flows_keys[i] for i in mem_estimates.keys()]
        flowlines = [flowlines[i] for i in mem_estimates.keys()]
        catchments = [catchments[i] for i in mem_estimates.keys()]
        hucs = [hucs[i] for i in mem_estimates.keys()]
        lidar_index = [lidar_index[i] for i in mem_estimates.keys()]

    if args.restart and no_restart_file:
        with open(args.restart, 'wb') as output:
            pickle.dump(
                [
                    flows_keys,
                    flowlines,
                    catchments,
                    hucs,
                    lidar_index
                ],
                output,
                pickle.HIGHEST_PROTOCOL
            )

    start_time = time.time()

    prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
    remove_keys = []
    if (
        not args.overwrite or not (
            args.overwrite_flowlines and
            args.overwrite_catchments and
            args.overwrite_roughnesses and
            args.overwrite_rasters
        )
    ):
        for huc in flows_keys:
            subdirectory = os.path.join(args.directory, prefix+'-'+str(huc))
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
    remove_keys_idcs = [flows_keys.index(key) for key in remove_keys]
    flowlines = [
        flowlines[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
    catchments = [
        catchments[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
    hucs = [
        hucs[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
    lidar_index = [
        lidar_index[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
    flows_keys = [
        flows_keys[key] 
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]

    dst_crs = rasterio.crs.CRS.from_dict(init=hucs.crs)

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
        flows_keys,
        flowlines,
        catchments,
        hucs,
        lidar_index,
        repeat(args),
        repeat(prefix),
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

