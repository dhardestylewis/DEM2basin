## GeoFlood preprocessing 1m DEM data
## Author: Daniel Hardesty Lewis

## Import needed modules
import argparse
import pandas as pd
import fiona
import geopandas as gpd
import utm
from pyproj import CRS
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
results = []
#from multiprocessing import Pool, set_start_method#, Manager
import time
from itertools import repeat
import tblib.pickling_support
tblib.pickling_support.install()
import logging
import psutil
from memory_profiler import profile
import pickle
import gc

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
    ## WBD HUC12 vector data
    parser.add_argument(
        "-u",
        "--huc12",
        type=str,
        help="WBD HUC12 dataset"
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
        "--raster",
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
        "--availability",
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

    args = parser.parse_args()

    ## Check that the required input files have been defined
    if not args.shapefile:
        parser.error('-s --shapefile Input shapefile cutline not specified')
    if not args.huc12:
        parser.error('-u --huc12 Input HUC12 shapefile not specified')
    if not args.nhd:
        parser.error('-n --nhd Input NHD geodatabase not specified')
    if not args.raster:
        parser.error('-r --raster Input raster not specified')
    if not args.availability:
        parser.error('-a --availability Availability shapefile not specified')

    return(args)

def flows_catchs():
    ## Identfy catchments and flowlines to HUCs

    ## Find the HUC12s that intersect with the input polygon
    #shapefile = 'data/TX-Counties/Young/TX-County-Young.shp'
    shape = gpd.read_file(args.shapefile)
    shape.drop(columns=['Shape_Leng','Shape_Area'],inplace=True,errors='ignore')
    shape.rename(columns={'HUC12':'HUC12_shapefile'},inplace=True,errors='ignore')
    #huc12 = 'data/WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp'
    hu12 = gpd.read_file(args.huc12)
    hu12.drop(columns=['AREAACRES','AREASQKM','SHAPE_Leng','SHAPE_Area'],inplace=True,errors='ignore')
    hu12shape = gpd.sjoin(hu12,shape.to_crs(hu12.crs),op='intersects',how='inner')
    #del(hu12,shape)
    hu12shape.drop(columns=['index_right'],inplace=True)

    #nhd = 'data/NFIEGeo_12.gdb'
    ## Find the flowlines whose representative points are within these HUC12s
    flows = gpd.read_file(args.nhd,layer='Flowline')
    flows.drop(columns=['Shape_Length','Shape_Area','AreaSqKM'],inplace=True,errors='ignore')
    flows.reset_index(inplace=True)
    flows.set_index('COMID',inplace=True)
    flows.sort_index(inplace=True)
    flows_rep = flows.copy()
    flows_rep['geometry'] = flows.representative_point()
    flowshu12shape_rep = gpd.sjoin(flows_rep,hu12shape[['HUC12','geometry']].to_crs(flows_rep.crs),op='intersects',how='inner')
    #del(flows_rep,hu12shape)
    flowshu12shape_rep.drop(columns=['index_right'],inplace=True)

    ## Find the catchments corresponding with these flowlines
    catchs = gpd.read_file(args.nhd,layer='Catchment')
    catchs.reset_index(inplace=True)
    catchs.set_index('FEATUREID',inplace=True)
    catchs.sort_index(inplace=True)
    catchshu12shape = catchs[catchs.index.isin(flowshu12shape_rep.index)]
    #del(catchs)
    ## Find the flowlines corresponding with these cactchments
    ##  (Note: this line is optional.
    ##  Commenting it out will result in non-COMID-identified flowlines)
    flowshu12shape = flows[flows.index.isin(catchshu12shape.index)]
    #del(flows)

    ## Determine which HUC12s each of the flowlines and catchments belong to
    flowshu12shape['HUC12'] = flowshu12shape_rep.loc[flowshu12shape.index]['HUC12']
    #del(flowshu12shape_rep)
    flowshu12shape.loc[flowshu12shape['StreamOrde']==0,'Roughness'] = .99
    flowshu12shape.loc[flowshu12shape['StreamOrde']==1,'Roughness'] = .2
    flowshu12shape.loc[flowshu12shape['StreamOrde']==2,'Roughness'] = .1
    flowshu12shape.loc[flowshu12shape['StreamOrde']==3,'Roughness'] = .065
    flowshu12shape.loc[flowshu12shape['StreamOrde']==4,'Roughness'] = .045
    flowshu12shape.loc[flowshu12shape['StreamOrde']==5,'Roughness'] = .03
    flowshu12shape.loc[flowshu12shape['StreamOrde']==6,'Roughness'] = .01
    flowshu12shape.loc[flowshu12shape['StreamOrde']==7,'Roughness'] = .025
    catchshu12shape['HUC12'] = flowshu12shape.loc[catchshu12shape.index]['HUC12']
    flowshu12shape = flowshu12shape[flowshu12shape.is_valid]
    catchshu12shape = catchshu12shape[catchshu12shape.is_valid]
    catchshu12shape = catchshu12shape[catchshu12shape.index.isin(flowshu12shape.index)]
    flowshu12shape = flowshu12shape[flowshu12shape.index.isin(catchshu12shape.index)]

    return(flowshu12shape,catchshu12shape)

def buffer(catchshu12shape):
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

    ## Create new HUC12 boundaries from the catchments pertaining to them
    hu12catchs = catchshu12shape.dissolve(by='HUC12')

    ## Are the catchments all within the same UTM?
    uniq = unique(hu12catchs)
    print(type(uniq))

    ## Buffer the HUC12 catchments
    if hu12catchs.crs.datum.name=='World Geodetic System 1984':
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
    elif hu12catchs.crs.datum.name=='North American Datum 1983' or hu12catchs.crs.datum.name=='D_NORTH_AMERICAN_1983' or hu12catchs.crs.datum.name=='NAD83 (National Spatial Reference System 2011)':
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
    if args.buffer:
        hu12catchs['geometry'] = hu12catchs.to_crs(crs).buffer(args.buffer)
    else:
        hu12catchs['geometry'] = hu12catchs.to_crs(crs).buffer(500.)
    hu12catchs.crs = crs

    ## Are the buffered catchments all within the same UTM?
    unique(hu12catchs)

    return(crs,hu12catchs)

def available(hu12catchs):
    ## Identify each DEM tile file for our study area

    ## Find the DEM tiles that intersect with these buffered HUC12 catchments
    #availibility = 'data/TNRIS-LIDAR-Availability-20191213.shp/TNRIS-LIDAR-Availability-20191213.shp'
    avail = gpd.read_file(args.availability)
    avail_hu12catchs = gpd.sjoin(avail,hu12catchs.to_crs(avail.crs),op='intersects',how='inner')
    #del(avail,hu12catchs)
    ## Construct an exact path for each DEM tile
    fnexts = ['.dem','.img']
    for fnext in fnexts:
        avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
    #del(fnext)
    for dirname in avail_hu12catchs['dirname'].unique():
        stampede2names = []
        #raster = '/scratch/projects/tnris/tnris-lidardata'
        basename = os.path.join(args.raster,dirname,'dem')+os.sep
        for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
        #del(fnext)
        direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
        ## If more than one vector image extension found in a DEM project,
        ##  then figure out each file's extension individually
        ## TODO: Test this against stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro
        if len(direxts) > 1:
            for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                #del(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,x+fnext))
                    if glob.glob(stampede2name.iloc[0]):
                        break
                    #else:
                        #del(stampede2name)
                    #del(fnext)
                avail_hu12catchs.loc[truth,'stampede2name'] = stampede2name
                #del(truth,demname)
        ## Else do all the files in a DEM project at once
        elif len(direxts) == 1:
            stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,x+list(direxts)[0]))
            stampede2name.drop_duplicates(inplace=True)
            p = Path(basename)
            for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
            stampede2name = stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])]
            avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] = stampede2name
        else:
            continue
        #del(stampede2names,stampede2name,basename,direxts,dirname)
    #del(fnexts)
    avail_hu12catchs.dropna(subset=['stampede2name'],inplace=True)
    avail_hu12catchs_grouped = avail_hu12catchs.groupby('index_right')
    #del(avail_hu12catchs)

    return(avail_hu12catchs_grouped)

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise(self.ee.with_traceback(self.tb))

#@profile
def output_files(arguments):
#def output(flow_key,flowshu12shape,catchshu12shape,hu12catchs,avail_hu12catchs_group,args,prefix,dst_crs,mem_estimates):
    ## Output catchments, flowlines, roughnesses, and rasters

    try:

        def output_nhd(flowshu12shape,catchshu12shape,hu):
            ## For each HUC, write the flowlines, catchments, and roughnesses corresponding to it

            #print('INSIDE OUTPUT_NHD\t',arguments[0])
            #print('INSIDE OUTPUT_NHD\t',flows_key)
            #sys.stdout.flush()
            out_path = os.path.join(subdirectory, 'Flowlines.shp')
            my_file = Path(out_path)
            #if my_file.is_file() and not arguments[1].args.overwrite and not arguments[1].args.overwrite_flowlines:
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_flowlines:
            #if my_file.is_file() and not args.overwrite and not args.overwrite_flowlines:
                #del(my_file)
                pass
            else:
                my_file.unlink(missing_ok=True)
                #del(my_file)
                #flowshu12shape[flowshu12shape['HUC12']==hu].reset_index().to_file(out_path)
                #print('OUTPUTTING FLOWS\t',arguments[0])
                #print('OUTPUTTING FLOWS\t',flows_key)
                #sys.stdout.flush()
                flowshu12shape.reset_index().to_file(out_path)
            #del(out_path)

            out_path = os.path.join(subdirectory, 'Roughness.csv')
            my_file = Path(out_path)
            #if my_file.is_file() and not arguments[1].args.overwrite and not arguments[1].args.overwrite_roughnesses:
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_roughnesses:
            #if my_file.is_file() and not args.overwrite and not args.overwrite_roughnesses:
                #del(my_file)
                pass
            else:
                my_file.unlink(missing_ok=True)
                #del(my_file)
                with open(out_path, 'w', newline='') as outcsv:
                    #print('OUTPUTTING ROUGHS\t',arguments[0])
                    #print('OUTPUTTING ROUGHS\t',flows_key)
                    #sys.stdout.flush()
                    writer = csv.writer(outcsv)
                    writer.writerow(['COMID','StreamOrde','Roughness'])
                    for comid in np.sort(flowshu12shape.index.unique()):
                        writer.writerow([comid,flowshu12shape.loc[comid,'StreamOrde'],flowshu12shape.loc[comid,'Roughness']])
                    #del(comid,writer)
            #del(out_path,flowshu12shape)

            out_path = os.path.join(subdirectory, 'Catchments.shp')
            my_file = Path(out_path)
            #if my_file.is_file() and not arguments[1].args.overwrite and not arguments[1].args.overwrite_catchments:
            if my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_catchments:
            #if my_file.is_file() and not args.overwrite and not args.overwrite_catchments:
                #del(my_file)
                pass
            else:
                my_file.unlink(missing_ok=True)
                #del(my_file)
                #catchshu12shape[catchshu12shape['HUC12']==hu].reset_index().to_file(out_path)
                #print('OUTPUTTING CATCHS\t',arguments[0])
                #print('OUTPUTTING CATCHS\t',flows_key)
                #sys.stdout.flush()
                catchshu12shape.reset_index().to_file(out_path)
            #del(out_path,catchshu12shape)

        def get_mosaic(avail_hu12catchs_group,hu,break_hu,dst_crs):
            ## Get mosaic of DEMs for each HUC

            def append_check(src_files_to_mosaic,var,subdirectory,hu):
                ## Check each raster's resolution in this HUC

                if any(np.float16(i) > 1. for i in var.res):
                    out_path = os.path.join(subdirectory, "gt1m.err")
                    Path(out_path).touch()
                    print('WARNING: >1m raster input for HUC12: '+str(hu))
                    sys.stdout.flush()
                    #del(out_path)
                else:
                    src_res_min_to_mosaic.append(min(var.res))
                    src_res_max_to_mosaic.append(min(var.res))
                    src_x_to_mosaic.append(var.res[0])
                    src_y_to_mosaic.append(var.res[1])
                    src_files_to_mosaic.append(var)

                return(src_files_to_mosaic,src_res_min_to_mosaic,src_res_max_to_mosaic,src_x_to_mosaic,src_y_to_mosaic)

            ## Reproject the mosaic to DEM tiles pertaining to each HUC
            #print('INSIDE GET MOSAIC\t',arguments[0])
            #print('INSIDE GET MOSAIC\t',flows_key)
            #sys.stdout.flush()
            dem_fps = list(avail_hu12catchs_group['stampede2name'])
            src_files_to_mosaic = []
            src_res_min_to_mosaic = []
            src_res_max_to_mosaic = []
            src_x_to_mosaic = []
            src_y_to_mosaic = []
            memfile = {}
            for fp in dem_fps:
                memfile[fp] = MemoryFile()
                #del(fp)
            for fp in dem_fps:
                with rasterio.open(fp) as src:
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
                    #del(width,height)

                    ## Don't do an expensive reprojection if projection already correct
                    if src.meta==out_meta:
                        #del(fp,transform)
                        src_files_to_mosaic, src_res_min_to_mosaic, src_res_max_to_mosaic, src_x_to_mosaic, src_y_to_mosaic = append_check(src_files_to_mosaic,src,subdirectory,hu)
                    else:
                        dst = memfile[fp].open(**out_meta)
                        #del(fp,out_meta)
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
                            #del(i)
                        #del(transform)
                        src_files_to_mosaic, src_res_min_to_mosaic, src_res_max_to_mosaic, src_x_to_mosaic, src_y_to_mosaic = append_check(src_files_to_mosaic,dst,subdirectory,hu)
                        #del(dst)
            #del(dem_fps)

            if len(src_files_to_mosaic) == 0:

                out_path = os.path.join(subdirectory, "allGT1m.err")
                Path(out_path).touch()
                print('WARNING: Found no <=1m raster input data for HUC12: '+str(hu))
                sys.stdout.flush()
                #del(out_path)

                break_hu = True
                mosaic_tuple = ()
                return(break_hu,mosaic_tuple)

            else:

                src_files_to_mosaic = pd.DataFrame(data={
                    'Files':src_files_to_mosaic,
                    'min(resolution)':src_res_min_to_mosaic,
                    'max(resolution)':src_res_max_to_mosaic
                })
                #del(src_res_min_to_mosaic,src_res_max_to_mosaic)
                src_files_to_mosaic.sort_values(by=['min(resolution)','max(resolution)'],inplace=True)
                mosaic, out_trans = merge(list(src_files_to_mosaic['Files']),res=(max(src_x_to_mosaic),max(src_y_to_mosaic)))
                #del(src_x_to_mosaic,src_y_to_mosaic)
                for src in src_files_to_mosaic['Files']:
                    src.close()
                #del(src_files_to_mosaic)
                out_meta = src.meta.copy()
                #del(src)
                out_meta.update({
                    "driver": 'GTiff',
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": dst_crs
                })
                #del(out_trans)
                for keyvalue in memfile.items():
                    keyvalue[1].close()
                    #del(keyvalue)
                #del(memfile)

                mosaic_tuple = (mosaic,out_meta)
                return(break_hu,mosaic_tuple)

        def output_raster(hu_buff_geom,mosaic,out_meta,path_elevation):
            ## Crop and output the mosaic to the buffered catchments of each HUC

            #print('INSIDE OUTPUT RASTER\t',arguments[0])
            #print('INSIDE OUTPUT RASTER\t',flows_key)
            #sys.stdout.flush()
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as dataset:
                    dataset.write(mosaic)
                with memfile.open(**out_meta) as dataset:
                    out_image, out_trans = rasterio.mask.mask(dataset,hu_buff_geom,crop=True)
            #del(hu_buff_geom)

            out_meta.update({
                "height": out_image.shape[1],
                "width":out_image.shape[2],
                "transform": out_trans
            })
            #del(out_trans)

            with rasterio.open(path_elevation,"w",**out_meta) as dst:
                dst.write(out_image)
            #del(path_elevation,out_meta,out_image,dst)

        #print('INSIDE\t',arguments[0])
        #print('INSIDE\t',flows_key)
        #sys.stdout.flush()
        #subdirectory = os.path.join(arguments[1].args.directory, arguments[1].prefix+'-'+str(arguments[0]))
        subdirectory = os.path.join(arguments[5].directory, arguments[6]+'-'+str(arguments[0]))
        #subdirectory = os.path.join(args.directory, prefix+'-'+str(flow_key))
        Path(subdirectory).mkdir(parents=True, exist_ok=True)

        path_notime = os.path.join(subdirectory, "jobNoTimeLeftWhileProcessing.err")
        Path(path_notime).touch()

        path_gt1m = os.path.join(subdirectory, "allGT1m.err")
        file_gt1m = Path(path_gt1m)
        path_enclose = os.path.join(subdirectory, "rasterDataDoesNotEnclose.err")
        file_enclose = Path(path_enclose)

        if file_gt1m.is_file() or file_enclose.is_file():

            #del(file_gt1m,path_gt1m,file_enclose,path_enclose)
            pass

        else:

            #print('GOING IN OUTPUT_NHD\t',arguments[0])
            #print('GOING IN OUTPUT_NHD\t',flows_key)
            #sys.stdout.flush()
            #output_nhd(arguments[1].flowshu12shape,arguments[1].catchshu12shape,arguments[0])
            output_nhd(arguments[1],arguments[2],arguments[0])
            #output_nhd(flowshu12shape,catchshu12shape,flow_key)

            path_elevation = os.path.join(subdirectory, 'Elevation.tif')
            file_elevation = Path(path_elevation)
            #if file_elevation.is_file() and not arguments[1].args.overwrite and not arguments[1].args.overwrite_rasters:
            if file_elevation.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_rasters:
            #if file_elevation.is_file() and not args.overwrite and not args.overwrite_rasters:

                #del(file_elevation,path_elevation)
                pass

            else:

                file_elevation.unlink(missing_ok=True)
                #del(file_elevation)

                #avail_hu12catchs_group = arguments[1].avail_hu12catchs_grouped.get_group(arguments[0])
                break_hu = False

                #print('GOING IN GET MOSAIC\t',arguments[0])
                #print('GOING IN GET MOSAIC\t',flows_key)
                #sys.stdout.flush()
                #break_hu, mosaic_tuple = get_mosaic(avail_hu12catchs_group,arguments[0],break_hu,arguments[1].dst_crs)
                break_hu, mosaic_tuple = get_mosaic(arguments[4],arguments[0],break_hu,arguments[7])
                #break_hu, mosaic_tuple = get_mosaic(avail_hu12catchs_group,flow_key,break_hu,dst_crs)

                if break_hu!=True:

                    with rasterio.Env():
                        results = ({
                            'properties': {
                                'Elevation': v
                            },
                            'geometry': s
                        }
                        for i, (s, v) in enumerate(shapes(
                            (mosaic_tuple[0]==mosaic_tuple[1]['nodata']).astype(np.int16),
                            mask=mosaic_tuple[0]!=mosaic_tuple[1]['nodata'],
                            transform=mosaic_tuple[1]['transform']
                        )))
                    geoms = list(results)
                    raster = gpd.GeoDataFrame.from_features(geoms,crs=mosaic_tuple[1]['crs'])

                    #hu_buff = arguments[1].hu12catchs.loc[[arguments[0]]].drop(columns=['index_left','index_right'],errors='ignore').to_crs(mosaic_tuple[1]['crs'])
                    hu_buff = arguments[3].to_crs(mosaic_tuple[1]['crs'])
                    #hu_buff = hu12catchs.to_crs(mosaic_tuple[1]['crs'])
                    hu_buff_geom = list(hu_buff['geometry'])

                    if len(gpd.sjoin(hu_buff,raster,op='within',how='inner').index) == 0:
                        out_path = os.path.join(subdirectory, "rasterDataDoesNotEnclose.err")
                        Path(out_path).touch()
                        print('WARNING: <=1m raster input data does not enclose HUC12: '+str(arguments[0]))
                        #print('WARNING: <=1m raster input data does not enclose HUC12: '+str(flow_key))
                        sys.stdout.flush()
                        #del(out_path)
                    else:
                        #print('GOING IN OUTPUT RASTER\t',arguments[0])
                        #print('GOING IN OUTPUT RASTER\t',flows_key)
                        output_raster(hu_buff_geom,mosaic_tuple[0],mosaic_tuple[1],path_elevation)

        #print('ZEROING MEM_ESTIMATE\t',arguments[0])
        #print('ZEROING MEM_ESTIMATE\t',flows_key)
        #sys.stdout.flush()
        #mem_estimates[flows_key] = 0.
        Path(path_notime).unlink()
        #del(path_notime,subdirectory)

    except OSError as e:
        Path(path_notime).unlink()
        out_path = os.path.join(subdirectory, "OS.err")
        Path(out_path).touch()
        with open(out_path, 'w') as f:
            #f.write("{}".format(e))
            f.write(str(e))
        print('[ERROR] OSError on HUC12: '+str(arguments[0]))
        print(e)
        sys.stdout.flush()
        #if arguments[1].args.log:
        if arguments[5].log:
        #if args.log:
            logging.debug('[ERROR] OSError on HUC '+str(arguments[0]))
            #logging.debug('HUC '+str(flow_key))

    except Exception as e:

        #if arguments[1].args.log:
        if arguments[5].log:
        #if args.log:
            logging.debug('[EXCEPTION] on HUC '+str(arguments[0]))
            #logging.debug('HUC '+str(flow_key))
        return(ExceptionWrapper(e))

    #except:
    #    print(sys.exc_info()[0])
    #    raise

def collect_results(result):
    results.append(result)

class TaskProcessor(Thread):
    """
    Processor class which monitors memory usage for running tasks (processes).
    Suspends execution for tasks surpassing `max_b` and completes them one
    by one, after behaving tasks have finished.
    """

    def __init__(self, n_cores, max_b, tasks):
        super().__init__()
        self.n_cores = n_cores
        self.max_b = max_b
        self.tasks = deque(tasks)

        self._running_tasks = []
        self._suspended_tasks = []

    def run(self):
        """Main-function in new thread."""
        self._update_running_tasks()
        self._monitor_running_tasks()
        self._update_suspended_tasks()
        self._monitor_suspended_tasks()
#        self._process_suspended_tasks()

    def _update_running_tasks(self):
        """Start new tasks if we have less running tasks than cores."""
        while len(self._running_tasks) < self.n_cores and len(self.tasks) > 0:
            p = self.tasks.popleft()
            gc.collect()
            p.start()
            # for further process-management we here just need the
            # psutil.Process wrapper
            self._running_tasks.append(psutil.Process(pid=p.pid))
            print(f'Started process: {self._running_tasks[-1]}')

    def _monitor_running_tasks(self):
        """
        Monitor running tasks. Replace completed tasks and suspend tasks
        which exceed the memory threshold `self.max_b`.
        """
        # loop while we have running or non-started tasks
        while self._running_tasks or self.tasks:
            multiprocessing.active_children() # Joins all finished processes.
            # Without it, p.is_running() below on Unix would not return
            # `False` for finished processes.
            self._update_running_tasks()
            actual_tasks = self._running_tasks.copy()

            for p in actual_tasks:
                if not p.is_running():  # process has finished
                    self._running_tasks.remove(p)
                    print(f'Removed finished process: {p}')
                else:
                    if p.memory_info().rss > psutil.virtual_memory().available - self.max_b*.1:
                        p.suspend()
                        self._running_tasks.remove(p)
                        self._suspended_tasks.append(p)
                        print(f'Suspended process: {p}')

            time.sleep(.005)

    def _update_suspended_tasks(self):
        """Start new tasks if we have less running tasks than cores."""
        while len(self._running_tasks) < self.n_cores and len(self._suspended_tasks) > 0:
            p = self._suspended_tasks.popleft()
            gc.collect()
            p.resume()
            # for further process-management we here just need the
            # psutil.Process wrapper
            self._running_tasks.append(p)
            print(f'Resumed process: {self._running_tasks[-1]}')

    def _monitor_suspended_tasks(self):
        """
        Monitor running tasks. Replace completed tasks and suspend tasks
        which exceed the memory threshold `self.max_b`.
        """
        # loop while we have running or non-started tasks
        while self._running_tasks or self._suspended_tasks:
            multiprocessing.active_children() # Joins all finished processes.
            # Without it, p.is_running() below on Unix would not return
            # `False` for finished processes.
            self._update_suspended_tasks()
            actual_tasks = self._running_tasks.copy()

            for p in actual_tasks:
                if not p.is_running():  # process has finished
                    self._running_tasks.remove(p)
                    print(f'Removed finished process: {p}')
                else:
                    if p.memory_info().rss > psutil.virtual_memory().available - self.max_b*.1:
                        p.suspend()
                        self._running_tasks.remove(p)
                        self._suspended_tasks.append(p)
                        print(f'Suspended process: {p}')

            time.sleep(.005)

#    def _process_suspended_tasks(self):
#        """Resume processing of suspended tasks."""
#        for p in self._suspended_tasks:
#            p.resume()
#            print(f'\nResumed process: {p}')
#            p.wait()

def main():

    oldgdal_data = os.environ['GDAL_DATA']
    os.environ['GDAL_DATA'] = os.path.join(fiona.__path__[0],'gdal_data')

    global args

    args = argparser()

    ## TODO: Also a shared file, potentially causing deadlock
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG)

    #ns.args = args

    no_restart_file = False
    if args.restart:
        my_file = Path(args.restart)
        if my_file.is_file():
            with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
        else:
            no_restart_file = True

    if not args.restart or no_restart_file:

        flowshu12shape,catchshu12shape = flows_catchs()
        crs,hu12catchs = buffer(catchshu12shape)
        flowshu12shape.to_crs(crs,inplace=True)
        catchshu12shape.to_crs(crs,inplace=True)
        avail_hu12catchs_grouped = available(hu12catchs)
    
        ## Ensure lists share the same HUC12s
        flows_keys = np.sort(list(set(avail_hu12catchs_grouped.groups.keys()).intersection(flowshu12shape['HUC12'])))
        flows_keys = np.sort(list(set(flows_keys).intersection(catchshu12shape['HUC12'])))
        flows_keys = np.sort(list(set(flows_keys).intersection(hu12catchs.index)))
    
        ## Divide into lists per HUC12
        flowshu12shape = list(dict(tuple(flowshu12shape[flowshu12shape['HUC12'].isin(flows_keys)].sort_values('HUC12').groupby('HUC12'))).values())
        #flowshu12shape = dict(tuple(flowshu12shape[flowshu12shape['HUC12'].isin(flows_keys)].sort_values('HUC12').groupby('HUC12')))
        catchshu12shape = list(dict(tuple(catchshu12shape[catchshu12shape['HUC12'].isin(flows_keys)].sort_values('HUC12').groupby('HUC12'))).values())
        #catchshu12shape = dict(tuple(catchshu12shape[catchshu12shape['HUC12'].isin(flows_keys)].sort_values('HUC12').groupby('HUC12')))
        hu12catchs.drop(columns=['index_left','index_right'],errors='ignore',inplace=True)
        hu12catchs = list(dict(tuple(hu12catchs[hu12catchs.index.isin(flows_keys)].sort_index().groupby('HUC12'))).values())
        #hu12catchs = dict(tuple(hu12catchs[hu12catchs.index.isin(flows_keys)].sort_index().groupby('HUC12')))
        avail_hu12catchs_grouped = list({k:dict(tuple(avail_hu12catchs_grouped))[k] for k in flows_keys}.values())
        #avail_hu12catchs_grouped = {k:dict(tuple(avail_hu12catchs_grouped))[k] for k in flows_keys}
    
        ## Sort lists by estimated memory usage
        mem_estimates = {}
        for i in range(len(avail_hu12catchs_grouped)):
            mem_estimates[i] = avail_hu12catchs_grouped[i]['stampede2name'].apply(lambda x: Path(x).stat().st_size).sum()
        mem_estimates = {k: v for k, v in sorted(mem_estimates.items(), key=lambda item: item[1])}
        mem_estimates = {k: v for k, v in mem_estimates.items() if v < psutil.virtual_memory().total}
        flows_keys = [flows_keys[i] for i in mem_estimates.keys()]
        flowshu12shape = [flowshu12shape[i] for i in mem_estimates.keys()]
        catchshu12shape = [catchshu12shape[i] for i in mem_estimates.keys()]
        hu12catchs = [hu12catchs[i] for i in mem_estimates.keys()]
        avail_hu12catchs_grouped = [avail_hu12catchs_grouped[i] for i in mem_estimates.keys()]

    if args.restart and no_restart_file:
        with open(args.restart, 'wb') as output:
            pickle.dump(
                [flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs],
                output,
                pickle.HIGHEST_PROTOCOL
            )

    start_time = time.time()

    prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
    #ns.dst_crs = rasterio.crs.CRS.from_wkt(ns.crs.to_wkt())
    dst_crs = rasterio.crs.CRS.from_dict(init=crs)
    #del(crs)

    #multiprocessing.set_start_method('forkserver')
    multiprocessing.set_start_method('spawn')

    MAX_B = psutil.virtual_memory().total
    N_CORES = multiprocessing.cpu_count()-1

    ## Run the output functions for each of the HUCs
    #mgr = multiprocessing.Manager()
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    #pool = multiprocessing.Pool(processes=1)
    #mem_estimates = mgr.dict()
#    keys_remaining = set(flows_keys)
#    while len(keys_remaining) > 0:
#        keys_remove = set()
#        for key in keys_remaining:
#            time.sleep(1./12.)
#            mem_estimates[key] = avail_hu12catchs_grouped[key]['stampede2name'].apply(lambda x: Path(x).stat().st_size).sum()*7.*1.1
#            running_estimate = sum(mem_estimates.values())
#            if running_estimate > psutil.virtual_memory().total*.9:
#                mem_estimates[key] = 0.
#                continue
#            subtraction = psutil.virtual_memory().available - psutil.virtual_memory().total*.1 - running_estimate
#            print(key,'\t',mem_estimates[key],'\t',running_estimate,'\t',subtraction)
#            #if psutil.virtual_memory().available - psutil.virtual_memory().total*.1 - mem_estimate > 0.:
#            if psutil.virtual_memory().available - psutil.virtual_memory().total*.1 - running_estimate > 0.:
#                arguments = [key, flowshu12shape[key], catchshu12shape[key], hu12catchs[key], avail_hu12catchs_grouped[key], args, prefix, dst_crs, mem_estimates]
#                #pool.apply_async(output, args=arguments[i], callback=collect_results)
#                print('GOING IN\t',key)
#                pool.apply_async(output, args=arguments, callback=collect_results)
#                keys_remove.add(key)
#            else:
#                mem_estimates[key] = 0.
#        keys_remaining -= keys_remove
    #results = pool.imap_unordered(output,zip(flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, repeat(args), repeat(prefix), repeat(dst_crs)))
    arguments = zip(flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, repeat(args), repeat(prefix), repeat(dst_crs))
    tasks = [multiprocessing.Process(target=output_files, args=(argument,)) for argument in arguments]
    pool = TaskProcessor(n_cores=N_CORES, max_b=MAX_B, tasks = tasks)
    pool.start()
    #pool.close()
    pool.join()
    #print(results)
    #del(prefix,dst_crs)
    #for result in results:
    #    if isinstance(result, ExceptionWrapper):
    #        result.re_raise()

    print("All catchments, flowlines, roughnesses, and rasters created for each HUC")
    print("Time spent with ", N_CORES, " threads in milliseconds")
    print("-----", int((time.time()-start_time)*1000), "-----")

    os.environ['GDAL_DATA'] = oldgdal_data

if __name__ == "__main__":
    main()

