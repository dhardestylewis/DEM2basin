from dem2basin import dem2basin
import os
hucs = get_hucs_by_shape(args.shapefile,args.hucs)
hucs = dem2basin.get_hucs_by_shape(args.shapefile,args.hucs)
hucs = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
hucs = get_hucs_by_shape(shapefile,hucs)
hucs = dem2basinget_hucs_by_shape(shapefile,hucs)
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs)
shapefile
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
shapefile
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs)
nhd = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/NFIEGeo_TX.gdb'
nhd
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
nhd
import geopandas as gpd
nhd_gdf = gpd.read_file(nhd)
nhd_gdf
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
from importlib import reload
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
catchments = get_catchments_by_huc(
            hucs,
            args.nhd,
            flowline_representative_points
        )
catchments = dem2basin.get_catchment_by_huc(hucs,nhd,flowline_representative_points)
catchments = dem2basin.get_catchments_by_huc(hucs,nhd,flowline_representative_points)
flowlines = dem2basin.index_dataframe_by_dataframe(flowlines,catchments)
hucs = get_hucs_from_catchments(catchments)
hucs = dem2basin.get_hucs_from_catchments(catchments)
hucs
lidar_index_obj = LidarIndex()
lidar_index_obj = dem2basin.LidarIndex()
lidar_index = lidar_index_obj.index_lidar_files(hucs,,)
lidar_availability = '/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TNRIS-LIDAR-Availability-20210812.shp'
lidar_parent_directory = '/scratch/projects/tnris/tnris-lidardata'
lidar_index = lidar_index_obj.index_lidar_files(hucs,lidar_availability,lidar_parent_directory)
lidar_index
to_crs(hucs.crs,[flowlines,catchments,lidar_index])
dem2basin.to_crs(hucs.crs,[flowlines,catchments,lidar_index])
huc_ids = dem2basin.get_merged_column('HUC', [lidar_index,flowlines,catchments,hucs])
huc_ids
lidar_index['HUC']
lidar_index['HUC'].unique()
ls
huc_id = huc_ids[7]
hucs
hucs.columns
filenames
lidar_index
lidar_index.columns
reload(dem2basin)
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,dst_crs)
temporary_directory = os.getenv('SCRATCH')
temporary_directory
temporary_directory = os.path.join(os.getenv('SCRATCH'),'tmp')
temporary_directory
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,dst_crs)
dst_crs
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,hucs.crs)
reload(dem2basin
reload(dem2basin)
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,hucs.crs)
reload(dem2basin)
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,hucs.crs)
5000./30.
166
166/5.
dir()
huc_id
hucs
hucs['HUC'].unique()[0]
lidar_index
lidar_index.columns
lidar_index['lidar_file'].unique()
lidar_index['lidar_file'].unique().shape[0]
reload(dem2basin)
huc
hucs
huc = hucs[7]
huc = hucs.iloc[7]
huc
huc = hucs.iloc[[7]]
huc
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique()[0]]
lidar_index_by_huc
subdirecotry
subdirectory
subdirectory = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur'
subdirectory = os.path.join(args.directory, output_prefix+'-'+str(huc))
subdirectory = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur'
hucs
huc
huc_id = 120302010605
subdirectory = os.path.join(subdirectory,'HUC12s-Arctur-'+str(huc_id))
subdirectory
huc_prefix = Path(str(huc['HUC'].unique()[0]))
from pathlib import Path
huc_prefix = Path(str(huc['HUC'].unique()[0]))
temporary_directory = Path(str(parent_temporary_directory)).joinpath(
        huc_prefix
    )
parent_temporary_directory = temporary_directory
temporary_directory
temporary_directory = Path(str(parent_temporary_directory)).joinpath(
        huc_prefix
    )
temporary_directory
if not temporary_directory.is_dir():
        temporary_directory.mkdir(parents=True, exist_ok=True)
filenames = lidar_index_by_huc['lidar_file'].to_list()
filenames
reprojected_filenames = lidar_index_by_huc['lidar_file'].apply(
        lambda filename: temporary_directory.joinpath(
            Path(str(filename)).name
        )
reprojected_filenames = lidar_index_by_huc['lidar_file'].apply(
        lambda filename: temporary_directory.joinpath(
            Path(str(filename)).name
        )
    ).to_list()
reproject_rasters(filenames,reprojected_filenames,huc.crs)
dem2basin.reproject_rasters(filenames,reprojected_filenames,huc.crs)
reload(dem2basin)
dem2basin.reproject_rasters(filenames,reprojected_filenames,huc.crs)
reload(dem2basin)
dem2basin.reproject_rasters(filenames,reprojected_filenames,huc.crs)
temporary_vrt_file = temporary_directory.joinpath(
        huc_prefix + '.vrt'
    )
temporary_vrt_file = temporary_directory.joinpath(
        str(huc_prefix) + '.vrt'
    )
build_vrt(reprojected_filenames,temporary_vrt_file)
dem2basin.build_vrt(reprojected_filenames,temporary_vrt_file)
reload(dem2basin)
dem2basin.build_vrt(reprojected_filenames,temporary_vrt_file)
temporary_huc_file = temporary_directory.joinpath(
        str(huc_prefix) + '.geojson'
    )
huc.to_file(temporary_huc_file)
reproject_raster(
        temporary_vrt_file,
        output_raster_filename,
        raster_mask_filename = temporary_huc_file
    )
dem2basin.reproject_raster(temporary_vrt_file,output_raster_filename,raster_mask_filename = temporary_huc_file)
huc_id
output_raster_filename = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur/HUC12s-Arctur-120701030702/Elevation.tif'
huc_id = 120701030702
huc = hucs[hucs['HUC']==huc_id]
huc
huc = hucs[hucs['HUC']==str(huc_id)]
huc
hucs
huc = hucs[hucs['HUC']==str(120701010305)]
huc
huc = hucs[hucs['HUC']==120701010305]
huc
huc_id = '120701010405'
huc_id
huc = hucs[hucs['HUC']==huc_id]
huc
subdirectory
dir()
parent_output_directory = Path(str(subdirectory)).parent
parent_output_directory
parent_output_directory.glob('*')
list(parent_output_directory.glob('*'))
output_directories = list(parent_output_directory.glob('*'))
output_directories
output_directories = pd.DataFrame(output_directories)
import pandas as pd
output_directories = pd.DataFrame(output_directories)
output_directories
output_directories.apply(lambda fn: fn.name)
output_directories[0].apply(lambda fn: fn.name)
output_directories[0].apply(lambda fn: fn.name.split('-'))
output_directories[0].apply(lambda fn: fn.name.split('-')[-1])
output_hucs_existing = output_directories[0].apply(lambda fn: fn.name.split('-')[-1])
hucs
hucs['HUC']
hucs['HUC'].isin(output_hucs_existing)
hucs['HUC'].astype(int64).isin(output_hucs_existing)
hucs['HUC'].astype(np.int64).isin(output_hucs_existing)
import numpy as np
hucs['HUC'].astype(np.int64).isin(output_hucs_existing)
hucs['HUC'].astype(str).isin(output_hucs_existing)
hucs
huc
dir()
import readline
os.getcwd()
readline.write_history_file('troubleshooting-history.py')
