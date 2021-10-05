from dem2basin import dem2basin
from itertools import cycle
import gdal
from pyproj import from_crs
from pyproj import Proj
import pyproj
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import os
shapefile = 'HUC12s-Arctur.geojson'
hucs = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs)
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs)
nhd = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/NFIEGeo_TX.gdb'
flowlines,flowline_representative_points = dem2basin.get_flowlines_and_representative_points_by_huc(hucs,nhd)
catchments = dem2basin.get_catchments_by_huc(hucs,nhd,flowline_representative_points)
flowlines = dem2basin.index_dataframe_by_dataframe(flowlines,catchments)
hucs = dem2basin.get_hucs_from_catchments(catchments)
lidar_index_obj = dem2basin.LidarIndex()
lidar_availability = '/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TNRIS-LIDAR-Availability-20210812.shp'
lidar_parent_directory = '/scratch/projects/tnris/tnris-lidardata'
lidar_index = lidar_index_obj.index_lidar_files(hucs,lidar_availability,lidar_parent_directory)
dem2basin.to_crs(hucs.crs,[flowlines,catchments,lidar_index])
huc_ids = dem2basin.get_merged_column('HUC', [lidar_index,flowlines,catchments,hucs])
huc_id = huc_ids[7]
temporary_directory = os.path.join(os.getenv('SCRATCH'),'tmp')
dem2basin.reproject_lidar_tiles_by_huc(lidar_index,temporary_directory,hucs.crs)
huc = hucs.iloc[[7]]
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique()[0]]
subdirectory = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur'
huc_id = 120302010605
subdirectory = os.path.join(subdirectory,'HUC12s-Arctur-'+str(huc_id))
huc_prefix = Path(str(huc['HUC'].unique()[0]))
parent_temporary_directory = temporary_directory
temporary_directory = Path(str(parent_temporary_directory)).joinpath(
        huc_prefix
    )
if not temporary_directory.is_dir():
        temporary_directory.mkdir(parents=True, exist_ok=True)
filenames = lidar_index_by_huc['lidar_file'].to_list()
reprojected_filenames = lidar_index_by_huc['lidar_file'].apply(
        lambda filename: temporary_directory.joinpath(
            Path(str(filename)).name
        )
    ).to_list()
dem2basin.reproject_rasters(filenames,reprojected_filenames,huc.crs)
temporary_vrt_file = temporary_directory.joinpath(
        str(huc_prefix) + '.vrt'
    )
dem2basin.build_vrt(reprojected_filenames,temporary_vrt_file)
temporary_huc_file = temporary_directory.joinpath(
        str(huc_prefix) + '.geojson'
    )
huc.to_file(temporary_huc_file)
dem2basin.reproject_raster(temporary_vrt_file,output_raster_filename,raster_mask_filename = temporary_huc_file)
output_raster_filename = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur/HUC12s-Arctur-120701030702/Elevation.tif'
huc_id = '120701010405'
huc = hucs[hucs['HUC']==huc_id]
parent_output_directory = Path(str(subdirectory)).parent
output_directories = list(parent_output_directory.glob('*'))
output_directories = pd.DataFrame(output_directories)
output_hucs_existing = output_directories[0].apply(lambda fn: fn.name.split('-')[-1])
hucs_file_to_check = '/scratch/04950/dhl/tmp/hucs.geojson'
hucs_file_to_check = Path(hucs_file_to_check)
hucs.to_file('/scratch/04950/dhl/tmp/hucs.geojson',driver='GeoJSON')
hucs_test_file = '/scratch/04950/dhl/tmp/hucs.geojson'
hucs_test_file = Path(hucs_test_file)
catchments_test_file = hucs_test_file.parent.joinpath('catchments.geojson')
catchments.to_file(catchments_test_file)
flowlines_test_file = hucs_test_file.parent.joinpath('flowlines.geojson')
catchments.to_file(catchments_test_file,driver='GeoJSON')
flowlines.to_file(flowlines_test_file,driver='GeoJSON')
lidar_index_test_file = hucs_test_file.parent.joinpath('lidar_index.geojson')
lidar_index.to_file(lidar_index_test_file,driver='GeoJSON')
huc = hucs.iloc[[25]]
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique().shape[0]]
output_raster_filename = Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+str(huc['HUC'].unique()[0])).joinpath('Elevation.tif')
lidar_index_by_huc = lidar_index[lidar_index['HUC']==str(huc['HUC'].unique().shape[0])]
lidar_index['HUC'].unique()
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique()[0]]
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
src_dems_dir = '/scratch/04950/dhl/tmp/src_dems'
src_dems = [Path(str(src_dems_dir)).joinpath(Path(str(filename)).name) for filename in filenames]
[shutil.copyfile(fns[0],fns[1]) for fns in zip(filenames,src_dems)]
filenames = lidar_index_by_huc['lidar_file'].to_list()
src_dems = [Path(str(src_dems_dir)).joinpath(Path(str(filename)).name) for filename in filenames]
[shutil.copyfile(fns[0],fns[1]) for fns in zip(filenames,src_dems)]
p = Proj('EPSG:4326')
lidar_file = '/scratch/projects/tnris/tnris-lidardata/usgs-2017-70cm-brazos-freestone-robertson/dem/usgs17-1m_14RPV690620.img'
lidar_file_obj =  gdal.Open(lidar_file)
projection = lidar_file_obj.GetProjection()
test = lidar_index_by_huc.copy()
test.iloc[test.iloc[-1].name+1]['dirname'] = 'test'
test = test.append({'dirname':'test'},ignore_index=True)
test = lidar_index_by_huc.groupby('dirname')
by_project = [test.get_group(x) for x in test.groups]
dataframe_data = [[project['dirname'].unique()[0],project.shape[0]] for project in by_project]
lidar_projects_with_counts = pd.DataFrame(dataframe_data,columns=['dirname','count'])
lidar_projects_with_counts.sort_values(by=['count'],ascending=False,inplace=True)
lidar_projects_with_info_tile = lidar_index_by_huc.groupby('dirname').first()[['lidar_file']].reset_index()
lidar_projects_with_counts = lidar_projects_with_info_tile.merge(lidar_projects_with_counts,on=['dirname'])
lidar_projects_with_counts.sort_values(
        by = ['count'],
        ascending = False,
        inplace = True
    )
lidar_projects_with_counts['crs'] = lidar_projects_with_counts.iloc[0]['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts['crs'] = lidar_projects_with_counts['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
filenames = lidar_index_by_huc['lidar_file'].to_list()
filenames_repeated = [filenames] * 2
vrts_to_composite = []
for project in lidar_projects_with_counts['dirname'].to_list():
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
filenames_repeated = [filenames] * len(vrts_to_composite)
for filenames_inner,vrts_to_composite_inner in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        dem2basin.build_vrt(filenames_inner,vrts_to_composite_inner)
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
for vrt,reprojected_vrt,crs in zip(
        vrts_to_composite,
        reprojected_vrts_filenames,
        lidar_projects_with_counts['crs'].to_list()
    ):
        dem2basin.reproject_raster(vrt,reprojected_vrt,dst_crs=crs)
temporary_vrt_file = temporary_directory.joinpath(
        str(huc_prefix) + '.vrt'
    )
dem2basin.build_vrt(reprojected_vrts_filenames,temporary_vrt_file)
temporary_huc_file = temporary_directory.joinpath(
        str(huc_prefix) + '.geojson'
    )
huc.to_file(temporary_huc_file)
dem2basin.reproject_raster(
        temporary_vrt_file,
        output_raster_filename,
        raster_mask_filename = temporary_huc_file
    )
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
huc_ids = list(hucs['HUC'].unique())
subdirectories = [Path(str(subdirectory)).parent.joinpath('HUC12s-Arctur-'+str(huc_id)) for huc_id in huc_ids]
flowlines_files = [subdirectory.joinpath('Flowline.shp') for subdirectory in subdirectories]
roughnesses_files = [subdirectory.joinpath('Roughness.csv') for subdirectory in subdirectories]
catchments_files = [subdirectory.joinpath('Catchment.shp') for subdirectory in subdirectories]
elevations_files = [subdirectory.joinpath('Elevation.tif') for subdirectory in subdirectories]
flowlines_by_huc = [huc for _,huc in flowlines.groupby('HUC')]
catchments_by_huc = [huc for _,huc in catchments.groupby('HUC')]
lidar_indices_by_huc = [huc for _,huc in lidar_index.groupby('HUC')]
roughnesses_by_huc = [huc for _,huc in roughnesses.groupby('HUC')]
