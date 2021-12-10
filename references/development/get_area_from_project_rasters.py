import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
import re
import rasterio
import os
from pathlib import Path
project_study_area = 'stratmap-2012-50cm-tceq-dam-sites'
tnris_lidar_parent_dir = '/scratch/projects/tnris/tnris-lidardata'
project_study_area_path = Path(tnris_lidar_parent_dir).joinpath(project_study_area)
project_study_area_path = project_study_area_path.joinpath('dem')
project_study_area_files = list(project_study_area_path.glob('*'))
project_study_area_files = pd.DataFrame(project_study_area_files,columns=['filename'])
project_study_area_rasters = project_study_area_files[project_study_area_files['filename'].apply(lambda fn : os.path.splitext(str(fn))[1]=='.img')]
project_study_area_rasters.reset_index(drop=True,inplace=True)
project_study_area_test_raster = project_study_area_rasters.loc[0,'filename']
project_study_area_test_raster = rasterio.open(project_study_area_test_raster)
dst_crs = 'EPSG:6579'
transform, width, height = calculate_default_transform(project_study_area_test_raster.crs,dst_crs,project_study_area_test_raster.width,project_study_area_test_raster.height,*project_study_area_test_raster.bounds)
kwargs = project_study_area_test_raster.meta.copy()
kwargs.update({'crs':dst_crs,'transform':transform,'width':width,'height':height})
nodata_before_reproj = (przoject_study_area_test_raster.read(1)==project_study_area_test_raster.nodata).sum()
data_before_reproj = (project_study_area_test_raster.read(1)!=project_study_area_test_raster.nodata).sum()
dst_fn = Path(os.getenv('SCRATCH')).joinpath('project_study_area_test_raster_reprojected.img')
dst = rasterio.open(dst_fn,'w',**kwargs)
reproject(source=rasterio.band(project_study_area_test_raster,1),destination=rasterio.band(dst,1),src_transform=project_study_area_test_raster.transform,src_crs=project_study_area_test_raster.crs,dst_transform=transform,dst_crs=dst_crs,resampling=Resampling.nearest)
dst.close()
dst = rasterio.open(dst_fn,'r')
nodata_after_reproj = (dst.read(1)==dst.nodata).sum()
data_after_reproj = (dst.read(1)!=dst.nodata).sum()
data_after_reproj*dst.res[0]*dst.res[1]
