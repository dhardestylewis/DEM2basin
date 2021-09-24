exit()
61201 * 43201
exit() 
exit()
import flopy
ls
exit()
import requests
import json
import uuid
import pprint
import datetime
pp = pprint.PrettyPrinter(indent=2)
def handle_api_response(response, print_response=False):
    parsed_response = response.json()
    if print_response:
        pp.pprint({"API Response": parsed_response})
    
    if response.status_code == 200:
        return parsed_response
    elif response.status_code == 400:
        raise Exception("Bad request ^")
    elif response.status_code == 403:
        msg = "Please make sure your request headers include X-Api-Key and that you are using correct url"
        raise Exception(msg)
    else:
        now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        msg = f"""\n\n
        ------------------------------------- BEGIN ERROR MESSAGE -----------------------------------------
        It seems our server encountered an error which it doesn't know how to handle yet. 
        This sometimes happens with unexpected input(s). In order to help us diagnose and resolve the issue, 
        could you please fill out the following information and email the entire message between ----- to
        danf@usc.edu:
        1) URL of notebook (of using the one from https://hub.mybinder.org/...): [*****PLEASE INSERT ONE HERE*****]
        2) Snapshot/picture of the cell that resulted in this error: [*****PLEASE INSERT ONE HERE*****]
        
        Thank you and we apologize for any inconvenience. We'll get back to you as soon as possible!
        
        Sincerely, 
        Dan Feldman
        
        Automatically generated summary:
        - Time of occurrence: {now}
        - Request method + url: {response.request.method} - {response.request.url}
        - Request headers: {response.request.headers}
        - Request body: {response.request.body}
        - Response: {parsed_response}
        --------------------------------------- END ERROR MESSAGE ------------------------------------------
        \n\n
        """
def handle_api_response(response, print_response=False):
    parsed_response = response.json()
    if print_response:
        pp.pprint({"API Response": parsed_response})
    if response.status_code == 200:
        return parsed_response
    elif response.status_code == 400:
        raise Exception("Bad request ^")
    elif response.status_code == 403:
        msg = "Please make sure your request headers include X-Api-Key and that you are using correct url"
        raise Exception(msg)
    else:
        now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        msg = f"""\n\n
        ------------------------------------- BEGIN ERROR MESSAGE -----------------------------------------
        It seems our server encountered an error which it doesn't know how to handle yet.
        This sometimes happens with unexpected input(s). In order to help us diagnose and resolve the issue,
        could you please fill out the following information and email the entire message between ----- to
        danf@usc.edu:
        1) URL of notebook (of using the one from https://hub.mybinder.org/...): [*****PLEASE INSERT ONE HERE*****]
        2) Snapshot/picture of the cell that resulted in this error: [*****PLEASE INSERT ONE HERE*****]
        Thank you and we apologize for any inconvenience. We'll get back to you as soon as possible!
        Sincerely,
        Dan Feldman
        Automatically generated summary:
        - Time of occurrence: {now}
        - Request method + url: {response.request.method} - {response.request.url}
        - Request headers: {response.request.headers}
        - Request body: {response.request.body}
        - Response: {parsed_response}
        --------------------------------------- END ERROR MESSAGE ------------------------------------------
        \n\n
        """
        raise Exception(msg)
url = "https://sandbox.mint-data-catalog.org"
provenance_id = "e8287ea4-e6f2-47aa-8bfc-0c22852735c8"
resp = requests.get(f"{url}/get_session_token").json()
resp
print(resp)
api_key = resp['X-Api-Key']
api_key
request_headers = {
    'Content-Type': "applications/json",
    'X-Api-Key': api_key
}
request_headers
exit()
import flopy
import os
os.getcwd()
exit()
import geopandas
exit()
import geopandas
ls
inv = geopandas.read_file("ATX-LU-Inv-HUC120902050408buf/ATX-LU-Inv-HUC120902050408buf.shp")
inv.columns
inv.drop(columns=['created_by', 'date_creat', 'time_creat', 'general_la', 'modified_b', 'date_modif', 'time_modif', 'objectid', 'shape_area', 'shape_leng', 'index_righ', 'HUC_8', 'HUC_10', 'HUC_12', 'Area_acres', 'HU_10_DS', 'HU_10_NAME', 'HU_12_DS', 'HU_12_NAME', 'HUC_2', 'HUC_4', 'HUC_6', 'Shape_Le_1', 'Area_sqkm', 'Area_sqmi', 'HAND1m_url', 'FPgdb_url', 'Ver_date', 'Ver_commen', 'Data_url', 'BUFF_DIST', 'ORIG_FID', 'Shape_Le_2', 'Shape_Ar_1'])
inv.drop(columns=['created_by', 'date_creat', 'time_creat', 'general_la', 'modified_b', 'date_modif', 'time_modif', 'objectid', 'shape_area', 'shape_leng', 'index_righ', 'HUC_8', 'HUC_10', 'HUC_12', 'Area_acres', 'HU_10_DS', 'HU_10_NAME', 'HU_12_DS', 'HU_12_NAME', 'HUC_2', 'HUC_4', 'HUC_6', 'Shape_Le_1', 'Area_sqkm', 'Area_sqmi', 'HAND1m_url', 'FPgdb_url', 'Ver_date', 'Ver_commen', 'Data_url', 'BUFF_DIST', 'ORIG_FID', 'Shape_Le_2', 'Shape_Ar_1']).columns
inv = inv.drop(columns=['created_by', 'date_creat', 'time_creat', 'general_la', 'modified_b', 'date_modif', 'time_modif', 'objectid', 'shape_area', 'shape_leng', 'index_righ', 'HUC_8', 'HUC_10', 'HUC_12', 'Area_acres', 'HU_10_DS', 'HU_10_NAME', 'HU_12_DS', 'HU_12_NAME', 'HUC_2', 'HUC_4', 'HUC_6', 'Shape_Le_1', 'Area_sqkm', 'Area_sqmi', 'HAND1m_url', 'FPgdb_url', 'Ver_date', 'Ver_commen', 'Data_url', 'BUFF_DIST', 'ORIG_FID', 'Shape_Le_2', 'Shape_Ar_1'])
inv
inv.to_file("ATX-LU-Inv-HUC120902050408buf/ATX-LU-Inv-HUC120902050408buf.shp")
ls
inv.to_file("ATX-LU-Inv-HUC120902050408buf/ATX-LU-Inv-HUC120902050408buf.shp")
ls
inv
inv.crs
geopandas.read_file("Travis-DEM-10m-HUC120902050408bufdd-0m_6m.shp")
dd = geopandas.read_file("Travis-DEM-10m-HUC120902050408bufdd-0m_6m/Travis-DEM-10m-HUC120902050408bufdd-0m_6m.shp")
dd.columns
dd.crs
dd.to_crs(inv.crs)
dd_std = dd.to_crs(inv.crs)
ls
inv06 = geopandas.sjoin(inv,dd_std,how="inner",op="intersects")
import rtree
exit()
import geopandas
exit()
import geopandas
exit()
import sys
if hasattr(sys, 'real_prefix'):
    print(sys.real_prefix)
print(sys.prefix)
import geopandas
exit()
import geopandas
exit()
import geopandas
exit()
import geopandas
exit()
import geopandas as gpd
exit()
import sys
sys.prefix
exit()
import sys
sys.prefix
ls
import geopandas as gpd
exit()
ls
exit()
import rasterio as rio
exit()
import rasterio
exit()
import rasterio
population = rasterio.open("GPW-Pop_count-UNWPP-4.11/GPW-Pop_count-UNWPP-4.11.tif")
population
population.read()
population.read().unique()
import numpy as np
np.unique(population.read())
exit()
import rasterio
population = rasterio.open("GPW-Pop_count-UNWPP-4.11/GPW-Pop_count-UNWPP-4.11.tif")
population.shape
admin_area = rasterio.open("GPW-Mean_admin_unit_area-4.11/GPW-Mean_admin_unit_area-4.11.tif")
admin_area.shape
exit()
ls
import rasterio as rio
rio.open("Awash/GPW-Mean_admin_unit_area-4.11-Awash_buf.tif")
exit()
exit(0)
exit()
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]
onlyfiles
onlyfiles.shape]
onlyfiles.shape
len(onlyfiles)
exit()
import rasterio as rio
exit()
import rasterio
ned10m = rasterio.open("USGS-NED-10m-HUC120903020106buff.tif")
ned10m
ned10m.crs
ned10.
ned10
ned10m
ned10m.meta
ned10m.profile
ned10m.crs
ned10m.crs.is_projected
ned10m.crs.is_geographic
ned10m.crs.is_valid
ned10m.crs.is_epsg_code
ned10m.crs.data
ned10m.crs.to_wkt()
ned10m.crs.wkt
ned10m.meta
import gdal
ds = gdal.open("USGS-NED-10m-HUC120903020106buff.tif",gdal.GA_ReadOnly)
ds = gdal.Open("USGS-NED-10m-HUC120903020106buff.tif",gdal.GA_ReadOnly)
ds
ds.GetProjection()
projection = ds.GetProjection()
import osr
outRasterSRS = osr.SpatialReference(wkt=projection)
outRasterSRS
outRasterSRS.GetAuthorityCode("PROJCS")
authoritycode = outRasterSRS.GetAuthorityCode("PROJCS")
authoritycode
exit()
import flopy
flopy.modflow.mfwel.ModflowWel
m = flopy.modflow.Modflow()
wel = flopy.modflow.ModflowWel(m)
wel
wel.data_list
wel.stress_period_data
wel.stress_period_data.data
wel.check()
wel.dtype
os.getcwd()
import os
os.getcwd()
wel.to_shapefile('wel.shp')
wel.export('wel.shp')
m.grid_type
m
m = flopy.modflow.Modflow.load('EDWARDS_BFZ-BARTON_SPRINGS-TRANSIENT.nam')
m = flopy.modflow.Modflow.load('EDWARDS_BFZ-BARTON_SPRINGS-TRANSIENT/EDWARDS_BFZ-BARTON_SPRINGS-TRANSIENT.nam')
m = flopy.modflow.Modflow.load('Edwards_BFZ-Barton_Springs-Transient-mf05/EDWARDS_BFZ-BARTON_SPRINGS-TRANSIENT.nam')
m
m.wel
m.wel.export('wel.shp')
exit()
import pymake
pymake('./src','mfnwt','ifort','gcc',makeclean=True,expedite=False,dryrun=False,double=False,debug=False,include_subdirs=False)
pymake.main('./src','mfnwt','ifort','gcc',makeclean=True,expedite=False,dryrun=False,double=False,debug=False,include_subdirs=False)
exit()
from __future__ import division
import os
import gdal, osr
from osgeo import ogr
from skimage.graph import route_through_array
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import configparser
import inspect
from time import perf_counter
import dask_image.imread
import dask.array as da
import dask
from dask.distributed import Client, LocalCluster, performance_report
from dask_jobqueue import SLURMCluster
#from dask_mpi import initialize
import dask.dataframe as dd
q()
exit()
from __future__ import division
import os
import gdal, osr
from osgeo import ogr
from skimage.graph import route_through_array
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import configparser
import inspect
from time import perf_counter
import dask_image.imread
import dask.array as da
import dask
exit()
from __future__ import division
import os
import gdal, osr
from osgeo import ogr
from skimage.graph import route_through_array
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import configparser
import inspect
from time import perf_counter
import dask_image.imread
import dask.array as da
import dask
from dask.distributed import Client, LocalCluster, performance_report
from dask_jobqueue import SLURMCluster
#from dask_mpi import initialize
import dask.dataframe as dd
originX = 0.0
originY = 0.0
pixelWidth = 0.0
pixelHeight = 0.0
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal')
92*.85
62*.85
cluster = SLURMCluster(cores=48,memory='52GB',project='PT2050-DataX',processes=8,queue='skx-normal')
cluster
client = Client(cluster) # Cores selected
client.scale(8) # The value in "scale()" should be equal to the number of nodes
cluster.scale(8_
cluster.scale(8)
client.scale(8,jobs=1,cores=48) # The value in "scale()" should be equal to the number of nodes
cluster.scale(8,jobs=1,cores=48) # The value in "scale()" should be equal to the number of nodes
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra='-N 1')
client.close()
cluster.close()
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra='-N 1')
client = Client(cluster) # Cores selected
cluster.scale(8) # The value in "scale()" should be equal to the number of nodes
client.close(
client.close()
cluster.close()
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',n_workers=8,job_extra='-N 1')
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra='-N 1',n_workers=8)
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
client.close()
cluster.close()
cluster = SLURMCluster(cores=48,memory='78GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
cluster.close()
cluster = SLURMCluster(cores=48,memory='39GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
cluster.close()
cluster = SLURMCluster(cores=48,memory='4GB',project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
cluster.close()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',n_workers=8)
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],n_workers=8)
cluster.close()
exit()
from __future__ import division
import os
import gdal, osr
from osgeo import ogr
from skimage.graph import route_through_array
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import configparser
import inspect
from time import perf_counter
import dask_image.imread
import dask.array as da
import dask
from dask.distributed import Client, LocalCluster, performance_report
from dask_jobqueue import SLURMCluster
#from dask_mpi import initialize
import dask.dataframe as dd
originX = 0.0
originY = 0.0
pixelWidth = 0.0
pixelHeight = 0.0
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',n_workers=8)
cluster.close()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB')
header_lines = cluster.job_header.split('\n')
mem_pos = find_mem_pos(header_lines)
header_lines
def find_mem_pos(header_lines):
    for i,line in enumerate(header_lines):
        if('--mem=' in line):
            return i
mem_pos = find_mem_pos(header_lines)
mem_pos
header_lines = header_lines[:mem_pos] + header_lines[mem_pos+1:]
cluster.job_header = '\n'.join(header_lines)
cluster.job_header
type(cluster.job_header)
cluster.job_header = '\n'.join(header_lines)
cluster.setattr()
setattr
setattr(cluster,job_header,'\n'.join(header_lines))
setattr(cluster,'job_header','\n'.join(header_lines))
cluster.__setattr__('job_header','\n'.join(header_lines))
cluster.__setattr__()
cluster
cluster.job_cls()
cluster.job_script()
type(cluster.job_script())
cluster.job_script()
cluster.job_script(1)
cluster.security
cluster.scheduler_spec
cluster.status
cluster.worker_spec
cluster.workers
cluster.requested
cluster.plan
cluster.periodic_callbacks
cluster.observed
cluster.new_worker_spec()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',header_skip=['--mem='])
cluster.close()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',header_skip=['--mem='])
cluster
cluster.job_header
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',header_skip=['--mem=73G'])
cluster.job_header
cluster.job_script()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB')
cluster.job_script()
cluster = SLURMCluster(cores=48,project='PT2050-DataX',processes=8,queue='skx-normal',job_extra=['-N 1'],memory='78GB',header_skip=['--mem=73G'])
cluster.scale(8)
cluster.workers
cluster.worker_spec
client = Client(cluster)
client
config = configparser.RawConfigParser()
import os
os.getcwd()
config.read(r'/scratch/04950/dhl/GeoFlood/Tools/GeoFlood.cfg')
config
geofloodHomeDir = config.get('Section', 'geofloodhomedir')
geofloodHomeDir
projectName = config.get('Section', 'projectname')
geofloodResultsDir = os.path.join(geofloodHomeDir, "Outputs",
                                      "GIS", projectName)
DEM_name = config.get('Section', 'dem_name')
Name_path = os.path.join(geofloodResultsDir, DEM_name)
flowline_csv = Name_path + '_endPoints.csv'
curvaturefn = Name_path + '_curvature.tif'
facfn = Name_path + '_fac.tif'
skeletonfn = Name_path + '_flowskeleton.tif'
handfn = Name_path + '_NegaHand.tif'
flowlinefn = Name_path + '_channelNetwork.shp'
costsurfacefn = Name_path + '_cost.tif'
pathfn = Name_path + '_path.tif'
streamcell_csv = Name_path + '_streamcell.csv'
facArray = raster2array(facfn)
facArray = dask_image.imread,imread(facfn)
facArray = dask_image.imread.imread(facfn)
os.getcwd()
flowlinefn
flowline_csv
os.getcwd()
flowline_shp = './orig/Inputs/GIS/OnionCreek_1m_test/Flowline.shp'
in_shp = './orig/Inputs/GIS/OnionCreek_1m_test/Flowline.shp'
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(in_shp, 0)
layer = dataSource.GetLayer()
layer
for feature in layer:
    print(feature)
feature
geom = feature.GetGeometryRef()
geom
dataSourcegpd = gpd.read_file(in_shp)
import geopandas as gpd
dir()
os.listdir()
import readline
readline.write_history_file('Network-history.py
readline.write_history_file('Network-history.py')
exit()
import geopandas as gpd
with gpd.open('TX-County-Young.shp') as county:
    del(county)
with gpd.read_file('TX-County-Young.shp') as county:
    del(county)
with gpd.read_file('TX-County-Young.shp') as county:
    print(county)
exit()
import rasterio as rio
with rio.open('TX-County-Young-Elevation-120301010101.tif') as tif:
    print(tif)
with rio.open('TX-County-Young-Elevation-120301010101.tif') as tif:
    del(tif)
tif
for i in range(10):
    print(i)
for i in range(10):
    del(i)
i
unique
crs
dir()
import geopandas as gpd
shape = gpd.read_file('TX-County-Young-Catchments-120602010104.shp')
shape.crs
shape.crs.datum
from pyproj import CRS
CRS(proj='utm', zone=14, datum=shape.crs.datum)
shape.crs.datum
test = shape.crs.datum
test.name
test.type_name
test.to_wkt()
test.scope
test.name
CRS(proj='utm', zone=14, datum=shape.crs.datum.name)
CRS(proj='utm', zone=14, datum="NAD83")
CRS(proj='utm', zone=14, datum="NAD83")
test = CRS(proj='utm', zone=14, datum="NAD83")
crs = shape.crs
crs
crs.to_epsg()
import pyproj
crs
type(crs)
crs.datum
pyproj.crs.Datum.from_name(crs.datum.name)
pyproj.__version__
test = CRS(proj='utm', zone=14, datum="NAD83")
CRS(proj='utm', zone=14, datum="NAD83")
CRS(proj='utm', zone=14, datum="North American Datum 1983")
CRS.datum
CRS(proj='utm', zone=14)
test = CRS(proj='utm', zone=14)
test.datum = 
crs.datum
crs
crs.datum
test.datum = crs.datum
test.utm_zone
test
test.to_proj4(
test.to_proj4()
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_authority(crs.to_authority())
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_authority(crs.to_authority()))
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_epsg((crs.to_epsg())
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_epsg((crs.to_epsg()))
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_epsg((crs.to_epsg())))
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_epsg(crs.to_epsg()),prime_meridian=pyproj.crs.PrimeMeridian.from_epsg(crs.to_epsg())
cd = pyproj.crs.datum.CustomDatum(ellipsoid=pyproj.crs.Ellipsoid.from_epsg(crs.to_epsg()),prime_meridian=pyproj.crs.PrimeMeridian.from_epsg(crs.to_epsg()))
shape
buffer
import os
test = os.path.join('test','test')
test
os.path.join(test,'test.tif')
shape.index
shape.index.unique()
np.sort(shape.index.unique())
import numpy as np
np.sort(shape.index.unique())
2.2*47.
2.9*47.
100./2.9
var
139.8*.6
exit
exit()
import geopandas as gpd
lidar = gpd.read_file('TNRIS-LIDAR-Availability-20200219.shp/TNRIS-LIDAR-Availability-20200219.shp')
lidar
lidar['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro'
lidar[lidar['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro']
lidar = lidar[lidar['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro']
lidar
lidar = lidar.dissolve(by='dirname')
lidar
lidar.crs
lidar.to_file('stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro.shp')
exit()
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12.columns
hu12['HUC12']==120301020101
(hu12['HUC12']==120301020101).sum9)
(hu12['HUC12']==120301020101).sum()
(hu12['HUC12']=='120301020101').sum()
hu12[hu12['HUC12']=='120301020101']
hu = hu12[hu12['HUC12']=='120301020101']
hu
hu.crs
hu.to_file('WBD-HUC120301020101.shp')
exit()
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12['HUC12']
hu12['HUC12'].str.contains('408$')
hu12[hu12['HUC12'].str.contains('408$')]
hu12[hu12['HUC12'].str.contains('120601010804')]
hu12 = hu12[hu12['HUC12'].str.contains('120601010804')]
import os
os.getcwd()
hu12.to_file('WBD-HUC120601010804.shp')
exit()
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12['HUC12']=='120601020602'
hu12[hu12['HUC12']=='120601020602']
hu12 = hu12[hu12['HUC12']=='120601020602']
hu12
import os
os.getcwd()
hu12.to_file('WBD-HUC120601020602.shp')
exit()
120601010807
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12['HUC12']==120601010807
hu12[hu12['HUC12']==120601010807]
hu12[hu12['HUC12']=='120601010807']
hu12 = hu12[hu12['HUC12']=='120601010807']
hu12.to_file('WBD-HUC120601010807.shp')
exit()
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12[hu12['HUC12']=='120601010705']
hu12[hu12['HUC12']=='120601010705'].to_file('WBD-HUC120601010705.shp')
exit()
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
utm = gpd.read_file('utmzone/utmzone.shp')
utm
utm['ZONE'].str.contains('13.0|14.0|15.0')
utm['ZONE'].astype(str).str.contains('13.0|14.0|15.0')
utm[utm['ZONE'].astype(str).str.contains('13.0|14.0|15.0')]
utm131415 = utm[utm['ZONE'].astype(str).str.contains('13.0|14.0|15.0')]
utm131415
utm131415[utm131415['ROW'].str.contains('R|S')]
utm131415
utm131415[utm131415['ROW_'].str.contains('R|S')]
utmtx = utm131415[utm131415['ROW_'].str.contains('R|S')]
utmtx
utmtx.dissolve(by='ZONE')
utmtx_dissolve = utmtx.dissolve(by='ZONE')
utmtx_dissolve.loc[13.0]
utmtx_dissolve.loc[13.0,'Shape_Area']
utmtx[utmtx['ZONE'==13.0]
utmtx[utmtx['ZONE'==13.0]]
utmtx[utmtx['ZONE']==13.0]
utmtx[utmtx['ZONE']==13.0]['Shape_Area']
utmtx[utmtx['ZONE']==13.0]['Shape_Area'].sum()
utmtx_dissolve.loc[13.0,'Shape_Area'] = utmtx[utmtx['ZONE']==13.0]['Shape_Area'].sum()
utmtx_dissolve.loc[14.0,'Shape_Area'] = utmtx[utmtx['ZONE']==14.0]['Shape_Area'].sum()
utmtx_dissolve.loc[15.0,'Shape_Area'] = utmtx[utmtx['ZONE']==15.0]['Shape_Area'].sum()
utmtx_dissolve
utmtx
utmtx_dissolve['ROW_']
utmtx_dissolve['ROW_'] = 'S|R'
utmtx_dissolve
utmtx_dissolve.crs
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12.columns
hu12['STATES'].unique()
hu12['STATES'].str.contains('TX')
hu12[hu12['STATES'].str.contains('TX')]['STATES'].unique()
hu12[hu12['STATES'].str.contains('TX')]
hu12 = hu12[hu12['STATES'].str.contains('TX')]
hu12
catch = gpd.read_file('NFIEGeo_12.gdb',layer='Catchment')
catchs
catch
flow = gpd.read_file('NFIEGeo_12.gdb',layer='Flowline')
flow
flow_rep = flow.copy()
flow_rep['geometry'] = flow.representative_point()
flow_rep
hu12
hu12.columns
hu12['dissolve'] = True
hu12.dissolve(by='dissolve')
hu12
hu12.geometry
hu12
utm
utmtx_dissolve
utmtx_dissolve.loc[13.0]
utmtx_dissolve.loc[13.0]['geometry']
print(utmtx_dissolve.loc[13.0]['geometry'])
hu12
hu12.bounds
hu12.bounds.min
hu12.bounds.min()
hu12.bounds.max()
utmtx
hu12.bounds
hu12.bounds['minx']
hu12.bounds['minx']<-102.0
hu12[hu12.bounds['minx']<-102.0]
np.logical_and(hu12.bounds['minx']<-102.0)
import numpy as np
np.logical_and(hu12.bounds['minx']<-102.0)
np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)
hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)]
hu12[np.logical_and(hu12.bounds['maxx']<-102.0,hu12.bounds['minx']>-102.0)]
hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)]
hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index
hu12[~hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index]
hu12[hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index]
hu12.loc[hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index]
hu12.loc[~hu12[np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index]
np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)].index
np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)
~np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)
hu12[~np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)]
hu12
hu12 = hu12[~np.logical_and(hu12.bounds['minx']<-102.0,hu12.bounds['maxx']>-102.0)]
hu12
utm
utmtx
hu12[~np.logical_and(hu12.bounds['minx']<-96.0,hu12.bounds['maxx']>-96.0)]
hu12 = hu12[~np.logical_and(hu12.bounds['minx']<-96.0,hu12.bounds['maxx']>-96.0)]
hu12
hu12.bounds
hu12.bounds['maxx']<-102.0
hu12[hu12.bounds['maxx']<-102.0]
hu12[hu12.bounds['minx']>-96.0]
hu12[hu12.bounds['maxx']>-96.0]
hu12[hu12.bounds['minx']>-96.0]
hu12[np.logical_and(hu12.bounds['minx']>-102.0,hu12.bounds['maxx']<-96.0)]
4049+899+1210
hu12
hu12_buff = hu12.copy()
hu12_buff
hu12_buff13
hu12_buff[hu12_buff.bounds['maxx']<-102.0]
hu12_buffutm13 = hu12_buff[hu12_buff.bounds['maxx']<-102.0]
hu12_buffutm1
hu12_buffutm13
hu12_buffutm13.to_crs('epsg:26913')
hu12_buffutm13.to_crs('epsg:26913',inplace=True)
hu12_buffutm13
hu12_buffutm13['geometry']
hu12_buffutm13.buffer(500.0)
import os
import readline
os.getcwd()
os.listdir()
readline.write_history_file('history-input_shp.py')
exit()
import geopandas as gpd
hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp')
hu12
hu12.columns
hu12[hu12['HUC12']=='121102060501']
hu12[hu12['HUC12']=='121102060501'].crs
hu12[hu12['HUC12']=='121102060501'].to_file('WBD-HUC12/WBD-HUC121102060501.shp/WBD-HUC121102060501.shp')
hu12
hu12['STATES'].unique()
hu12tx
dir()
hu12['STATES'].str.contains('TX')
hu12[hu12['STATES'].str.contains('TX')]
28+26+60
28+26+60*48
hu12tx = hu12[hu12['STATES'].str.contains('TX')]
hu12tx[hu12tx['HUC12']=='120601020503']
os.getcwd()
import os
os.getcwd()
os.listdir()
hu12tx[hu12tx['HUC12']=='120601020503'].to_file('WBD-HUC12/WBD-HUC120601020503.shp')
hu12tx[hu12tx['HUC12']=='120601020306'].to_file('WBD-HUC12/WBD-HUC120601020306.shp')
hu12tx[hu12tx['HUC12']=='120601020306']
hu12tx[hu12tx['HUC12']=='120601020607']
hu12tx[hu12tx['HUC12']=='120601020607'].to_file('WBD-HUC12/WBD-HUC120601020607.shp')
hu12tx[hu12tx['HUC12']=='120601020702'].to_file('WBD-HUC12/WBD-HUC120601020702.shp')
hu12tx[hu12tx['HUC12']=='120601020403'].to_file('WBD-HUC12/WBD-HUC120601020403.shp')
hu12tx[hu12tx['HUC12']=='120601020207'].to_file('WBD-HUC12/WBD-HUC120601020207.shp')
hu12tx[hu12tx['HUC12']=='120601020207']
hu12tx[hu12tx['HUC12']=='120601020303'].to_file('WBD-HUC12/WBD-HUC120601020303.shp')
hu12tx[hu12tx['HUC12']=='120601010905'].to_file('WBD-HUC12/WBD-HUC120601010905.shp')
hu12tx[hu12tx['HUC12']=='120601010802'].to_file('WBD-HUC12/WBD-HUC120601010802.shp')
hu12tx[hu12tx['HUC12']=='120601010501'].to_file('WBD-HUC12/WBD-HUC120601010501.shp')
exit()
import numpy as np
np.arange()
np.arange(1)
np.arange(10)
np.range(10)
np.arange(10)
np.arange(10,1)
np.arange(1,9)
np.arange(1,7
np.arange(1,8)
gpd
import geopandas as gpd
os
import os
os.getcwd()
flows = gpd.read_file('data/NFIEGeo_TX.gdb',layer='Flowline')
flows
flows.columns
flows['StreamOrde']
flows.groupby('StreamOrde')
dict(tuple(flows.groupby('StreamOrde')))
sorted(dict(tuple(flows.groupby('StreamOrde'))),keys=[3,2,5])
sorted(dict(tuple(flows.groupby('StreamOrde'))),key=[3,2,5])
sorted(dict(tuple(flows.groupby('StreamOrde'))))
sorted(dict(tuple(flows[flows['StreamOrde'].isin([3,2,5])].groupby('StreamOrde'))),key=[3,2,5])
sorted(dict(tuple(flows[flows['StreamOrde'].isin([3,2,5])].groupby('StreamOrde'))))
dict(tuple(flows[flows['StreamOrde'].isin([3,2,5])].groupby('StreamOrde')))
flows[flows['StreamOrde'].isin([3,2,5])]
flows[flows['StreamOrde'].isin([3,2,5])]['StreamOrde']
flows[flows['StreamOrde'].isin([3,2,5])]['StreamOrde'].unique()
(flows.iloc[25],flows.iloc[100389])
(flows.iloc[[25]],flows.iloc[[100389]])
pd.concat(flows.iloc[[25]],flows.iloc[[100389]])
import pandas as pd
pd.concat(flows.iloc[[25]],flows.iloc[[100389]])
pd.concat([flows.iloc[[25]],flows.iloc[[100389]]])
pd.concat([flows.iloc[[100389]],flows.iloc[[25]]])
pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde')
dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde')))
pd.concat([flows.iloc[[100389]],flows.iloc[[25]]])
pd.concat([flows.iloc[[100389]],flows.iloc[[25]]])['StreamOrde']
pd.concat([flows.iloc[[100389]],flows.iloc[[25]]])['StreamOrde'].unique()
dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde'))).values()
list(dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde'))).values())
list(dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde'))).values())[0]
type(list(dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde'))).values())[0])
list(dict(tuple(pd.concat([flows.iloc[[100389]],flows.iloc[[25]]]).groupby('StreamOrde'))).values())[0]
flows.sort_values('StreamOrde').groupby('StreamOrde')
flows.sort_values('StreamOrde')
flows.sort_values('StreamOrde')['StreamOrde']
flows.sort_values('StreamOrde').groupby('StreamOrde')
list(flows.sort_values('StreamOrde').groupby('StreamOrde'))
dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde')))
dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values
dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values()
type(dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values())
list(dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values())
type(list(dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values()))
len(list(dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values()))
list(dict(tuple(flows.sort_values('StreamOrde').groupby('StreamOrde'))).values())
flows.flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())
flows[flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())]
flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())
flows[flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())]['StreamOrde']
flows[flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())]['StreamOrde'].unique()
np.sort(flows[flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())]['StreamOrde'].unique())
flows_keys = np.sort(flows[flows['StreamOrde'].isin(flows.groupby('StreamOrde').groups.keys())]['StreamOrde'].unique())
flows.groupby('StreamOrde').groups.keys().isin(flows_keys)
flows.groupby('StreamOrde').groups.keys()
set(flows.groupby('StreamOrde').groups.keys())
set(flows.groupby('StreamOrde').groups.keys()).intersection(flows_keys)
list(set(flows.groupby('StreamOrde').groups.keys()).intersection(flows_keys))
np.sort(set(flows.groupby('StreamOrde').groups.keys()).intersection(flows_keys))
np.sort(list(set(flows.groupby('StreamOrde').groups.keys()).intersection(flows_keys)))
set(flows.groupby('StreamOrde').groups.keys()).intersection(flows['StreamOrde'])
set(flows[flows['HUC12'].isin([3,1,5,6]).groupby('StreamOrde').groups.keys()).intersection(flows['StreamOrde'])
set(flows[flows['HUC12'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows['StreamOrde'])
set(flows[flows['StreamOrde'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows['StreamOrde'])
set(flows[flows['StreamOrde'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows[flow['StreamOrde'].isin([1,3,4])])
set(flows[flows['StreamOrde'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows[flows['StreamOrde'].isin([1,3,4])])
set(flows[flows['StreamOrde'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows[flows['StreamOrde'].isin([1,3,4])]['StreamOrde'])
np.sort(list(set(flows[flows['StreamOrde'].isin([3,1,5,6])].groupby('StreamOrde').groups.keys()).intersection(flows[flows['StreamOrde'].isin([1,3,4])]['StreamOrde'])))
flows.groupby('StreamOrde')
dict(tuple(flows.groupby('StreamOrde')))
flows_keys
dict(tuple(flows.groupby('StreamOrde')))[flows_keys]
dict(tuple(flows.groupby('StreamOrde')))[[flows_keys]]
{k:dict(tuple(flows.groupby('StreamOrde'))[k] for k in flows_keys)
{k:dict(tuple(flows.groupby('StreamOrde'))[k] for k in flows_keys}
{k:dict(tuple(flows.groupby('StreamOrde')))[k] for k in flows_keys}
type({k:dict(tuple(flows.groupby('StreamOrde')))[k] for k in flows_keys})
{k:dict(tuple(flows.groupby('StreamOrde')))[k] for k in flows_keys}
860.*12.
exit()
import geopandas as gpd
exit()
import geopandas as gpd
utmzone = gpd.read_file('utmzone/utmzone.shp')
utmzone
utmzone['ROW_'].isin('S')
utmzone['ROW_']=='S'
utmzone[utmzone['ROW_']=='S']
utmzone[utmzone['ROW_']=='S'].bounds
utmzone[utmzone['ROW_']=='R'].bounds
utmzone[utmzone['ROW_']=='S'].bounds
utmzone[utmzone['ROW_']=='R'].bounds
utmzone[utmzone['ROW_']=='S'].bounds
utmzone[utmzone['ZONE']==13.].bounds
import shapely
shapely.geometry.Polygon(shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.))
shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])
gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])])
utmzone.to_crs('epsg
utmzone[utmzone['ZONE']==13.].to_crs('epsg:4269).bounds
utmzone[utmzone['ZONE']==13.].to_crs('epsg:4269').bounds
utmzone[utmzone['ROW_']=='S'].to_crs('epsg:4269').bounds
gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])],crs='epsg:4269')
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])],crs='epsg:4269')
shapefile.crs
shapefile.to_file('TX-UTM13-.1.shp')
exit()
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])],crs='epsg:4269')
import geopandas as gpd
import shapely
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.9,40.),shapely.geometry.Point(-102.1,40.),shapely.geometry.Point(-102.1,24.),shapely.geometry.Point(-107.9,24.),shapely.geometry.Point(-107.9,40.)])],crs='epsg:4269')
shapefile = gpd.GeoDataFrame(geometry=[shapely\.geometry.Polygon([shapely.geometry.Point(-107.8,40.),shapely.geometry.Point(-102.2,40.),shapely.geometry.Point(-102.2,24.),shapely.geometry.Point(-107.8,24.),shapely.geometry.Point(-107.8,40.)])],crs='epsg:4269')
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.8,40.),shapely.geometry.Point(-102.2,40.),shapely.geometry.Point(-102.2,24.),shapely.geometry.Point(-107.8,24.),shapely.geometry.Point(-107.8,40.)])],crs='epsg:4269')
shapefile.to_file('TX-UTM13-.2.shp')
exit()
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.5,40.),shapely.geometry.Point(-102.5,40.),shapely.geometry.Point(-102.5,24.),shapely.geometry.Point(-107.5,24.),shapely.geometry.Point(-107.5,40.)])],crs='epsg:4269')
import geopandas as gpd
import shapely
shapefile = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([shapely.geometry.Point(-107.5,40.),shapely.geometry.Point(-102.5,40.),shapely.geometry.Point(-102.5,24.),shapely.geometry.Point(-107.5,24.),shapely.geometry.Point(-107.5,40.)])],crs='epsg:4269')
shapefile.to_file('TX-UTM13-.5.shp')
exit()
import geopandas as gpd
import glob
glob.glob('./TX-UTM13*/Elevation.tif')
elevations = glob.glob('./TX-UTM13*/Elevation.tif')
elevations
[os.path.dirname(elevation) for elevation in elevations]
import os
[os.path.dirname(elevation) for elevation in elevations]
directories [os.path.dirname(elevation) for elevation in elevations]
directories = [os.path.dirname(elevation) for elevation in elevations]
[os.path.join(directory,'Catchments.shp') for directory in directories]
catchments = [os.path.join(directory,'Catchments.shp') for directory in directories]
gdfs = [gpd.read_file(catchment) for catchment in catchments]
[gdf.crs for gdf in gdfs]
gdfs[0].crs
test = gdfs[0].crs
test.name
[ for gdf in gdfs]
gpd.GeoDataFrame(pd.concat(gdfs),crs=gdfs[0].crs)
import pandas as pd
gpd.GeoDataFrame(pd.concat(gdfs),crs=gdfs[0].crs)
elevations_covered = gpd.GeoDataFrame(pd.concat(gdfs),crs=gdfs[0].crs)
elevations_covered.to_file(TX-UTM13-.5-1m-GeoFlood-Catchments.shp')
elevations_covered.to_file('TX-UTM13-.5-1m-GeoFlood-Catchments.shp')
os.getcwd()
os.chdir('../')
pd.DataFrame
elevations_covered
elevations_covered.loc['GRIDCODE']
elevations_covered.loc[['GRIDCODE']]
elevations_covered['GRIDCODE']
gridcode = elevations_covered['GRIDCODE']
gridcode
elevations_covered
elevations_covered['FEATUREID'].unique()
len(elevations_covered['FEATUREID'].unique())
elevations_covered['FEATUREID']
elevations_covered
elevations_covered.set_index('FEATUREID')
elevations_covered = elevations_covered.set_index('FEATUREID')
elevations_covered
elevations_covered['GRIDCODE']
gridcode = elevations_covered['GRIDCODE']
gridcode.iloc[[3,4,5]]
gridcode.iloc[[2,3,4]]
gridcode = gridcode.iloc[[2,3,4]]
elevations_covered
elevations_covered['GRIDCODE_NEW'] = gridcode
elevations_covered
os.getcwd()
utmzone = gpd.read_file('utmzone/utmzone.shp')
utmzone
utmzone['ZONE'].isin([13.,14.,15.])
utmzon[utmzone['ZONE'].isin([13.,14.,15.])]
utmzone[utmzone['ZONE'].isin([13.,14.,15.])]
utmzone = utmzone[utmzone['ZONE'].isin([13.,14.,15.])]
utmzone['ROW_'].isin(['R','S'])
utmzone[utmzone['ROW_'].isin(['R','S'])]
utmzone = utmzone[utmzone['ROW_'].isin(['R','S'])]
utmzone
utmzone.dissolve('ZONE')
utmzone = utmzone.dissolve('ZONE')
utmzone
utmzone.loc[15.]
utmzone.loc[15.].bounds
utmzone.bounds
from shapely.geometry import Polygon, Point
Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])
gpd.GeoDataFrame(geometry=Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)]))
gpd.GeoDataFrame(geometry=[Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])])
gpd.GeoDataFrame(geometry=[Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])],crs='epsg:4269')
gpd.GeoDataFrame(geometry=[Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])],crs='epsg:4269').crs
gpd.GeoDataFrame(geometry=[Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])],crs='epsg:4269')
shapefile = gpd.GeoDataFrame(geometry=[Polygon([Point(-95.5,40.),Point(-90.5,40.),Point(-90.5,24.),Point(-95.5,24.),Point(-95.5,40.)])],crs='epsg:4269')
shapefile.to_file('TX-UTM15-.5.shp')
exit()
exit
exit()
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
ls
utmzone = gpd.read_file('data/utmzone/utmzone.shp')
utmzone
utmzone[utmzone['ZONE']==14.]
utmzone = utmzone[utmzone['ZONE']==14.]
utmzone
utmzone['ROW_']
utmzone['ROW_'].isin(['S','R'])
utmzone = utmzone['ROW_'].isin(['S','R'])
utmzone
utmzone = gpd.read_file('data/utmzone/utmzone.shp')
utmzone = utmzone[utmzone['ZONE']==14.]
utmzone = utmzone[utmzone['ROW_'].isin(['S','R'])]
utmzone
utmzone.dissolve('ZONE')
utmzone = utmzone.dissolve('ZONE')
utmzone.bounds
utmzone.crs
gpd.GeoDataFrame(geometry=[Polygon(Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.))])
gpd.GeoDataFrame(geometry=[Polygon([Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.)])])
gpd.GeoDataFrame(geometry=[Polygon([Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.)])],crs='epsg:4269')
gpd.GeoDataFrame(geometry=[Polygon([Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.)])],crs='epsg:4269').crs
gpd.GeoDataFrame(geometry=[Polygon([Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.)])],crs='epsg:4269')
shapefile = gpd.GeoDataFrame(geometry=[Polygon([Point(-101.5,40.),Point(-96.5,40.),Point(-96.5,24.),Point(-101.5,24.),Point(-101.5,40.)])],crs='epsg:4269')
shapefile.to_file('TX-UTM14-.5.shp')
utmzone
exit()
3450.*1.037
3450.*1.037*.1037
3450.*1.037*1.037
3450.*1.031*1.031
3450.*1.031
3450.*1.033
3450.*1.033*1.033
exit()
pi
import math
math.pi
math.pi*10^2
math.pi*10.^2
math.pi*10.**2
math.pi*10.**2/
math.pi*10.**2.
math.pi*14.**2.
exit()
250./4
1791+3221+815
5827.*.4
exit()
30279.22+300.09-4641.99
30279.22+300.09-4641.99-2600
30279.22+300.09-4641.99-26000
exit()
import geopandas as gpd
gpd.read_file('TX-County_boundaries.shp/County.shp')
counties = gpd.read_file('TX-County_boundaries.shp/County.shp')
counties
counties[counties['CNTY_NM']=='Orange']
orange = counties[counties['CNTY_NM']=='Orange']
orange
orange.to_file('TX-County-Orange.shp')
import os
os.getcwd()
orange_hu12 = gpd.read_file('WBD_National_GDB/WBD_National_GDB.shp/WBDHU12.shp', mask=orange)
orange_hu12
orange_hu12['HU12']
orange_hu12['HUC12']
dir()
os.getcwd()
os.listdir()
print(os.listdir())
pprint(os.listdir())
import pprint
pprint.(os.listdir())
pp = pprint.PrettyPrinter()
pp.pprint(os.listdir())
dir()
p
.575*3.1*2.
.575*3.1*2.+25.+10.
(.575*3.1*2.+25.+10.)*7.
40.*7.
50000./70.
50000./70./12.
50000./5300.
8.*8.*.9
11.+10.*3.+8.+7.+6.*2.
(11.+10.*3.+8.+7.+6.*2.)/8.
(10.95+9.95*3.+7.95.+6.95+5.95*2.)/8.
(10.95+9.95*3.+7.95+6.95+5.95*2.)/8.
8.45*8.*.9
2.*7.+1.
(2.*7.+1.)/8.
74./18.
(8.45*8.*.9)/(1.875*8.)
84.95/28.
18./3.
28./6.
3.+2.
443.*2.*.575
443.*2.
443.*2.*.575+18.
443.*2./25.1
443.*2./25.1*2.23
280.+208.43
443.*2.*.575+18.
443.*2.*.575+18.+13.01*2.*(6.+49./60.)
280.+208.43+13.01*2.*(12.+14.+1.*2.+(22.*2.+31.+2.)/60.)
13.01*2.*(12.+14.+1.*2.+(22.*2.+31.+2.)/60.)
(12.+14.+1.*2.+(22.*2.+31.+2.)/60.)
443.*2.*.575+18.+13.01*2.*(6.+49./60.)*2.
13.01*2.*(6.+49./60.)*2.
(6.+49./60.)*2.
6./7.
550/4.*2.+385./4.+800./4.
dir()
exit()
1147.*400.
import os
os.getcwd()
exit()
import geopandas as gpd
exit()
ls
import geopandas as gopd
import geopandas as gpd
exit()
import os
import h5py
from netCDF4 import Dataset
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import datetime
import scipy.signal as sig
from scipy.ndimage import zoom
exit()
import h5py
exit()
import h5py
exit()
import h5py
exit()
import h5py
import os
import h5py
from netCDF4 import Dataset
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import datetime
import scipy.signal as sig
from scipy.ndimage import zoom
from gennetcdf import ncgen
catalog
catalog = "era_30_-99_31_-98_dish.h5"
dataset
dataset = "satellite"
newRes = (240,240)
dem = 'na_dem_15s.nc'
dataFormat = 'h5'
latl,latu = (30,31)
lonl,lonu = (-99,-98)
h5py.File(catalog,'r')
hf = h5py.File(catalog,'r')
hf
dataset_names = list(hf.keys())
dataset_names
data = np.array(hf[dataset][:])
data
outputdir = 'test1'
os.makedirs(outputdir)
hydroshedsDEM = 'na_dem_15s.nc'
im.shape
inputMat
dir()
hydroshedsDEM
hf
dem
data
im = data[98, :, :]
im
inshape = im.shape
inshape
dir()
dem
hydroshedsDEM = dem
hydroshedsDEM = dem.copy()
hydroshedsDEM = dem
f
f = Dataset(hydroshedsDEM, 'r')
f
lons
lons = np.array(f['lon'][:])
lats = np.array(f['lat'][:])
demBig = np.array(f['Band1'][:], dtype=np.float64)
demBig
lons
lats
lat0
lat0 = np.min(lats)
lon0 = np.min(lons)
cellsize = 0.004166666666662877
self.latl,self.latu = (30,31)
latl
latu
lonl
lonu
indxl
indxl = np.int(np.floor((lonl-lon0)/cellsize))
indxl
indxu = np.int(np.floor((lonu-lon0)/cellsize))
indxu
indyl = np.int(np.floor((latl-lon0)/cellsize))
indyu = np.int(np.floor((latu-lon0)/cellsize))
indxu
indyl
indyu
demHiRes = demBig[indyl:indyu, indxl:indxu]
demHiRes
demBig
demBig.shape
demHiRes[demHiRes<-3000] = 0.0
demHiRes
demHiRes.shape
demBig.shape
kernelSize = np.int(demHiRes.shape[1],coarseShp[0])
kernelSize = np.int(demHiRes.shape[1],inshape[0])
inshape
inshape[0]
demHiRes.shape
demHiRes.shape[1]
kernelSize = np.int(demHiRes.shape[1]/inshape[0])
kernelSize
average_kernel = np.ones((kernelSize,kernelSize))/(kernelSize*kernelSize)
average_kernel
blurred_array = sig.convolve2d(demHiRes, average_kernel, mode='full')
blurred_array
blurred_array.unique()
np.unique(blurred_array)
demHiRes
demHiRes = demBig[indyl:indyu, indxl:indxu]
demHiRes
demBig
np.unique(demBig)
indyl
indyu
indxl
indxu
lon0
indyl = np.int(np.floor((latl-lat0)/cellsize))
indyu = np.int(np.floor((latu-lat0)/cellsize))
indyl
indyu
demHiRes = demBig[indyl:indyu, indxl:indxu]
demHiRes
demHiRes.shape
demHiRes[demHiRes<-3000] = 0.0
np.unique(demHiRes)
kernelSize = np.int(demHiRes.shape[1]/coarseShp[0])
kernelSize = np.int(demHiRes.shape[1]/inshape[0])
average_kernel = np.ones((kernelSize,kernelSize))/(kernelSize*kernelSize)
kernelSize
average_kernel
blurred_array = sig.convolve2d(demHiRes, average_kernel, mode='full')
blurred_array
demLoRes = blurred_array[kernelSize-1::kernelSize,kernelSize-1::kernelSize]
demLoRes
demLoRes = zoom(demLoRes,kernelSize,order=3,mode='nearest')
demLoRes
Zr = np.max(demHiRes) - np.min(demHiRes)
Zr
anomaly = (demHiRes-demLoRes)/Zr
anomaly
anomaly.max
anomaly.max()
anomaly.abs().max()
anomaly.abs.max()
np.abs(anomaly).max()
anomaly.min()
im
im = Image.fromarray(im)
im
kernelSize
factor = kernelSize
coarseShp = inshape
anomaly
gridRes = [inshape[0]*np.int(factor),inshape[1]*np.int(factor)]
gridRes
output
output = np.array(im.resize(size=gridRes,resample=Image.BILINEAR))
output
output = output*(1.0+anomaly)
output
dely
dely = (latu-latl)/gridRes[0]
delx = (lonu-lonl)/gridRes[1]
lats
np.arange(latl,latu,dely)
lats
15*1.7*2.
15*1.7
20.*1.7
lats = np.arange(latl,latu,dely)
lons = np.arange(lonl,lonu,delx)
ncgen('test1/test1{0}.nc'.format(op),lats,lons,[datetime.datetime.now()],output)
op = "downscale"
ncgen('test1/test1{0}.nc'.format(op),lats,lons,[datetime.datetime.now()],output)
dir()
ds
output
outputdir
op
ds = Dataset(os.path.join(outputdir,'test1downscale.nc')
ds = Dataset(os.path.join(outputdir,'test1downscale.nc'))
ds
print(ds)
ds.__dict__
ds.dimensions.values()
ds.variables.values()
ds
ds['data']
ds['data'][:]
import matplotlib.pyplot as plt
plt.figure(figsize(10,10))
plt.figure(figsize=(10,10))
plt.imshow(ds.variables['data'][:],origin='lower')
plt.figure()
im = plt.imshow(output,'output_data{0}.png'.format(op),title='Downscaled Precip')
im = plt.imshow(output,cmap=cmap)
im = plt.imshow(output,cmap='viridis')
plt.title('')
plt.colorbar(im)
plt.savefig('output_data_precip.png')
import readline
readline.write_history_file('history-dish.py')
exit()
21665./.3
21665./.4
exit()
413000.*.15
413000.*.15-6500.-1800.-2500.*7.
413000.*.15-6500.-1800.-2500.*20.
413000.*.15-6500.-1800.-2500.*21.
413000.*.15-6500.-1800.-2500.*22.
413000.*.15-6500.-1800.-4000.*10.
413000.*.15-6500.-1800.-4000.*13.
413000.*.15-6500.-1800.-4000.*14.
(190.+150.)/12.
(190.+150.)/12./8.
4.*44.56+5.*11.98
(4.*44.56+5.*11.98)*1.0825
(4.*44.56+5.*11.98)*1.0825*1.1
413000.*.15
413000.*.15-10000.
413000.*.15-10000.-13000.
413000.*.15-10000.-13000.-5000.
exit()
175000./.8
175000./.8*.01
175000./.8*.01*3.
175000./.8*1.03
175000./.8*1.03*.1
415746.*.1
exit()
100.*1.0336**7
165*1.2
785.-120.
785.-42.
exit()
1088./16./
1088./16.
exit()
14/76.
14./76.
14*25/(14*25+62*20)
7*25/(7*25+62*20)
12*25/(12*25+62*20)
5*3
5*35
exit()
11*13
exit()
import geopandas
import geopandas as gpd
exit()
1800*5*25
1500*(2020-1963)*5
5000*.2
50000*.2
50000*.1
50000*.05
50000*.2
1500*8
1500*8*.2
55000*.2
54900*.2
54900.*.06
54900.*.06/12
29000/1000
1252/365
dir()
exit
exit()
(750+80+60)*6
(750+80+60)*6-846.79
(750+80+60)*6-846.79+324.74+120+600
(750+80+60)*7/29
(750+80+60)*7/30
(750+80+60)*7/31
(750+80+60)*6/31
(750+80+60)*5/31
(750+80+60)*5/30
(750+80+60)*6/30
626.72*7/29
626.72*6/29
626.72*6/30
626.72*6/31
626.72*7/30
exit()
626.72*6/14
626.72*6/15
626.72*6/15.5
626.72*6/14
626.72*4/14
626.72*5/14
626.72*5/15.5
626.72*4/15.5
626.72*3/15.5
626.72*3/14
21.96*5
21.96*6
21.96*7
21.96*6.5
21.96*6.75
21.96*6.70
21.96*6.71
21.96*6.72
21.96*6.719
21.96*6.718
21.96*6.7185
6/29*750
6/30*750
6/31*750
7/31*750
7/30.5*750
6/30.5*750
6/(365/12)*750
(5/29+1/31)*750
6/29*750
6/31*750
dir()
exit()
159.99*1.0825*1.1
exit()
30733.41-26000.
(30733.41-26000.)+2902.64
(30733.41-26000.)/((30733.41-26000.)+2902.64)
(30733.41-26000.)/((30733.41-26000.)+2902.64)*4321.45
2902.64/((30733.41-26000.)+2902.64)*4321.45
(30733.41-26000.)+2902.64-4321.45
7.+16.+20.*2.
7.+16.+19.*2.
7./(7.+16.+19.*2.)
7.*25.
(16.+19.*2.)*20.
7.*25./(7.*25.+(16.+19.*2.)*20.)
7.*25./(7.*25.+(16.+19.*2.)*25.)
7.*25./(7.*25.+(16.+19.*2.)*20.)
12.*25./(12.*25.+(16.+19.*2.)*20.)
15.*25./(15.*25.+(16.+19.*2.)*25.)
15.*25./(15.*25.+(16.+19.*2.)*20.)
7.*25./(7.*25.+(16.+19.*2.)*20.)
7.*30./(7.*30.+(16.+19.*2.)*20.)
7.*50./(7.*50.+(16.+19.*2.)*20.)
7.*55./(7.*55.+(16.+19.*2.)*20.)
7.*50./(7.*50.+(16.+19.*2.)*20.)
904*.07
20./904
550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.07
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.07/7.
452.*.07
452.*.05
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.05/7.
559.*.05
909.*.05
(385.+80.+60.)*.05
(780.+80.+60.)*.05
20./830.
20./620.
700.+162.
(700.+162.)*.07
(904.+452.)*(19.*1.5)
(904.+452.)/2.*(19.*1.5)
19.*1.5
(19.*1.5)*28.
20./798.
(19.*1.5)*28.
(19.*1.5)
(904.+452.)/2.
20./((904.+452.)/2.)
902.*.024
904.*.024
452.*.032
452.*.07
452.*.05
452.*.024
21.70/452.
559.*.024
909.*.024
21.816/559.
(385.+60.+80.)*.07
(780.+60.+80.)*.07
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.07/7.
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.07
(550.*2.+800.+700.*2.+780.+385.+80.*6.+60.*7.)*.07*12.
.075*.02
.075*.2
7.*50./(7.*50.+(16.*1.2+19.*1.5.)*20.)
7.*50./(7.*50.+(16.*1.2+19.*1.5)*20.)
7./(7.+(16.*1.2+19.*1.5))
16.+19.*1.5
16.*.2
3.*909.+13.*559.
(3.*909.+13.*559.)/16.
22./(3.*909.+13.*559.)/16.
22./((3.*909.+13.*559.)/16.)
19.*.5
19.*904.
904.*19.
(904.*19.)/29.
22./((904.*19.)/29.)
452.*.024
909.*.024
2.4/3.
.07*.8
3.2/2.4
(385.+60.+80.)*.032
(780.+60.+80.)*.032
(780.+60.+80.)*.48
(780.+60.+80.)*.048
(385.+60.+80.)*.048
862.*.05
(385.+60.*80.)*.05
(385.+60.+80.)*.05
.05*.8
(385.+60.+80.)*.04
(780.+60.+80.)*.04
4./2.4
5./.8
5.*1.2
exit
exit()
3247.19*12
815-80-60
675+80
exit
exit()
import pylas
exit()
2176/32
exit()
ls
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
import geopandas as gpd
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
screen -ls
wbds
wbds.centroid
wbds.representative_point(
wbds.representative_point()
gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/ATX-Districts/geo_export_c4e1640a-fcb5-4ad9-a436-e419c76eaec1.shp')
atx = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/ATX-Districts/geo_export_c4e1640a-fcb5-4ad9-a436-e419c76eaec1.shp')
atx['const'] = True
atx.dissolve('const')
atx.dissolve('const').representative_point()
wbds.representative_point() - atx.dissolve('const').representative_point()
wbds.representative_point().distance(atx.dissolve('const').representative_point())
wbds.representative_point().distance(atx.dissolve('const').to_crs(wbds.crs).representative_point())
wbds.representative_point()
atx.dissolve('const').to_crs(wbds.crs).representative_point()
atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()
wbds.representative_point().distance(atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point())
wbds.representative_point().apply(lambda wbd: wbd.distance(atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point())
wbds.representative_point().apply(lambda wbd: wbd.distance(atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()))
wbds.representative_point().apply(lambda wbd: wbd)
wbds.representative_point().apply(lambda wbd: type(wbd))
wbds.representative_point().apply(lambda wbd: wbd)
wbds.representative_point().apply(lambda wbd: wbd.distance(atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point())[0])
atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point())
atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()
atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]
wbds.representative_point().apply(lambda wbd: wbd.distance(atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]))
atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]
atx_rep = atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]
wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)))
wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep))
wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort()
wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values()
wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index
wbds[[wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index]]
wbds.iloc[wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index]
wbds.iloc[wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index]['HUC12']
wbds.iloc[wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index]
wbds_atx = wbds.iloc[wbds.representative_point().apply(lambda wbd: wbd.distance(atx_rep)).sort_values().index]
wbds_atx
wbds_atx.to_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX-ATX.shp')
wbds_atx
wbds_atx['HUC12']
glob
import glob
os.getcwd()
import os
os.getcwd()
os.listdir()
wbds_atx['HUC12'].apply(lambda huc12: glob.glob('*/*/'+str(huc12)))
wbds_atx['HUC12'].apply(lambda huc12: glob.glob('*/*'+str(huc12)))
wbds_atx['filename'] = wbds_atx['HUC12'].apply(lambda huc12: glob.glob('*/*'+str(huc12)))
wbds_atx['filename']
wbds_atx['filename'].apply(lambda filename: len(filename))
wbds_atx['filename'].apply(lambda filename: len(filename))>0
(wbds_atx['filename'].apply(lambda filename: len(filename))>0).sum()
wbds_atx_fns = wbds_atx.copy()
wbds_atx_fns
(wbds_atx_fns['filename'].apply(lambda filename: len(filename))>0).sum()
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))>0)]
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))>0]
wbds_atx_fns = wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))>0]
wbds_atx_fns['filename'][0]
wbds_atx_fns['filename'][1]
wbds_atx_fns['filename'][2]
wbds_atx_fns['filename'][3]
wbds_atx_fns['filename'].apply(lambda filename: len(filename))
wbds_atx_fns['filename'].apply(lambda filename: len(filename)).max()
wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6]
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6]['filename']
str(wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6]['filename'])
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6]['filename'].loc[0]
wbds_atx_fns[wbds_atx_fns['filename'].apply(lambda filename: len(filename))==6]['filename'].iloc[0]
wbds_atx_fns
wbds_atx_fns['filename'].apply(lambda filename: len(filename))
wbds_atx_fns['filename'].apply(lambda filename: len(filename)).sum()
wbds_atx_fns['filename'].apply(lambda filename: len(filename)).sum()/4692.
4692./wbds_atx_fns['filename'].apply(lambda filename: len(filename)).sum()
wbds_atx_fns['filename'].apply(lambda filename: filename[0])
wbds_atx_fns['filename'].apply(lambda filename: filename[0]).tolist
wbds_atx_fns['filename'].apply(lambda filename: filename[0]).tolist()
wbds_atx_fns_list = wbds_atx_fns['filename'].apply(lambda filename: filename[0]).tolist()
wbds_atx_fns_list
len(wbds_atx_fns_list)
os.getcwd()
os.listdir()
with open('TWDB-Basins-HUC12s_distance_from_ATX.txt','w') as f:
    for item in wbds_atx_fns_list:
        f.write("%s\n" % item)
wbds_atx_fns_list
import readline
readline.write_history_file('TWDB-Basins-HUC12s_distance_from_ATX-history.py')
2176./32.
202*1.0825
202*1.0825*1.1
4.5/7
4.5/7*4.5
4.5/(7/2)*4.5
2.5/(7/2)*4.5
3.5/(7/2)*4.5
(3+1/3)/(7/2)*4.5
5000/.4
220*8
220*8*1.59
220*8/40
220*8/40*11.68
220*8/40*11.68*1.0825
90*24
90*20
2500/36*24
2500/36*24*.1
2500/36*20
dir()
41664./868
41664./868.
dir()
wbs
wbds
wbds[0]
wbds.loc[0]
wbds
wbds.columns
wbds['SHAPE_Area'].unique()
wbds['SHAPE_Area'].unique().sort
wbds['SHAPE_Area'].unique().sort()
wbds['SHAPE_Area'].unique()
wbds.index
wbds
wbds['SHAPE_Area'].unique()<0.02
wbds['SHAPE_Area'].unique()<0.01
wbds['SHAPE_Area'].unique()<0.015
wbds[wbds['SHAPE_Area'].unique()<0.015]
wbds[wbds['SHAPE_Area'].unique()<0.015][0]
wbds[wbds['SHAPE_Area'].unique()<0.015,0]
wbds.loc[wbds['SHAPE_Area'].unique()<0.015,0]
wbds.loc[wbds['SHAPE_Area'].unique()<0.015]
wbds.loc[wbds['SHAPE_Area'].unique()<0.015][0]
wbds.loc[0,wbds['SHAPE_Area'].unique()<0.015]
wbds.loc[wbds['SHAPE_Area'].unique()<0.015].loc[0]
wbds.loc[wbds['SHAPE_Area'].unique()<0.015]
3300./2.
5*7
400-35
(400-35)/3
860.*1.1
385+80+60
dir()
wbds.loc[wbds['SHAPE_Area'].unique()<0.015]
920.-776.
wbds.loc[wbds['SHAPE_Area'].unique()<0.015]
wbds.loc['SHAPE_Area'][0]
wbds['SHAPE_Area'][0]
wbds['SHAPE_Area']
57120./50./40.
dir()
actual_tasks = [{0:{'queue':'development'}},{1:{'queue':'normal'}}]
actual_tasks
for p in actual_tasks:
    p
for p in actual_tasks:
    p.keys[0]
for p in actual_tasks:
    p.keys
for p in actual_tasks:
    p.keys()
for p in actual_tasks:
    p.keys()[0]
    list(p.keys())[0]
for p in actual_tasks:
    list(p.keys())[0]
deque(actual_tasks)
from collections import deque
deque(actual_tasks)
for p in deque(actual_tasks):
    p
p
running_tasks = deque(actual_tasks)
running_tasks
running_tasks.remove
running_tasks.remove(p)
running_tasks
p
p['queue']
p[list(p.keys())[0]]
p[list(p.keys())[0]]['queue']
dir()
os.getcwd()
os.listdir()
pd
import pandas as pd
pd.read_csv('hand-taudem.log')
p_logcsv = pd.read_csv('hand-taudem.log')
p_logcsv['elapsed_time'][0]
p_logcsv
csv
csv = pd.DataFrame(columns=['queue','elapsed_time','error_long_queue_timeout','complete'])
csv
type(csv)
dir()
p
list(p.keys())[0].poll()
list(p.keys())[0]
pd
bashCmd
p
p_job_id = 7033323
from io import StringIO
bashCmd = "squeue -j " + p_job_id
bashCmd = "squeue -j " + str(p_job_id)
bashCmd
subprocess
import subprocess
process = subprocess.Popen(bashCmd.split(),stdout=subprocess.PIPE)
process
output,error = process.communicate()
error
print(error)
output
pd.read_csv(StringIO(output))
import io
pd.read_csv(io.BytesIO(output))
job_id = pd.read_csv(io.BytesIO(output))
job_id
job_id.columns
job_id.iloc[0]
job_id = pd.read_csv(io.BytesIO(output),sep="\s+")
job_id
bashCmd = "squeue -j " + str(1)
process = subprocess.Popen(bashCmd.split(),stdout=subprocess.PIPE)
error
output,error = process.communicate()
error
print(error)
output
process = subprocess.Popen(bashCmd.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
output,error = process.communicate()
error
output
bashCmd = "squeue -j " + str(p_job_id)
bashCmd
process = subprocess.Popen(bashCmd.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
output,error = process.communicate()
error
output
if error is None:
    print(True)
if error is '':
    print(True)
if error == None:
    print(True)
if error == '':
    print(True)
if error == b'':
    print(True)
if error is b'':
    print(True)
dir()
p_logcsv
process
running_tasks
csv
job_id
squeue = job_id.copy()
squeue
squeue['PARTITION']
squeue['PARTITION']=='normal'
squeue['PARTITION']=='skx-normal'
(squeue['PARTITION']=='skx-normal').sum()
squeue.sum()
squeue.shape
squeue.shape[1]
squeue.shape[0]
logcsv
p_logcsv
p_logcsv.loc[0]
p_logcsv.loc[0,'error_long_queue_timeout']
p_logcsv.index = [1]
p_logcsv
p_logcsv.loc[0,'error_long_queue_timeout']
p_logcsv.index[0]
index
p_logcsv.loc[0,'error_long_queue_timeout']
p_logcsv.iloc[0,'error_long_queue_timeout']
p_logccsv
p_logcsv
p_logcsv.index
p_logcsv.index.name
dir()
os.listdir()
readline.write_history_file('hand_taudem-history-20210114.py')
48.*50.
dir()
os.getcwd()
import matplotlib
exit()
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import glob
import matplotlib
filename='hand-taudem-cmds.sh'
with open(filename) as f:
    content = f.readlines()
content = [x.strip() for x in content]
log = pd.read_csv('hand-taudem.log')
logs = glob.glob('*/*/hand-taudem.log')
logs
last_cmds = [log_gpd.loc[0,'last_cmd'] for log_gpd in [pd.read_csv(log) for log in logs]]
last_cmds
pct_finished = (np.array([np.array([ i for i, word in enumerate(content) if word.startswith(lastcmd) ]).max() for lastcmd in last_cmds])+1)/len(content)
content
content.insert(0,'touch')
content
pct_finished = (np.array([np.array([ i for i, word in enumerate(content) if word.startswith(lastcmd) ]).max() for lastcmd in last_cmds])+1)/len(content)
catchments = glob.glob('*/*/Catchments.shp')
catchments
catchments_gpd = [gpd.read_file(catchment) for catchment in catchments]
hucs = gpd.GeoDataFrame([catchment.dissolve(by=['HUC12']).iloc[0] for catchment in catchments_gpd])
hucs['pct_finished'] = pct_finished
hucs.plot(column='pct_finished',cmap='cividis',legend=True)
hucs.
hucs.plot(column='pct_finished',cmap='cividis',legend=True).savefig
plt = hucs.plot(column='pct_finished',cmap='cividis',legend=True)
plt
plt.savefig
plt.getfigure()
plt.get)figure()
plt.get_figure()
fig = plt.get_figure()
fig
os.listdir()
fig.savefig("hand_taudem-progress-vis.png")
os.getcwd()
os.listdir()
3600.*12.*.12
3600.*12.*.08
2400000./5000.
import contextily as ctx
def visualize(gdfs,edgecolors=False,units=7.):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i]
        )
    ctx.add_basemap(ax)
    plt.show()
import contextily as ctx
def visualize(gdfs,edgecolors=False,units=7.):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i]
        )
    ctx.add_basemap(ax)
    plt.show()
def visualize(gdfs,edgecolors=False,units=7.):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i]
        )
    ctx.add_basemap(ax)
    plt.show()
dir()
hucs
visualize([hucs],edgecolors=['k'],units=10.)
plt
plot = plt
import matplotlib as plt
visualize([hucs],edgecolors=['k'],units=10.)
import matplotlib.pyplot as plt
visualize([hucs],edgecolors=['k'],units=10.)
hucs.crs
hucs.geometry
catchments.crs
catchments[0].crs
dir()
catchments_gpd[0]
catchments_gpd[0].crs
hucs.crs = catchments_gpd[0].crs
visualize([hucs],edgecolors=['k'],units=10.)
args = type('', (), {})()
os.getcwd()
os.listdir()
args.path_hand_image = hand_taudem-progress_vis0.png
args.path_hand_image = 'hand_taudem-progress_vis0.png'
def visualize(gdfs,edgecolors=False,units=7.):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i]
        )
    ctx.add_basemap(ax)
    plt.show()
    fig = plt.get_figure()
    fig.savefig(args.path_hand_image)
visualize([hucs],edgecolors=['k'],units=10.)
def visualize(gdfs,edgecolors=False,units=7.):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i]
        )
    ctx.add_basemap(ax)
    plt.show()
    f.savefig(args.path_hand_image)
visualize([hucs],edgecolors=['k'],units=10.)
args.path_hand_image
visualize([hucs],edgecolors=['k'],units=100.)
def visualize(gdfs,edgecolors=False,units=7.,col='pct_finished'):
    f, ax = plt.subplots(figsize=(
        units,
        gdfs[0].shape[1]/gdfs[0].shape[0]*units
    ))
    if edgecolors==False:
        edgecolors = ['k']*len(gdfs)
    for i, gdf in enumerate(gdfs):
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            figsize=(
                units,
                gdfs[0].shape[1]/gdfs[0].shape[0]*units
            ),
            alpha=.5,
            edgecolor=edgecolors[i],
            column=col,
            cmap='cividis',
            legend=True
        )
    ctx.add_basemap(ax)
    plt.show()
    f.savefig(args.path_hand_image)
visualize([hucs],edgecolors=['k'],units=50.)
38.*5000.
19.*5000.
5000./19.
3500./12.*2.5
3680./12.*2.5
dir()
hucs
catchments_gpd
const
const()
constant
[catchment['const']=True for catchment in catchments_gpd]
[catchment for catchment in catchments_gpd]
[catchment['HUC12'] for catchment in catchments_gpd]
[catchment['const'] for catchment in catchments_gpd]
[type(catchment) for catchment in catchments_gpd]
catchment
catchment = catchments_gpd[0]
catchment
gpd.GeoDataFrame(catchments_gpd)
gpd.GeoDataFrame(catchments_gpd).shape[0]
catchments_gpd
len(catchments_gpd)
gpd.GeoDataFrame(catchments_gpd,ignore_index=True)
catchment.columns
catchment
catchment['index']
catchment.shape[0]
np.apply([catchment.shape[0] for catchment in catchments_gpd]).sum()
np.array([catchment.shape[0] for catchment in catchments_gpd]).sum()
catchment
catchment['FEATUREID'].array
[catchment['FEATUREID'].array for catchment in catchments_gpd]
np.flatten([catchment['FEATUREID'].array for catchment in catchments_gpd])
np.array([catchment['FEATUREID'].array for catchment in catchments_gpd]).flatten()
catchments_gpd
gpd.GeoDataFrame(catchments_gpd)
np.array([catchment['HUC12'] catchment in catchments_gpd]).unique()
np.array([catchment['HUC12'] for catchment in catchments_gpd]).unique()
np.array([catchment['HUC12'] for catchment in catchments_gpd])
np.array([catchment['HUC12'].tolist() for catchment in catchments_gpd])
np.array([catchment['HUC12'].array() for catchment in catchments_gpd])
np.array([catchment['HUC12'].array for catchment in catchments_gpd])
np.array([catchment['HUC12'].array for catchment in catchments_gpd]).flatten()
np.array([catchment['HUC12'].array for catchment in catchments_gpd]).flatten2d()
np.array([catchment['HUC12'].array for catchment in catchments_gpd]).flatten().flatten()
np.array([catchment['HUC12'].array for catchment in catchments_gpd]).shape[0]
np.array([catchment['HUC12'].array for catchment in catchments_gpd])
pd.concat(catchments_gpd)
pd.concat(catchments_gpd,ignore_index=True)
gpd.GeoDataFrame(pd.concat(catchments_gpd,ignore_index=True),crs=catchments_gpd[0].crs)
gpd.GeoDataFrame(pd.concat(catchments_gpd,ignore_index=True),crs=catchments_gpd[0].crs)['FEATUREID'].unique()
len(gpd.GeoDataFrame(pd.concat(catchments_gpd,ignore_index=True),crs=catchments_gpd[0].crs)['FEATUREID'].unique())
gpd.GeoDataFrame(pd.concat(catchments_gpd,ignore_index=True),crs=catchments_gpd[0].crs)
catchments_gdf = gpd.GeoDataFrame(pd.concat(catchments_gpd,ignore_index=True),crs=catchments_gpd[0].crs)
os.getcwd()
os.listdir()
catchments_gdf.to_file('hand_taudem-progress_vis-catchments.shp')
ls
dir()
os.getcwd()
os.listdir()
wbds
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TX-County_boundaries.shp/County.shp')
couties
counties
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TXDoT-Counties-GLO-Harvey.shp/TXDoT-Counties-GLO-Harvey.shp')
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TXDoT-Counties-GLO-Harvey.shp/TXDoT-Counties-GLO-Harvey.shp',driver='Shapefile')
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TXDoT-Counties-GLO-Harvey.shp',driver='Shapefile')
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TXDoT-Counties-GLO-Harvey.shp')
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TX-County_boundaries.shp')
2020.-1959.
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TX-County_boundaries.shp')
counties = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/TXDoT-Counties-GLO-Harvey.shp')
counties
wbds
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
wbds
gpd.sjoin(wbds,counties,how='intersects',op='inner')
gpd.sjoin(wbds,counties,how='inner',op='intersects')
gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects')
wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index]
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp',mask=counties)
wbds
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects')
wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index]
wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index].columns
wbds = wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index].columns
counties
counties.columns
counties.nunique()
const
counties['const'] = True
counties_rep = atx.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]
counties_rep = counties.dissolve('const').reset_index(drop=True).to_crs(wbds.crs).representative_point()[0]
counties_rep = counties.dissolve('const')
counties_rep
counties_rep.reset_index(drop=True)
counties_rep = counties_rep.reset_index(drop=True)
counties_rep.to_crs(wbds.crs)
wbds
wbds = wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index]
wbds = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
wbds = wbds.iloc[gpd.sjoin(wbds,counties.to_crs(wbds.crs),how='inner',op='intersects').index]
counties_rep.to_crs(wbds.crs)
counties_rep = counties_rep.to_crs(wbds.crs)
counties_rep.representative_point()[0]
counties_rep = counties_rep.representative_point()[0]
wbds_counties = wbds.iloc[(wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index)]
wbds.iloc[wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index]
wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index
wbds.loc[wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index]
wbds.loc[(wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index)]
wbds.loc[wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index]
wbds.loc[(wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index)]
wbds.loc[wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index]
wbds.loc[(wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index)]
wbds_counties = wbds.loc[(wbds.representative_point().apply(lambda wbd: wbd.distance(counties_rep)).sort_values().index)]
wbds_counties
wbds_counties['filename'] = wbds_counties['HUC12'].apply(lambda huc12: glob.glob(os.path.join('*','*'+str(huc12),'Elevation.tif')))
wbds_counties_fns = wbds_counties.copy()
wbds_counties_fns = wbds_counties_fns[wbds_counties_fns['filename'].apply(lambda filename: len(filename))) > 0]
wbds_counties_fns = wbds_counties_fns[wbds_counties_fns['filename'].apply(lambda filename: len(filename)) > 0]
wbds_counties_fns
wbds_counties['filename']
os.getcwd()
os.listdir()
os.chdir('/scratch/projects/tnris/dhl-flood-modelling')
wbds_counties['filename'] = wbds_counties['HUC12'].apply(lambda huc12: glob.glob(os.path.join('*','*'+str(huc12),'Elevation.tif')))
wbds_counties['filename']
wbds_counties_fns = wbds_counties.copy()
wbds_counties_fns = wbds_counties_fns[wbds_counties_fns['filename'].apply(lambda filename: len(filename)) > 0]
wbds_counties_fns
wbds_counties_fns['filename'].apply(lambda filename: filename[0])
wbds_counties_fns['filename'].apply(lambda filename: filename)
wbds_counties_fns['filename'].apply(lambda filename: len(filename))
wbds_counties_fns['filename'].apply(lambda filename: filename)[0]
wbds_counties_fns['filename'].apply(lambda filename: filename).iloc[0]
40230.80/38681.42
2614.47/(188.27*9.+83.33*12.)
2614.47/(188.27*9.+83.33*12.+100.)
2614.47/(188.27*9.+83.33*12.+116.*12.)
(188.27*9.+83.33*12.+116.*12.)
(188.27*9.+83.33*12.+116.*12.)*1.15
(188.27*9.+116.*12.)*1.15
2614.47/(188.27*9.+116.*12.)
2614.47/(188.27*9.)
dir()
wbds_counties_fns
cd
wbds_counties_fns_list = wbds_counties_fns['filename'].apply(lambda filename: filename[0]).tolist()
wbds_counties_fns_list
len(wbds_counties_fns_list)
gdf
gdf = gpd.read_file('/scratch/04950/dhl/HAND-TauDEM/regions/Texas/TWDB-Basins/hand_taudem-progress_vis.shp')
gdf
gdf.columns
gdf['level_0'].unique()
gdf['level_0'].unique().shape[0]
gdf['pct_finish']
gdf['pct_finish'].max()
gdf['pct_finish'].sum()
gdf['pct_finish'].max()*gdf.shape[0]
gdf['pct_finish'].sum()/(gdf['pct_finish'].max()*gdf.shape[0])
gdf.columns
gdf = gpd.read_file('/scratch/04950/dhl/HAND-TauDEM/regions/Texas/TWDB-Basins/hand_taudem-progress_vis.shp')
gdf.columns
gdfs['logfiles']
gdf['logfiles']
logs
logs = gdf['logfiles'].apply(lambda log: pd.read_csv(log))
logs
logs.columns
logs = [pd.read_csv(log) for log in gdf['logfiles']]
logs[0].columns
logs = [{log: pd.read_csv(log)} for log in gdf['logfiles']]
logs
[log['logfile']= for log in logs]
logs = [{log: pd.read_csv(log)} for log in gdf['logfiles']]
logs[0][1]
logs[0,1]
logs[0;1]
logs[0]
logfiles
logs
logs = [pd.read_csv(log) for log in gdf['logfiles']]
gdf['logfiles']
[log['logfile']=gdf['logfiles'].iloc[i] for i,log in enumerate(logs)]
[log.join(pd.DataFrame({'logfile':logfile})) for log,logfile in zip(logs,gdf['logfiles'])]
logs
[log.join(pd.DataFrame({'logfile':logfile})) for log,logfile in zip(logs,gdf['logfiles'].tolist())]
[log.index for log in logs]
[log.join(pd.DataFrame({'logfile':repeat(logfile)})) for log,logfile in zip(logs,gdf['logfiles'].tolist())]
from itertools import repeat
[log.join(pd.DataFrame({'logfile':repeat(logfile)})) for log,logfile in zip(logs,gdf['logfiles'].tolist())]
[log.join(pd.DataFrame({'logfile':[logfile]})) for log,logfile in zip(logs,gdf['logfiles'].tolist())]
logs = [log.join(pd.DataFrame({'logfile':[logfile]})) for log,logfile in zip(logs,gdf['logfiles'].tolist())]
[log['logfile'].fillna(log['logfile'].iloc[0]) for lof in logs]
[log['logfile'].fillna(log['logfile'].iloc[0]) for log in logs]
[log['logfile'].fillna(log['logfile'].iloc[0],inplace=True) for log in logs]
logs
np.array([log['elapsed_time'].max() for log in logs]).sum()
dir()
wbds_counties
wbds
wbds.columns
dir()
wbds_counties
wbds_counties.columns
wbds_counties['filename']
dir()
gdf
dir()
hucs
hucs.columns
dir()
gpd
progress = gpd.read_file('/scratch/04950/dhl/HAND-TauDEM/regions/Texas/TWDB-Basins/hand_taudem-progress_vis.shp')
progress
progress.columns
progress['index']
progress['logfiles']
progress
progress.columns
progress.loc[0,'logfiles']
progress['logfiles'].unique()
progress['logfiles'].unique().shape[0]
progress['logfiles'].apply(lambda fn: '/scratch/projects/tnris/dhl-flood-modelling/'+os.path.dirname(fn)+'Elevation.tif')
progress['logfiles'].apply(lambda fn: '/scratch/projects/tnris/dhl-flood-modelling/'+os.path.dirname(fn)+'Elevation.tif').loc[0,logfiles']
progress['logfiles'].apply(lambda fn: '/scratch/projects/tnris/dhl-flood-modelling/'+os.path.dirname(fn)+'Elevation.tif').loc[0,'logfiles']
progress['logfiles'].apply(lambda fn: '/scratch/projects/tnris/dhl-flood-modelling/'+os.path.dirname(fn)+'Elevation.tif').loc[0]
progress['logfiles'].apply(lambda fn: os.path.join('/scratch/projects/tnris/dhl-flood-modelling',os.path.dirname(fn),'Elevation.tif')).loc[0]
progress['logfiles'].apply(lambda fn: os.path.join('/scratch/projects/tnris/dhl-flood-modelling',*os.path.split(os.path.dirname(fn)),'Elevation.tif')).loc[0]
progress['logfiles'].apply(lambda fn: os.path.join('/scratch/projects/tnris/dhl-flood-modelling',*os.path.split(os.path.dirname(fn)),'Elevation.tif'))
progress['elevationfiles'] = progress['logfiles'].apply(lambda fn: os.path.join('/scratch/projects/tnris/dhl-flood-modelling',*os.path.split(os.path.dirname(fn)),'Elevation.tif'))
progress.__path__
progress.file
progress.name
progress.to_file('/scratch/04950/dhl/HAND-TauDEM/regions/Texas/TWDB-Basins/hand_taudem-progress_vis.shp')
progress['elevationfiles'].to_csv()
progress['elevationfiles'].to_csv(index=False)
progress['elevationfiles'].to_csv('/scratch/projects/tnris/dhl-flood-modelling/TWDB-Basins-HUC12s_Central_TX.txt',index=False)
os.getcwd()
import readline
readline.write_history_file('hand_taudem-history-20210204.py')
os.getcwd()
os.listdir()
Path('.').rglob('Elevationdd-vis.tif')
from pathlib import Path
Path('.').rglob('Elevationdd-vis.tif')
list(Path('.').rglob('Elevationdd-vis.tif')
list(Path('.').rglob('Elevationdd-vis.tif'))
[os.dirname(fn) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[os.path.dirname(fn) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[os.path.join(*os.path.split(os.path.dirname(fn)), for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[os.path.join(*os.path.split(os.path.dirname(fn)),Catchments.shp) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp') for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[Path.glob(os.path.join(*os.path.split(os.path.dirname(fn))),'Catchments.shp') for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[*list(Path.glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp')) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp') for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[*Path.glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp')) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[*list(Path.glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp'))) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[list(Path.glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp'))) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[list(glob.glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp'))) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[list(Path('.').glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp'))) for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[list(Path('.').glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp')))[0] for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
catchments = [list(Path('.').glob(os.path.join(*os.path.split(os.path.dirname(fn)),'Catchments.shp')))[0] for fn in list(Path('.').rglob('Elevationdd-vis.tif'))]
[gpd.read_file(catchment) for catchment in catchments]
catchments_fns = catchments.copy()
catchments = [gpd.read_file(catchment) for catchment in catchments]
catchments
pd.concat(catchments)
pd.concat(catchments).columns
gpd.GeoDataFrame(pd.concat(catchments),crs=catchments[0].crs)
catchments = gpd.GeoDataFrame(pd.concat(catchments),crs=catchments[0].crs)
catchments
catchments.to_file('catchments_HAND.geojson',driver='GeoJSON')
os.getcwd()
catchments['HUC12'].unique().shape[0]
catchments.columns
catchments['GRIDCODE'].unique().shape[0]
catchments['SOURCEFC'].unique().shape[0]
catchments.dissolve(by=['SOURCEFC'])
catchments_convex = catchments.dissolve(by=['SOURCEFC'])
catchments_convex.reset_index()
catchments_convex.reset_index(inplace=True)
catchments_convex.geometry.convex_hull
catchments_convex.geometry = catchments_convex.geometry.convex_hull
catchments_convex.to_file('catchments_HAND_convex.geojson',driver='GeoJSON')
wbd_tx = gpd.read_file('/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD-HU12-TX.shp')
wbd_tx['HUC12'].unique().shape[0]
gpd.overlay(wbd_tx['HUC12'].to_crs(catchments_convex.crs),catchments_convex,how='within')
gpd.overlay(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,how='within')
gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,how='within',op='inner')
gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner')
gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='contains',how='inner')
gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner')
gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='intersects',how='inner')
wbd_tx[[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]]
wbd_tx.loc[[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]]
wbd_tx.iloc[[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]]
wbd_tx.iloc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]
wbd_tx.loc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]
wbd_ctx = wbd_tx.loc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),catchments_convex,op='within',how='inner').index]
wbd_ctx.to_file('wbd_ctx.geojson',driver='GeoJSON')
cbsas = gpd.read_file('/scratch/04950/dhl/HAND-TauDEM/regions/United_States/US-CBSA-2019.shp')
cbsas
cbsas.columns
cbsas['NAME'].str.contains('Austin')
cbsas['NAME'].str.contains('Austin').sum()
cbsas[cbsas['NAME'].str.contains('Austin')]
cbsas[cbsas['NAME'].str.contains('Austin')]['NAME']
cbsas[cbsas['NAME'].str.contains('Austin&TX')]['NAME']
cbsas[cbsas['NAME'].str.contains('Austin-Round Rock-Georgetown, TX')]['NAME']
cbsas[cbsas['NAME'].str.contains('Austin-Round Rock-Georgetown, TX')]
cbsa_atx = cbsas[cbsas['NAME'].str.contains('Austin-Round Rock-Georgetown, TX')]
cbsa_atx
wbd_ctx = wbd_tx.loc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),cbsa_atx.to_crs(catchments_convex.crs),op='within',how='inner').index]
wbd_ctx
wbd_ctx.to_file('wbd_ctx.geojson',driver='GeoJSON')
wbd_ctx = wbd_tx.loc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),cbsa_atx.to_crs(catchments_convex.crs),op='intersection',how='inner').index]
wbd_ctx = wbd_tx.loc[gpd.sjoin(wbd_tx.to_crs(catchments_convex.crs),cbsa_atx.to_crs(catchments_convex.crs),op='intersects',how='inner').index]
wbd_ctx.to_file('wbd_ctx.geojson',driver='GeoJSON')
wbd_cts.shape[0]
wbd_ctx.shape[0]
wbd_ctx['HUC12']-catchments['HUC12']
set(wbd_ctx['HUC12'])-set(catchments['HUC12'])
len(set(wbd_ctx['HUC12'])-set(catchments['HUC12']))
len(set(wbd_ctx['HUC12']) & set(catchments['HUC12']))
55.*38.
set(wbd_ctx['HUC12'])-set(catchments['HUC12'])
hucs_remaining = set(wbd_ctx['HUC12'])-set(catchments['HUC12'])
hucs_remaining
Path('.').rglob(hucs_remaining)
list(Path('.').rglob(hucs_remaining))
[Path('.').rglob(huc) for huc in hucs_remaining]
[list(Path('.').rglob(huc)) for huc in hucs_remaining]
huc
hucs_remaining[0]
hucs_remaining = list(hucs_remaining)
hucs_remaining[0]
Path('.').rglob(hucs_remaining[0])
list(Path('.').rglob(hucs_remaining[0]))
os.getcwd()
list(Path('.').rglob('*'+hucs_remaining[0]))[0]
[list(Path('.').rglob('*'+hucs_remaining[0]))[0] for huc in hucs_remaining]
[list(Path('.').rglob('*'+huc))[0] for huc in hucs_remaining]
[list(Path('.').rglob('*'+huc))[0] for huc in hucs_remaining if len(list(Path('.').rglob('*'+huc)))>0]
[list(Path('.').rglob('*'+huc)) for huc in hucs_remaining]
import readline
readline.write_history_file('list_of_remaining_hucs-history.py')
exit()
import os
os.getcwd()
exit()
import pyspatialml as pml
exit()
import geopandas as gpd
import rasterio
from rasterio import features
gdf_fn = os.path.join(
    os.sep,
    'scratch',
    '04950',
    'dhl',
    'HAND-TauDEM',
    'regions',
    'Texas',
    'ATX-Floodplain',
    'geo_export_1d5ff87d-6f60-4cd5-a1fe-9e06f315ae51.shp'
)
import os
gdf_fn = os.path.join(
    os.sep,
    'scratch',
    '04950',
    'dhl',
    'HAND-TauDEM',
    'regions',
    'Texas',
    'ATX-Floodplain',
    'geo_export_1d5ff87d-6f60-4cd5-a1fe-9e06f315ae51.shp'
)
gdf_fn
floodplains = gpd.read_file(gdf_fn)
floodplains_utm = floodplains.to_crs('EPSG:32614')
floodplains_utm['Year'] = 0
floodplains_utm.loc[
    floodplains_utm['flood_zone'].str.contains('25'),
    'Year'
] = 25
floodplains_utm.loc[
    floodplains_utm['flood_zone'].str.contains('100'),
    'Year'
] = 100
floodplains_utm.drop(
    index = floodplains_utm[floodplains_utm['Year']==0].index,
    inplace = True
)
floodplains_utm
crop_fn = os.path.join(os.sep,'scratch','projects','tnris','dhl-flood-modelling','TWDB-Basin-Colorado','TWDB-Basin-Colorado-120902050307','Catchments.shp')
crop_fn
crop
crop = gpd.read_file(crop_fn)
crop
crop.columns
crop['SOURCEFC'].unique()
crop.dissolve(by=['SOURCEFC'])
crop.dissolve(by=['SOURCEFC']).reset_index()
crop.dissolve(by=['SOURCEFC'],inplace=True).reset_index(inplace=True)
crop.dissolve(by=['SOURCEFC']).reset_index(inplace=True)
crop
crop = crop.dissolve(by=['SOURCEFC']).reset_index()
crop
crop.geometry
crop
floodplains = gpd.read_file(gdf_fn,mask=crop)
floodplains
floodplains_utm = floodplains.to_crs('EPSG:32614')
floodplains_utm['Year'] = 0
floodplains_utm.loc[
    floodplains_utm['flood_zone'].str.contains('25'),
    'Year'
scratch_path = os.path.join(
    os.sep,
    'scratch',
    '04950',
    'dhl'
)
basin_path = os.path.join(
    os.sep,
    'scratch',
    'projects',
    'tnris',
    'dhl-flood-modelling',
    'TWDB-Basin-Colorado',
    'TWDB-Basin-Colorado-120902050307',
)
gdf_fn = os.path.join(
    scratch_path,
    'HAND-TauDEM',
    'regions',
    'Texas',
    'ATX-Floodplain',
    'geo_export_1d5ff87d-6f60-4cd5-a1fe-9e06f315ae51.shp'
)
crop_fn = os.path.join(
    basin_path,
    'Catchments.shp'
)
gdf_fn
crop_fn
crop = gpd.read_file(crop_fn)
crop = crop.dissolve(by=['SOURCEFC']).reset_index()
floodplains = gpd.read_file(gdf_fn,mask=crop)
floodplains
gpd.overlay(floodplains,crop,how='intersects')
gpd.overlay(floodplains,crop,how='intersection')
floodplains_utm = floodplains.to_crs('EPSG:32614')
gpd.overlay(floodplains_utm,crop,how='intersection')
exit()
eit
exit
exit()
eit
exit
exit()
import geopandas as gpd
import os
os.listdir()
catchshu12shape = gpd.read_file('catchshu12shape.geojson')
catchshu12shape
flowshu12shape = gpd.read_file('flowshu12shape.geojson')
flowshu12shape_rep = gpd.read_file('flowshu12shape_rep.geojson')
flowshu12shape_rep
catchshu12shape
flows = gpd.read_file('flows.geojson')
flows
flowshu12shape = flows[flows.index.isin(catchshu12shape.index)]
flowshu12shape
catchshu12shape
catchs.to_file('catchs.geojson',driver='GeoJSON')
flows
flowshu12shape
flowshu12shape.columns
catchshu12shape.columns
catchs
catchs = gpd.read_file('catchs.geojson')
os.listdir()
catchs = gpd.read_file('NFIEGeo_TX.gdb',layer='Catchments',mask=hu12)
shape = gpd.read_file('TX-Counties-Seven_NOAA.shp')
shape.drop(columns=['Shape_Leng','Shape_Area'])
shape.drop(columns=['Shape_Leng','Shape_Area'],inplace=True)
shape.rename(columns={'HUC12':'HUC12_shapefile'})
shape.rename(columns={'HUC12':'HUC12_shapefile'},inplace=True)
os.listdir()
hu12 = gpd.read_file('WBD-HU12-TX.shp',mask=shape)
hu12[['HUC12','geometry']]
hu12 = hu12[['HUC12','geometry']]
flows 
catchs = gpd.read_file('NFIEGeo_TX.gdb',layer='Catchments',mask=hu12)
catchs = gpd.read_file('NFIEGeo_TX.gdb',layer='Catchment',mask=hu12)
catchs
catchs.reset_index(inplace=True)
catchs.set_index('FEATUREID',inplace=True)
catchs.sort_index(inplace=True)
catchs
catchs.index.unique().shape[0]
catchs[catchs.index.isin(flowshu12shape_rep.index)]
flowshu12shape_rep
flowshu12shape_rep.set_index('COMID',inplace=True)
flowshu12shape.sort_index(inplace=True)
catchs[catchs.index.isin(flowshu12shape_rep.index)]
catchshu12shape
flowshu12shape
flows_rep = flows.copy()
flows_rep.geometry = flows.representative_point()
gpd.sjoin(flows_rep,hu12.to_crs(flows_rep.crs),op='intersects',how='inner')
flowshu12shape_rep
gpd.sjoin(flows_rep,hu12.to_crs(flows_rep.crs),op='intersects',how='inner')
flows_rep
flowshu12shape_rep
flows_rep
flowshu12shape
flowshu12shape['COMID']==5785385
(flowshu12shape['COMID']==5785385).sum()
flows_rep.columns
flows
flows.columns
flows.set_index('COMID',inplace=True)
flows.sort_index(inplace=True)
flowshu12shape_rep
flows
flowshu12shape
flows[flows.index.isin(flowshu12shape_rep.index(]
flows[flows.index.isin(flowshu12shape_rep.index)]
flowshu12shape
os.getcwd()
import readline
readline.write_history_file('find_holes-history0.py')
flows[flows.index.isin(flowshu12shape_rep.index(]
flows[flows.index.isin(flowshu12shape_rep.index)]
flows[flows.index.isin(flowshu12shape_rep.index)].loc[5785385]
flows[flows.index.isin(flowshu12shape_rep.index)].iloc[5785385]
flows[flows.index.isin(flowshu12shape_rep.index)][5785385]
flows[flows.index.isin(flowshu12shape_rep.index)]
flows[flows.index.isin(flowshu12shape_rep.index)].index==5785385
(flows[flows.index.isin(flowshu12shape_rep.index)].index==5785385).sum()
(flows[flows.index.isin(catchshu12shape.index)].index==5785385).sum()
(flowshu12shape.index==5785385).sum()
\
import geopandas as gpd
import os
shape = gpd.read_file('TX-Counties-Seven_NOAA.shp')
shape.drop(columns=['Shape_Leng','Shape_Area'],inplace=True)
shape.rename(columns={'HUC12':'HUC12_shapefile'},inplace=True)
hu12 = gpd.read_file('WBD-HU12-TX.shp',mask=shape)
hu12 = hu12[['HUC12','geometry']]
flows = gpd.read_file('NFIEGeo_TX.gdb',layer='Flowline',mask=hu12)
flows.drop(columns=['Shape_Length','Shape_Area','AreaSqKM','index_left','index_right'],inplace=True,errors='ignore')
flows.reset_index(inplace=True)
flows.set_index('COMID',inplace=True)
flows.sort_index(inplace=True)
flows_rep = flows.copy()
flows_rep.geometry = flows.representative_point()
flowshu12shape_rep = gpd.sjoin(flows_rep,hu12.to_crs(flows_rep.crs),op='intersects',how='inner')
flowshu12shape_rep.drop(columns=['index_right'],inplace=True)
catchs = gpd.read_file('NFIEGeo_TX.gdb',layer='Catchment')
catchs.reset_index(inplace=True)
catchs.set_index('FEATUREID',inplace=True)
catchs.sort_index(inplace=True)
catchshu12shape = catchs[catchs.index.isin(flowshu12shape_rep.index)]
flowshu12shape = flows[flows.index.isin(flowshu12shape_rep.index)]
flowshu12shape_orig = flowshu12shape.copy()
flowshu12shape = flowshu12shape[flowshu12shape.index.isin(catchshu12shape.index)]
catchshu12shape_orig = catchshu12shape.copy()
catchshu12shape.geometry = catchshu12shape.buffer(0)
flowshu12shape.geometry = flowshu12shape.buffer(0)
flowshu12shape
flowshu12shape = flowshu12shape_orig.copy()
flowshu12shape = flowshu12shape[flowshu12shape.index.isin(catchshu12shape.index)]
flowshu12shape
flowshu12shape.crs
catchshu12shape.crs
flowshu12shape.buffer(0)
flowshu12shape
catchshu12shape
flowshu12shape[flowshu12shape.index.isin(catchshu12shape.index)]
catchshu12shape[catchshu12shape.index.isin(flowshu12shape.index)]
catchshu12shape.to_file('catchshu12shapebuffer.geojson',driver='GeoJSON')
exit()
exit
exit()
import geopandas as gpd
Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCS.shp')
from pathlib import Path
Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCS.shp')
list(Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCS.shp'))
list(Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCs.shp'))
len(list(Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCs.shp')))
list(Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCs.shp'))
hucs = list(Path('/scratch/04950/dhl/GeoFlood/GeoFlood/Shapes').rglob('*_HUCs.shp'))
gdfs = [gpd.read_file(huc) for huc in hucs]
gdfs[0].columns
[gdf.reset_index(inplace=True) for gdf in gdfs]
gdfs
[gdf.rename(columns={'index':'FID'},inplace=True) for gdf in gdfs]
gdfs
[gdf.crs.to_epsg() for gdf in gdfs]
[gdf.crs for gdf in gdfs]
gdfs
[gdf.crs.to_epsg() for gdf in gdfs]
[gdf.to_crs('EPSG:4269',inplace=True) for gdf in gdfs]
gdfs
[gdf.crs.to_epsg() for gdf in gdfs]
gdf.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=gdfs[0].crs)
gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=gdfs[0].crs)
import pandas as pd
gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=gdfs[0].crs)
gdf = gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=gdfs[0].crs)
gdf
dir()
hucs
import pathlib
[huc.parts for huc in hucs]
[huc.parts[-2] for huc in hucs]
[gdf['county'] = huc.parts[-2] for gdf,huc in zip(gdfs,hucs)]
for gdf,huc in zip(gdfs,hucs):
    gdf['county'] = huc.parts[-2]
gdfs
gdf = gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=gdfs[0].crs)
gdf
gdf.apply(lambda gdf: gdf['FID'])
gdf[['FID']].apply(lambda gdf: gdf['FID'])
gdf[['FID']].apply(lambda gdf: gdf.loc[0])
gdf[['FID']].apply(lambda gdf: gdf)
gdf[['FID','county']].apply(lambda gdf: gdf)
gdf[['FID','county']].apply(lambda gdf: gdf[0])
gdf[['FID','county']].apply(lambda fid,county: county)
gdf[['FID','county']].apply(lambda gdf: gdf.county)
gdf[['FID','county']].apply(lambda gdf: gdf['FID'], axis=1)
os.listdir()
import os
os.listdir()
import readlne
import readline
readline.write_history_file('directory_prep-history.sh')
exit(0
exit()
1088.
1088./16.
17408./256.
139264./2048.
exit()
import geopandas as gpd
wbd_ctx = gpd.read_file('wbd_ctx.geojson')
wbd_ctx
dir(
dir()
exit
exit()
from typing import Dict, List
from rio_tiler.io import COGReader
from rio_tiler.models import ImageData, Info, Metadata, ImageStatistics
os.getcwd()
import os
os.getcwd()
os.chdir('/scratch/04950/dhl/HAND-TauDEM/')
cog = COGReader('Elevationdd-vis.tif')
cog
info: Info = cog.info()
info
Info
assert info.nodata_type
info.nodata_type
info.band_descriptions
assert info.band_descriptions
info.band_descriptions
stats: ImageStatistics = cog.stats()
assert stats.min
meta: Metadata = cog.metadata()
meta
assert meta.statistics
assert meta.nodata_type
assert meta.band_descriptions
img: ImageData = cog.tile(tile_x, tile_y, tile_zoom, tilesize=256)
exit()
import geopandas as gpd
catchment = gpd.read_file('TX-Counties-Seven_NOAA-120901050308/Catchment.shp')
flowline = gpd.read_file('TX-Counties-Seven_NOAA-120901050308/Flowline.shp')
catchment
flowline
dir()
exit()
ls
from pathlib import Path
Path('.').rglob('Elevationdd-vis.tif')
list(Path('.').rglob('Elevationdd-vis.tif'))
elevations = list(Path('.').rglob('Elevationdd-vis.tif'))
len(elevations)
elevations
elevation_gdfs = pd.DataFrame({'elevation_fns':elevations})
import pandas as pd
elevation_gdfs = pd.DataFrame({'elevation_fns':elevations})
elevation_gdfs
elevation_gdfs['elevation_fns'].apply(lambda fn: fn[-22:-10])
elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[-22:-10])
elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[-32:-20])
elevation_gdfs['HUC12'] = elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[-32:-20])
elevation_gdfs
elevation_gdfs['version'] = elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[0:4])
elevation_gdfs['version']
elevation_gdfs[elevation_gdfs['version']=='wbd_']
elevation_gdfs[elevation_gdfs['version']=='wbd_'] = 1
elevation_gdfs[elevation_gdfs['version']=='TWDB'] = 0
elevation_gdfs['version']
elevation_gdfs.sort_values(by=['version'])
elevation_gdfs
elevation_gdfs = pd.DataFrame({'elevation_fns':elevations})
elevation_gdfs['HUC12'] = elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[-32:-20])
elevation_gdfs['version'] = elevation_gdfs['elevation_fns'].apply(lambda fn: str(fn)[0:4])
elevation_gdfs
elevation_gdfs.loc[elevation_gdfs['version']=='wbd_','version']
elevation_gdfs.loc[elevation_gdfs['version']=='wbd_','version'] = 1
elevation_gdfs.loc[elevation_gdfs['version']=='TWDB','version'] = 0
elevation_gdfs
elevation_gdfs.sort_values(by=['version'])
elevation_gdfs.sort_values(by=['version'],inplace=True)
elevation_gdfs.drop_duplicates(subset=['HUC12'],keep='last')
elevation_gdfs.drop_duplicates(subset=['HUC12'],keep='last',inplace=True)
elevation_gdfs
elevation_gdfs.sort_values(by=['HUC12'])\
from pathlib import Path
Path('./GeoInputs/GIS').rglob('Flowline.shp')
list(Path('./GeoInputs/GIS').rglob('Flowline.shp'))
len(list(Path('./GeoInputs/GIS').rglob('Flowline.shp')))
list(Path('./GeoInputs/GIS').rglob('Flowline.shp'))
flowlines = list(Path('./GeoInputs/GIS').rglob('Flowline.shp'))
pd.DataFrame({'flowline_fns':flowlines})
import pandas as pd
pd.DataFrame({'flowline_fns':flowlines})
copy_gdf = pd.DataFrame({'flowline_fns':flowlines})
copy_gdf
copy_gdf['flowline_fns'].apply(lambda fn: os.path.dirname(fn))
import os
copy_gdf['flowline_fns'].apply(lambda fn: os.path.dirname(fn))
copy_gdf['flowline_fns'].apply(lambda fn: Path(os.path.dirname(fn)).rglob('*.tif'))
copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).rglob('*.tif')))
copy_gdf['flowline_fns'].apply(lambda fn: len(list(Path(os.path.dirname(fn)).rglob('*.tif'))))
copy_gdf['flowline_fns'].apply(lambda fn: len(list(Path(os.path.dirname(fn)).rglob('*.tif')))).max()
copy_gdf['flowline_fns'].apply(lambda fn: len(list(Path(os.path.dirname(fn)).rglob('*.tif'))))
copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).rglob('*.tif')))
copy_gdf['elevation_fns'] = copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).rglob('*.tif')))
copy_gdf['elevation_fns'].apply(lambda fn: not fn)
copy_gdf['elevation_fns'].apply(lambda fn: fn)
copy_gdf['elevation_fns'].apply(lambda fn: not not fn)
copy_gdf[copy_gdf['elevation_fns'].apply(lambda fn: not not fn)]
copy_gdf = copy_gdf[copy_gdf['elevation_fns'].apply(lambda fn: not not fn)]
copy_gdf
copy_gdf['elevation_fns'].apply(lambda fn: fn[0])
copy_gdf['elevation_fns'] = copy_gdf['elevation_fns'].apply(lambda fn: fn[0])
copy_gdf
copy_gdf['elevation_fns'].apply(lambda fn: os.path.basename(fn))
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(fn)[0])
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn_)[0])
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[0])
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp'))
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')))
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')))[0]
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0],os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')))[0]
copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0],os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')))
copy_gdf['channelnetwork_fns'] = copy_gdf['elevation_fns'].apply(lambda fn: Path(os.path.join('.','GeoOutputs','GIS',os.path.splitext(os.path.basename(fn))[0],os.path.splitext(os.path.basename(fn))[0]+'_channelNetwork.shp')))
copy_gdf
copy_gdf.columns
copy_gdf['flowline_fns']
copy_gdf['flowline_fns'].apply(lambda fn: Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*')
copy_gdf['flowline_fns'].apply(lambda fn: Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*'))
copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*')))
copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*')))[0]
copy_gdf['flowline_aux_fns'] = copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*')))[0]
copy_gdf['flowline_aux_fns'] = copy_gdf['flowline_fns'].apply(lambda fn: list(Path(os.path.dirname(fn)).glob(os.path.splitext(os.path.basename(fn))[0]+'.*')))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']]
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']])
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']]))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']])))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns'])))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']])))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']]))
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']])
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1] for ext in gdf['flowline_aux_fns']]
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']])
gdf.columns
copy_gdf.columns
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']],axis=1)
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']],axis=1)[0]
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(os.path.basename(gdf['channelnetwork_fns']))[0]+os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']],axis=1)[0]
copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(os.path.basename(gdf['channelnetwork_fns']))[0]+os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']],axis=1)
copy_gdf['channelnetwork_aux_fns'] = copy_gdf[['channelnetwork_fns','flowline_aux_fns']].apply(lambda gdf: [Path(os.path.join(os.path.dirname(gdf['channelnetwork_fns']),os.path.splitext(os.path.basename(gdf['channelnetwork_fns']))[0]+os.path.splitext(ext)[1])) for ext in gdf['flowline_aux_fns']],axis=1)
copy_gdf
from shutil import copyfile
copy_gdf[['flowline_aux_fns','channelnetwork_aux_fns']]
copy_gdf[['flowline_aux_fns','channelnetwork_aux_fns']].apply(lambda gdf: [copyfile(src,dst) for src,dst in zip(gdf['flowline_aux_fns'],gdf['channelnetwork_aux_fns'])],axis=1)
import readline
os.getcwd()
readline.write_history_file('Network_Copy.py')
dir()
copy_gdf
copy_gdf.columns
copy_gdf['elevation_fns']
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[1])
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[0])
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[0]).to_csv()
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[0]).to_csv(header=False,index=False)
copy_gdf['elevation_fns'].apply(lambda fn: os.path.splitext(os.path.basename(fn))[0]).to_csv('NOAA-10m-Final_flowlines-Seven_counties2.txt',header=False,index=False)
dir()
os.getcwd()
import readline
readline.write_history_file('history0.py')
exit()
list(Path('.').glob('wbd_ctx-*/rasterDataDoesNotEnclose.err'))
from pathlib import Path
list(Path('.').glob('wbd_ctx-*/rasterDataDoesNotEnclose.err'))
len(list(Path('.').glob('wbd_ctx-*/rasterDataDoesNotEnclose.err')))
list(Path('.').glob('wbd_ctx-*/rasterDataDoesNotEnclose.err'))
encloses = list(Path('.').glob('wbd_ctx-*/rasterDataDoesNotEnclose.err'))
encloses
pd.DataFrame({'does_not_enclose':encloses})
import pandas as pd
pd.DataFrame({'does_not_enclose':encloses})
encloses = pd.DataFrame({'does_not_enclose':encloses})
encloses
encloses['does_not_enclose']
encloses['does_not_enclose'].apply(lambda fn
encloses['does_not_enclose']
encloses['does_not_enclose'].apply(lambda fn: os.path.dirname(fn))
import os
encloses['does_not_enclose'].apply(lambda fn: os.path.dirname(fn))
encloses['does_not_enclose'].apply(lambda fn: Path('.').glob(os.path.join(os.path.dirname(fn),'raster.shp')))
encloses['does_not_enclose'].apply(lambda fn: list(Path('.').glob(os.path.join(os.path.dirname(fn),'raster.shp'))))
encloses['does_not_enclose'].apply(lambda fn: list(Path('.').glob(os.path.join(os.path.dirname(fn),'hu_buff.shp'))))
encloses['hu_buff'] = encloses['does_not_enclose'].apply(lambda fn: list(Path('.').glob(os.path.join(os.path.dirname(fn),'hu_buff.shp'))))
encloses['raster'] = encloses['does_not_enclose'].apply(lambda fn: list(Path('.').glob(os.path.join(os.path.dirname(fn),'hu_buff.shp'))))
encloses['does_not_enclose'].apply(lambda fn: os.path.dirname(fn))
encloses['path'] = encloses['does_not_enclose'].apply(lambda fn: os.path.dirname(fn))
encloses['path']
os.getcwd()
import numpy as np
encloses.columns
np.savetxt(r'./wbd_ctx-does_not_enclose.txt',encloses.path)
encloses.path
encloses.path.astype(str)
encloses.path.astype(str,inplace=True)
encloses['path'].astype(str)
encloses['path'] = encloses['path'].astype(str)
np.savetxt(r'./wbd_ctx-does_not_enclose.txt',encloses.path)
np.savetxt(r'./wbd_ctx-does_not_enclose.txt',encloses['path'],fmt='%s')
os.getcwd()
dir()
encloses.columns
encloses['raster']
3800./4.
key
dir()
exit()
import geopandas as gpd
from pathlib import Path
exit()
from pathlib import Path
list(Path('.').rglob('*dd-vis.tif'))
viss = list(Path('.').rglob('*dd-vis.tif'))
viss
len(viss)
viss
dir()
viss.
viss
import geopandas as gpd
exit()
exit
exit()
import pandas as pd
pd.read_json('https://api.tnris.org/api/v1/collections/')
collections = pd.read_json('https://api.tnris.org/api/v1/collections/')
collections['next'].unique()
collections['previous'].unique()
collections['count'].unique()
import urllib.request, json
with urllib.request.urlopen('https://api.tnris.org/api/v1/collections/') as url:
    data = json.loads(url.read().decode())
    print(data)
data
len(data)
data.keys()
data['results']
data['results'].shape[0]
len(data['results'])
data['results'].keys()
data['results']
data['results'][0]
data['results'][1]
data['results'][0].keys()
pd.concat(data['results'])
pd.DataFrame(data['results'])
collections = pd.DataFrame(data['results'])
collections
collections.columns
collections['availability']
collections['availability'].unique(
collections['availability'].unique()
collections['spatial_reference'].unique()
collections['resolution'].unique()
collections['source_website'].unique()
collections.columns
collections['source_data_website'].unique()
collections.columns
collections['tile_index_url'].unique()
import readline
readline.write_history_file('TNRIS-API.py')
exit()
c
import geopandas as gpd
avail20 = gpd.read_file('TNRIS-LIDAR-Availability-20200219.zip')
avail20 = gpd.read_file('zip://TNRIS-LIDAR-Availability-20200219.zip')
avail20 = gpd.read_file('zip://TNRIS-LIDAR-Availability-20200219.zip',layer='TNRIS-LIDAR-Availability-20200219.shp')
avail20 = gpd.read_file('zip://TNRIS-LIDAR-Availability-20200219.zip',layer='TNRIS-LIDAR-Availability-20200219')
avail20 = gpd.read_file('zip://./TNRIS-LIDAR-Availability-20200219.zip')
avail20 = gpd.read_file('TNRIS-LIDAR-Availability-20200219.shp')
avail21 = gpd.read_file('TNRIS-LIDAR-Availability-20210223.shp')
avail21
avail20
avail21-avail20
avail21
avail21.index
avail21.columns
avail21['tilename'].unique()
avail21['tilename'].unique().shape[0]
avail21['demname'].unique().shape[0]
avail21['dirname'].unique().shape[0]
avail20['dirname'].unique().shape[0]
avail20.columns
avail21.columns
set(avail21['dirname'])-set(avail20['dirname'])
avail21[avail21['dirname']==set(avail21['dirname'])-set(avail20['dirname'])]
avail21[avail21['dirname'].isin(set(avail21['dirname'])-set(avail20['dirname']))]
avail21[avail21['dirname'].isin(set(avail21['dirname'])-set(avail20['dirname']))].shape[0]
avail21.shape[0]-avail21[avail21['dirname'].isin(set(avail21['dirname'])-set(avail20['dirname']))].shape[0]
avail20.shape[0]
avail21.unique()
avail21.columns
avail21['tilename']
avail21['demname']
avail21['demname']-avail20['demname']
set(avail21['demname'])-set(avail20['demname'])
len(set(avail21['demname'])-set(avail20['demname']))
avail21['demname'].lower()
avail21['demname'].str.lower()
set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower())
len(set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower()))
avail21.shape[0]-avail21[avail21['dirname'].isin(set(avail21['dirname'])-set(avail20['dirname']))].shape[0]
avail21[avail21['dirname'].isin(set(avail21['dirname'])-set(avail20['dirname']))].shape[0]
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower()
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower().isin()
set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower())
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower().isin(set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower()))
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower().isin(set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower())).min()
set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower())-avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower()
(set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower()))-set(avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower())
set(avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower()) - (set(avail21['demname'].str.lower())-set(avail20['demname'].str.lower()))
set(avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())
set(avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['demname'].str.lower())
set(avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['dorname']
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['dirname']
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['dirname'].unique()
avail21[not avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['dirname'].unique()
avail21[~avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail20['dirname'].str.lower()))]['dirname'].unique()
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())&set(avail20['dirname'].str.lower()))]['dirname'].unique()
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())&set(avail20['dirname'].str.lower()))]['dirname'].unique().shape[0]
avail21['dirname'].unique().shape[0]
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())&set(avail20['dirname'].str.lower()))]
avail2120 = avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())&set(avail20['dirname'].str.lower()))]
avail2120[avail2120['demname'].str.lower().isin(set(avail2120['demname'].str.lower())-set(avail20['demname'].str.lower()))]['dirname'].unique()
avail2120[avail2120['demname'].str.lower().isin(set(avail2120['demname'].str.lower())-set(avail20['demname'].str.lower()))]
set(avail2120['dirname'].str.lower())-set(avail21['dirname'].str.lower())
set(avail21['dirname'].str.lower())-set(avail2120['dirname'].str.lower())
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail2120['dirname'].str.lower()))]['dirname'].unique()
avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail2120['dirname'].str.lower()))]
availdiffavail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail2120['dirname'].str.lower()))]
availdiff = avail21[avail21['dirname'].str.lower().isin(set(avail21['dirname'].str.lower())-set(avail2120['dirname'].str.lower()))]
availdiff
import pandas as pd
import urllib.request, json
with urllib.request.urlopen('https://api.tnris.org/api/v1/collections/') as url:
    data = json.loads(url.read().decode())
collections = pd.DataFrame(data['results'])
collections['tile_index_url'].unique()
collections
collections.columns
collections['tile_index_url']
collections['images']
collections.columns
collections['name']
dir()
availdiff.columns
collections.columns
collections['oe_service_names']
collections['source_name']
collections.columns
collections['collection_id']
collections['tile_index_url']
collections.loc[collections.index[0],'tile_index_url']
collections.loc['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])
from pathlib import PurePath
collections['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])
collections.dropna(subset=['tile_index_url'])
collections.dropna(subset=['tile_index_url']).apply(lambda fn: PurePath(fn).parts[-1])
collections.dropna(subset=['tile_index_url']).apply(lambda fn: Path(PurePath(fn)).parts[-1])
from pathlib import Path
collections.dropna(subset=['tile_index_url']).apply(lambda fn: Path(PurePath(fn)).parts[-1])
collections.dropna(subset=['tile_index_url']).apply(lambda fn: type(fn))
collections.dropna(subset=['tile_index_url'])
150./900.
collections.dropna(subset=['tile_index_url'])
type(collections.dropna(subset=['tile_index_url']))
collections.dropna(subset=['tile_index_url']).apply(lambda fn: fn)
collections.dropna(subset=['tile_index_url']).apply(lambda fn: Path(fn))
collections.dropna(subset=['tile_index_url']).apply(lambda fn: fn)
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: fn)
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn))
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])
dir()
availdiff.columns
availdiff['dirname'].unique()
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).contains('pecos-dallas')
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('pecos-dallas')
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('pecos-dallas')]
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('pecos-dallas')]['tile_index_url']
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('pecos-dallas')]['tile_index_url'].unique()
dir()
availdiff.columns
availdiff['dirname'].unique()
availdiff['dirname'].apply(lambda fn: str.replace(re.sub(r'*cm))'
availdiff['dirname'].apply(lambda fn: re.sub('*cm-','',fn))
import re
availdiff['dirname'].apply(lambda fn: str.replace(re.sub(r'*cm))'
availdiff['dirname'].apply(lambda fn: re.sub('*cm-','',fn))
availdiff['dirname'].apply(lambda fn: re.sub('.*cm-','',fn))
availdiff['dirname'].apply(lambda fn: re.sub('.*cm-','',fn)).unique()
strs = availdiff['dirname'].apply(lambda fn: re.sub('.*cm-','',fn)).unique()
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('pecos-dallas')]['tile_index_url'].unique()
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('|'.join(strs))]['tile_index_url'].unique()
'|'.join(str)
'|'.join(strs)
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]))
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])]
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: Path(PurePath(fn)).parts[-1])]
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1])
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: re.sub('-imagery.*','',PurePath(fn).parts[-1]))
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: re.sub('(-imagery|-lidar).*','',PurePath(fn).parts[-1]))
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: re.sub('(-imagery|-lidar).*','',PurePath(fn).parts[-1])).unique()
'|'.join(strs)
strs[2][:-1]
strs[2] = strs[2][:-1]
strs[3][:-5]
strs[3][:-5]+'-'+strs[3][-5:]
strs[3] = strs[3][:-5]+'-'+strs[3][-5:]
collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: re.sub('(-imagery|-lidar).*','',PurePath(fn).parts[-1])).unique()
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: Path(PurePath(fn)).parts[-1])]
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('|'.join(strs))]['tile_index_url'].unique()
collections
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: Path(PurePath(fn)).parts[-1])]
collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('|'.join(strs))]['tile_index_url'].unique()
collections_needed = collections.dropna(subset=['tile_index_url'])[collections.dropna(subset=['tile_index_url'])['tile_index_url'].apply(lambda fn: PurePath(fn).parts[-1]).str.contains('|'.join(strs))]['tile_index_url'].unique()
collections_needed
collections[collections['tile_index_url'].isin(collections_needed)]
collections_needed = collections[collections['tile_index_url'].isin(collections_needed)]
collections_needed
type(collections_needed)
collections_needed.to_file('TNRIS-Collections_needed.json',driver='JSON')
collections_needed.to_json(('TNRIS-Collections_needed.json')
collections_needed.to_json('TNRIS-Collections_needed.json')
import readline
readline.write_history_file('TNRIS-Collections_needed.py')
exit()
from pathlib import Path
hands = Path('.').glob('*_hand_srs.tif')
hands = list(Path('.').glob('*_hand_srs.tif'))
ha
Path('.').glob('*_hand_srs.tif')
list(Path('.').glob('*_hand_srs.tif'))
os.getcwd()
import os
os.getcwd()
exit()
from pathlib import Path
Path('.').rglob('Catchments.shp')
list(Path('.').rglob('Catchments.shp'))
len(list(Path('.').rglob('Catchments.shp')))
list(Path('.').rglob('Catchments.shp'))
catchments = list(Path('.').rglob('Catchments.shp'))
pd.DataFrame({'catchment_fns':catchments})
import pandas as pd
pd.DataFrame({'catchment_fns':catchments})
catchments = pd.DataFrame({'catchment_fns':catchments})
catchments
catchments['catchment_fns'].apply(lambda fn: 
catchments['catchment_fns'].apply(lambda fn: fn.parts[:-2])
catchments['catchment_fns'].apply(lambda fn: fn.parts[:-1])
catchments['catchment_fns'].apply(lambda fn: fn.parts[-1])
catchments['catchment_fns'].apply(lambda fn: fn.parts[-2])
catchments['catchment_fns'].apply(lambda fn: fn.parts[2])
catchments['catchment_fns'].apply(lambda fn: fn.parts[1])
catchments['catchment_fns'].apply(lambda fn: fn.parts[2])
catchments['catchment_fns'].apply(lambda fn: fn.parts[0])
catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif'))
catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').exists)
catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())
catchments[catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())]['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif'))]
catchments[catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())]['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif')]
catchments[catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())]['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif'))]
catchments[catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())]['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif'))
catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())
catchments['elevation_fn_exists'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif').is_file())
catchments['elevation_fn_exists']
catchments['elevation_fns'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'Elevation.tif'))
catchments['elevation_fns']
catchments[catchments['elevation_fn_exists'],'elevation_fns']
catchments.loc[catchments['elevation_fn_exists'],'elevation_fns']
catchments.loc[!catchments['elevation_fn_exists'],'elevation_fns']
catchments.loc[~catchments['elevation_fn_exists'],'elevation_fns']
catchments.loc[~catchments['elevation_fn_exists'],'elevation_fns'] = ''
catchments['elevation_fns']
catchments.iloc[0,'elevation_fns']
catchments.loc[0,'elevation_fns']
catchments.loc[1,'elevation_fns']
catchments['elevation_fns']
catchments['stillrunning_fn_exists'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'jobNoTimeLeftWhileProcessing.err').is_file())
catchments['stillrunning_fn_exists']
catchments['stillrunning_fn_exists'].max()
catchments['notenclose_fn_exists'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'rasterDataDoesNotEnclose.err').is_file())
catchments['notenclose_fn_exists']
catchments['notenclose_fns'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'rasterDataDoesNotEnclose.err'))
catchments['stillrunning_fns'] = catchments['catchment_fns'].apply(lambda fn: Path(fn.parts[0],'jobNoTimeLeftWhileProcessing.err'))
catchments.loc[~catchments['notenclose_fn_exists'],'notenclose_fns'] = ''
catchments.loc[~catchments['stillrunning_fn_exists'],'stillrunning_fns'] = ''
catchments['stillrunning_fns']
catchments['stillrunning_fns'].max()
catchments['stillrunning_fns']
catchments['stillrunning_fns'].unique()
catchments['notenclose_fns']
catchments['catchment_fns'].apply(lambda fn: gpd.read_file(fn))
import geopandas as gpd
catchments['catchment_fns'].apply(lambda fn: gpd.read_file(fn))
catchments['catchments'] = catchments['catchment_fns'].apply(lambda fn: gpd.read_file(fn))
catchments['catchments'].apply(lambda gdf: gdf.dissolve(by=['HUC12']))
catchments['catchments_dissolved'] = catchments['catchments'].apply(lambda gdf: gdf.dissolve(by=['HUC12']))
pd.concat(catchments['catchments_dissolved'])
pd.concat(catchments['catchments_dissolved'].tolist())
pd.concat(catchments['catchments_dissolved'].tolist()).columns
pd.concat(catchments['catchments_dissolved'].tolist())
pd.concat(catchments['catchments_dissolved'].tolist()).reset_index()
pd.concat(catchments['catchments_dissolved'].tolist()).reset_index().columns
for col in pd.concat(catchments['catchments_dissolved'].tolist()).reset_index().columns:
    print(col)
for col in pd.concat(catchments['catchments_dissolved'].tolist()).reset_index().columns:
    catchments[col] = pd.concat(catchments['catchments_dissolved'].tolist()).reset_index()[col]
catchments.columns
catchments['index']
catchments.drop(['index'])
catchments.drop(columns=['index'])
catchments.drop(columns=['index'],inplace=True)
catchments.columns
catchments['Shape_Leng']
catchments['Shape_Le_1']
catchments['Shape_Le_1'].unique()
catchments.drop(columnds=['Shape_Le_1'])
catchments.drop(columns=['Shape_Le_1'])
catchments.drop(columns=['Shape_Le_1'],inplace=True)
catchments
catchments.columns
catchments[catchments['notenclose_fn_exists'],'Shape_Leng']
catchments.loc[catchments['notenclose_fn_exists'],'Shape_Leng']
catchments.loc[catchments['stillrunning_fn_exists'],'Shape_Leng']
catchments.loc[catchments['stillrunning_fn_exists'],'status'] = 'StillRunning'
catchments.loc[catchments['notenclose_fn_exists'],'status'] = 'DoesNotEnclose'
catchments.loc[catchments['elevation_fn_exists'],'status'] = 'Done'
catchments
catchments['status']
catchments['status'].unique()
catchments['status']
catchments
os.getcwd()
import os
os.getcwd()
catchments.to_file('Harvey-Counties-Progress.geojson',driver='GeoJSON')
gpd.GeoDataFrame(catchments,crs=catchments.loc[catchments.index[0],'catchments'].crs)
catchments = gpd.GeoDataFrame(catchments,crs=catchments.loc[catchments.index[0],'catchments'].crs)
catchments.to_file('Harvey-Counties-Progress.geojson',driver='GeoJSON')
catchments.columns
catchments['catchment_fns']
catchments['catchment_fns'] = catchments['catchment_fns'].apply(lambda fn: str(fn))
catchments['elevation_fns'] = catchments['elevation_fns'].apply(lambda fn: str(fn))
catchments['stillrunning_fns'] = catchments['stillrunning_fns'].apply(lambda fn: str(fn))
catchments['notenclose_fns'] = catchments['notenclose_fns'].apply(lambda fn: str(fn))
catchments.to_file('Harvey-Counties-Progress.geojson',driver='GeoJSON')
catchments.columns
catchments.drop(columns=['catchments','catchments_dissolved']
catchments.drop(columns=['catchments','catchments_dissolved'])
catchments.drop(columns=['catchments','catchments_dissolved'],drop=True)
catchments.drop(columns=['catchments','catchments_dissolved'],inplace=True)
catchments.to_file('Harvey-Counties-Progress.geojson',driver='GeoJSON')
exit()
188./286.
1000./7.
1000./7.*.75
1000./7.*.75*188./286.
exit()
import geopandas as gpd
collin = gpd.read_file('stratmap20_50cm_colllin_vanzandt.shp')
collin.columns
collin['tilename']
collin['demname']
exit()
from pathlib import Path
Path('.').glob('Catchments.shp')
list(Path('.').glob('Catchments.shp'))
list(Path('.').glob('Catchment.shp'))
list(Path('.').rglob('Catchment.shp'))
list(Path('.').rglob('Catchments.shp'))
exit()
\
from pathlib import Path
Path('.').rglob('Catchments.shp')
list(Path('.').rglob('Catchments.shp'))
len(list(Path('.').rglob('Catchments.shp')))
list(Path('.').rglob('Catchments.shp'))
pd.DataFrame({'catchment_fns':list(Path('.').rglob('Catchments.shp'))})
import pandas as pd
exit()
import geopandas as gpd
flowlines = gpd.read_file('Flowlines.shp')
flowlines
flowlines.columns
flowlines['SLOPE'].unique()
:q
exit()
1440./2.
720.*6.
exit()
29720.*.3
29720.*.7
exit()
import geopandas as gpd
import fiona
giona.listlayers('demDerived_reaches_split_filtered_addedAttributes_crosswalked.gpkg')
fiona.listlayers('demDerived_reaches_split_filtered_addedAttributes_crosswalked.gpkg')
fiona.listlayers('demDerived_reaches_split_filtered_addedAttributes_crosswalked.gpkg','r')
gpd.read_file('demDerived_reaches_split_filtered_addedAttributes_crosswalked.gpkg')
exit()
69./94.
exit()
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
MAX_B = psutil.virtual_memory().total - psutil.virtual_memory().used
oldgdal_data = os.environ['GDAL_DATA']
os.environ['GDAL_DATA'] = os.path.join(fiona.__path__[0],'gdal_data')
global args
args = type('', (), {})()
no_restart_file = False
my_file = Path(args.restart)
args.restart = '/scratch/04950/dhl/GeoFlood/GeoFlood-preprocessing/pickles/Harvey-Counties.pickle'
with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
start_time = time.time()
prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
remove_keys = []
dst_crs = rasterio.crs.CRS.from_dict(init=crs)
args.tempdir = '/scratch/04950/dhl/tmp/'
tempdir = args.tempdir
arguments = zip(
        flows_keys,
        flowshu12shape,
        catchshu12shape,
        hu12catchs,
        avail_hu12catchs_grouped,
        repeat(args),
        repeat(prefix),
        repeat(dst_crs),
        repeat(tempdir)
    )
prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
args.shapefile = '/scratch/04950/dhl/GeoFlood/GeoFlood-preprocessing/Harvey-Counties/Shapefile/Harvey-Counties.geojson'
prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
arguments = zip(
        flows_keys,
        flowshu12shape,
        catchshu12shape,
        hu12catchs,
        avail_hu12catchs_grouped,
        repeat(args),
        repeat(prefix),
        repeat(dst_crs),
        repeat(tempdir)
    )
argument = list(arguments)[977]
arguments = argument
del(argument)
dem_fps = list(avail_hu12catchs_group['stampede2name'])
src_files_to_mosaic = []
avail_hu12catchs_group = avail_hu12catchs_grouped[977]
dem_fps = list(avail_hu12catchs_group['stampede2name'])
src_res_min_to_mosaic = []
src_res_max_to_mosaic = []
src_x_to_mosaic = []
src_y_to_mosaic = []
memfile = {}
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
hu
arguments[0]
hu = arguments[0]
fp
dst_crs
memfile
subdirectory
subdirectory = os.path.join(arguments[5].directory, arguments[6]+'-'+str(arguments[0]))
arguments[5].directory = '/scratch/04950/dhl/GeoFlood/GeoFlood-preprocessing/HUC120100051004'
subdirectory = os.path.join(arguments[5].directory, arguments[6]+'-'+str(arguments[0]))
Path(subdirectory).mkdir(parents=True, exist_ok=True)
path_notime = os.path.join(subdirectory, "jobNoTimeLeftWhileProcessing.err")
Path(path_notime).touch()
path_gt1m = os.path.join(subdirectory, "allGT1m.err")
file_gt1m = Path(path_gt1m)
path_enclose = os.path.join(subdirectory, "rasterDataDoesNotEnclose.err"
)
file_enclose = Path(path_enclose)
path_elevation = os.path.join(subdirectory, 'Elevation.tif')
file_elevation = Path(path_elevation)
file_elevation.unlink(missing_ok=True)
break_hu = False
def append_check(src_files_to_mosaic,var,subdirectory,hu):
                ## Check each raster's resolution in this HUC
                #if any(np.float16(i) > 1. for i in var.res):
                #    out_path = os.path.join(subdirectory, "gt1m.err")
                #    Path(out_path).touch()
                #    print('WARNING: >1m raster input for HUC12: '+str(hu))
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
                    if src.meta!=out_meta:
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
arguments[5].log = ''
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
                        '[EXCEPTION] Exception on HUC12: ' +
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
arguments[5].memfile = ''
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
                        '[EXCEPTION] Exception on HUC12: ' +
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
src_files_to_mosaic
src_files_to_mosaic = pd.DataFrame(data={
                    'Files':src_files_to_mosaic,
                    'min(resolution)':src_res_min_to_mosaic,
                    'max(resolution)':src_res_max_to_mosaic
                })
src_files_to_mosaic.sort_values(by=['min(resolution)','max(resolution)'],inplace=True)
mosaic, out_trans = merge(list(src_files_to_mosaic['Files']),res=(max(src_x_to_mosaic),max(src_y_to_mosaic)))
for src in src_files_to_mosaic['Files']:
                    src.close()
out_meta = src.meta.copy()
out_meta.update({
                    "driver": 'GTiff',
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": dst_crs
                })
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
mosaic_tuple = (mosaic,out_meta)
break_hu
raster = arguments[4].dissolve(by=['HUC12'])
raster.reset_index(inplace=True)
raster.drop(
                        columns = {'index','index_left','index_right'},
                        inplace = True,
                        errors = 'ignore'
                    )
raster.geometry = raster.buffer(.8).buffer(-.8)
hu_buff = arguments[3].to_crs(mosaic_tuple[1]['crs'])
hu_buff.reset_index(inplace=True)
hu_buff.drop(
                        columns = {'index','index_left','index_right'},
                        inplace = True,
                        errors = 'ignore'
                    )
hu_buff_geom = list(hu_buff['geometry'])
out_path = os.path.join(subdirectory, 'hu_buff.shp')
]
my_file = Path(out_path)
my_file.is_file() and not arguments[5].overwrite and not arguments[5].overwrite_flowlines
my_file.unlink(missing_ok=True)
hu_buff.to_file(str(out_path))
out_path = os.path.join(subdirectory, 'raster.shp')
my_file = Path(out_path)
my_file.unlink(missing_ok=True)
raster.to_file(str(out_path))
len(gpd.sjoin(hu_buff,raster.to_crs(hu_buff.crs),op='within',how='inner').index) == 0
out_image = output_raster(hu_buff_geom,mosaic_tuple[0],mosaic_tuple[1],path_elevation)
hu_buff_geom
mosaic
out_meta
path_elevation
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
return_dict[arguments[0]] = out_image
return_dict = manager.dict()
manager = multiprocessing.Manager()
return_dict = manager.dict()
return_dict[arguments[0]] = out_image
Path(path_notime).unlink()
os.environ['GDAL_DATA'] = oldgdal_data
exit()
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
exit()
from bs4 import BeautifulSoup
import time
import requests
exit()
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
url = 'https://hub.catalogit.app/4837/folder/1d330100-93ce-11eb-9bd8-bb7a6aa1a9bb'
response = requests.get(url)
response
soup = BeautifulSoup(response.text, "html.parser")
soup
soup.findAll('a')
soup.findAll('body')
import phantomjs
exit()
import phantomjs
exit()
from selenium import webdriver
driver = webdriver.PhantomJS()
driver.get('https://hub.catalogit.app/4837/folder/1d330100-93ce-11eb-9bd8-bb7a6aa1a9bb')
p_element = driver.find_element_by_id(id_='intro-text')
driver
html = None
selector = '#dataTarget > div'
delay = 10
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
browser = webdriver.Chrome()
exit()
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
html = None
url = 'http://demo-tableau.bitballoon.com/'
selector = '#dataTarget > div'
delay = 10  # seconds
browser = webdriver.Chrome()
browser.get(url)
exit()
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
html = None
url = 'http://demo-tableau.bitballoon.com/'
selector = '#dataTarget > div'
delay = 10  # seconds
browser = webdriver.Chrome()
exit()
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
html = None
url = 'http://demo-tableau.bitballoon.com/'
selector = '#dataTarget > div'
delay = 10  # seconds
browser = webdriver.PhantomJS()
browser.get(url)
try:
    # wait for button to be enabled
    WebDriverWait(browser, delay).until(
        EC.element_to_be_clickable((By.ID, 'getData'))
    )
    button = browser.find_element_by_id('getData')
    button.click()
    # wait for data to be loaded
    WebDriverWait(browser, delay).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )
except TimeoutException:
    print('Loading took too much time!')
else:
    html = browser.page_source
finally:
    browser.quit()
try:
    # wait for button to be enabled
    WebDriverWait(browser, delay).until(
        EC.element_to_be_clickable((By.ID, 'getData'))
    )
    button = browser.find_element_by_id('getData')
    button.click()
    # wait for data to be loaded
    WebDriverWait(browser, delay).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )
except TimeoutException:
    print('Loading took too much time!')
else:
    html = browser.page_source
finally:
    browser.quit()
url
url = 'https://hub.catalogit.app/4837/folder/1d330100-93ce-11eb-9bd8-bb7a6aa1a9bb'
browser.get(url)
url = 'https://google.com/ncr'
browser.get(url)
url = 'http://demo-tableau.bitballoon.com/'
browser.get(url)
driver.quit()
browser.quit()
browser.get(url)
browser = webdriver.Chrome()
browser = webdriver.PhantomJS()
url = 'https://google.com/ncr'
browser.get(url)
try:
    # wait for button to be enabled
    WebDriverWait(browser, delay).until(
        EC.element_to_be_clickable((By.ID, 'getData'))
    )
    button = browser.find_element_by_id('getData')
    button.click()
    # wait for data to be loaded
    WebDriverWait(browser, delay).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )
except TimeoutException:
    print('Loading took too much time!')
else:
    html = browser.page_source
finally:
    browser.quit()
try:
    # wait for button to be enabled
    WebDriverWait(browser, delay).until(
        EC.element_to_be_clickable((By.ID, 'getData'))
    )
    button = browser.find_element_by_id('getData')
    button.click()
    # wait for data to be loaded
    WebDriverWait(browser, delay).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )
except TimeoutException:
    print('Loading took too much time!')
else:
    html = browser.page_source
finally:
    browser.quit()
browser.quit()
browser = webdriver.Firefox()
exit()
from selenium import webdriver
browser = webdriver.Firefox()
browser.quit()
browser = webdriver.Firefox()
browser = webdriver.Chrome()
exit()
from selenium import webdriver
browser = webdriver.Chrome()
exit()
from selenium impport webdriver
from selenium import webdriver
profile = webdriver.FirefoxProfile()
browser = webdriver.Firefox(profile)
driver = webdriver.PhantomJS()
driver.get('https://google.com/')
sbtn = driver.find_element_by_css_selector('button.gbqfba')
driver.find_element_by_css_selector('')
driver.find_element_by_css_selector()
driver.page_source
driver.page_source()
driver.quit()
driver = webdriver.PhantomJS()
driver.get('https://google.com/')
driver.page_source()
driver.page_source
driver.quit()
driver = webdriver.PhantomJS()
driver.get('https://hub.catalogit.app/4837/folder/1d330100-93ce-11eb-9bd8-bb7a6aa1a9bb/entry/62a2f610-663c-11eb-91f8-a92a6e6cc498')
page_source = driver.page_source
page_source
page_sourceself.driver.set_window_size(1120, 550)
self.driver.set_window_size(1120, 550)
driver.set_window_size(1120, 550)
exit()
from selenium import webdriver
browser = webdriver.Firefox()
exit()
from selenium import webdriver
from selenium
exit()
rasterio.crs.CRS.from_dict(init='EPSG:32614')
import rasterio
rasterio.crs.CRS.from_dict(init='EPSG:32614')
dst_crs = rasterio.crs.CRS.from_dict(init='EPSG:32614')
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
src = rasterio.open(fp)
dem_fps
dir()
exit
exit()
rom __future__ import division
import os
import sys
import shutil
import subprocess
from time import perf_counter
os.chdir("/work2/02044/arcturdk/stampede2/GeoFlood/GeoFlood-master2/GeoNet")
from pygeonet_rasterio import *
import rasterio
grass7bin = r'grass78'
startcmd = [grass7bin, '--config', 'path']
p = subprocess.Popen(startcmd, shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
gisbase = out.decode("utf-8").strip('\n')
os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
os.environ['GISBASE'] = gisbase
home = os.path.expanduser("~")
os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')
gpydir = os.path.join(gisbase, "etc", "python")
sys.path.append(gpydir)
gisdb = os.path.join(home, "grassdata")
os.environ['GISDBASE'] = gisdb
path = os.getenv('LD_LIBRARY_PATH')
directory = os.path.join(gisbase, 'lib')
os.chdir("/work2/02044/arcturdk/stampede2/GeoFlood/GeoFlood-master2/GeoNet")
from pygeonet_rasterio import *
exit
exit()
from __future__ import division
import os
import sys
import shutil
import subprocess
from time import perf_counter
os.chdir("/work2/02044/arcturdk/stampede2/GeoFlood/GeoFlood-master2/GeoNet")
from pygeonet_rasterio import *
exit()
from __future__ import division
import os
import sys
import shutil
import subprocess
from time import perf_counter
os.chdir("/work2/02044/arcturdk/stampede2/GeoFlood/GeoFlood-master2/GeoNet")
from pygeonet_rasterio import *
os.chdir("/scratch/04950/dhl/GeoFlood/GeoFlood-DHL.git/GeoNet")
from pygeonet_rasterio import *
exit()
import argparse
import pandas as pd
import fiona
import geopandas as gpd
import utm
exit()
import argparse
import pandas as pd
import fiona
import geopandas as gpd
import utm
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
import pickle
import gc
import tempfile
import shutil
import uuid
import traceback
args = type('', (), {})()
with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
args.restart = '/scratch/04950/dhl/GeoFlood/Bridges/DEM2basin-Bridges.pickle'
with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
start_time = time.time()
prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
args.shapefile = '/scratch/04950/dhl/GeoFlood/Bridges/Arctur/dem_tiles.shp'
prefix = os.path.splitext(os.path.basename(args.shapefile))[0]
remove_keys = []
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
args.directory = '/scratch/04950/dhl/GeoFlood/Bridges/Output'
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
flowshu12shape = [
        flowshu12shape[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
catchshu12shape = [
        catchshu12shape[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
hu12catchs = [
        hu12catchs[key]
        for key
        in range(len(flows_keys))
        if key
        not in remove_keys_idcs
    ]
avail_hu12catchs_grouped = [
        avail_hu12catchs_grouped[key]
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
dst_crs = rasterio.crs.CRS.from_dict(init=crs)
multiprocessing.set_start_method('spawn')
N_CORES = multiprocessing.cpu_count() - 1
args.tempdir = '/scratch/04950/dhl/tmp'
tempdir = args.tempdir
manager = multiprocessing.Manager()
return_dict = manager.dict()
args.percent_free_mem = 25
args.percent_free_disk = 10
args.overwrite = False
args.overwrite_flowlines = False
args.overwrite_roughnesses = False
args.overwrite_catchments = False
args.log = ''
args.memfile = False
tempdir
dst_crs
prefix
args
avail_hu12catchs_grouped
arguments = zip(
        flows_keys,
        flowshu12shape,
        catchshu12shape,
        hu12catchs,
        avail_hu12catchs_grouped,
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
N_CORES
n_cores = N_CORES
MAX_B
MAX_B = psutil.virtual_memory().total - psutil.virtual_memory().used
oldgdal_data = os.environ['GDAL_DATA']
os.environ['GDAL_DATA'] = os.path.join(fiona.__path__[0],'gdal_data')
max_b = MAX_B
percent_free_mem = args.percent_free_mem
percent_free_disk = args.percent_free_disk
def get_mosaic(avail_hu12catchs_group,hu,break_hu,dst_crs):
            ## Get mosaic of DEMs for each HUC
                #if any(np.float16(i) > 1. for i in var.res):
                #    out_path = os.path.join(subdirectory, "gt1m.err")
                #    Path(out_path).touch()
                #    print('WARNING: >1m raster input for HUC12: '+str(hu))
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
                    ## Don't do an expensive reprojection if projection already correct
                    if src.meta!=out_meta:
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
dem_fps = list(avail_hu12catchs_group['stampede2name'])
avail_hu12catchs_grouped[0]
avail_hu12catchs_group = avail_hu12catchs_grouped[0]
dem_fps = list(avail_hu12catchs_group['stampede2name'])
src_files_to_mosaic = []
src_res_min_to_mosaic = []
src_res_max_to_mosaic = []
src_x_to_mosaic = []
src_y_to_mosaic = []
memfile = {}
if arguments[5].memfile:
                for fp in dem_fps:
                    memfile[fp] = MemoryFile()
arguments[5]
arguments
list(arguments)[0]
len(list(arguments)[0])
list(arguments)[0]
arguments
list(arguments)[0]
list(arguments)[1]
list(arguments)
arguments = zip(
        flows_keys,
        flowshu12shape,
        catchshu12shape,
        hu12catchs,
        avail_hu12catchs_grouped,
        repeat(args),
        repeat(prefix),
        repeat(dst_crs),
        repeat(tempdir)
    )
arguments = list(arguments)[0]
arguments
len(arguments)
if arguments[5].memfile:
                for fp in dem_fps:
                    memfile[fp] = MemoryFile()
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
                        '[EXCEPTION] Exception on HUC12: ' +
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
src = rasterio.open(fp)
transform, width, height = calculate_default_transform(
  1                         src.crs,
  2                         dst_crs,
  3                         src.width,
  4                         src.height,
  5                         *src.bounds
  6                     )
transform, width, height = calculate_default_transform(
                        src.crs,
                        dst_crs,
                        src.width,
                        src.height,
                        *src.bounds
                    )
out_meta = src.meta.copy()
out_meta.update({                        'crs': dst_crs,                        'transform': transform,                        'width': width,                        'height': height                    })
src.meta!=out_meta
arguments[5].memfile
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
def append_check(src_files_to_mosaic,var,subdirectory,hu):
                ## Check each raster's resolution in this HUC
                #if any(np.float16(i) > 1. for i in var.res):
                #    out_path = os.path.join(subdirectory, "gt1m.err")
                #    Path(out_path).touch()
                #    print('WARNING: >1m raster input for HUC12: '+str(hu))
                #    sys.stdout.flush()
                #else:
                src_res_min_to_mosaic.append(min(var.res))
                src_res_max_to_mosaic.append(min(var.res))
                src_x_to_mosaic.append(var.res[0])
                src_y_to_mosaic.append(var.res[1])
                src_files_to_mosaic.append(var)
                return(src_files_to_mosaic,src_res_min_to_mosaic,src_res_max_to_mosaic,src_x_to_mosaic,src_y_to_mosaic)
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
hu = flows_keys[0]
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
src_files_to_mosaic
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
                        '[EXCEPTION] Exception on HUC12: ' +
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
fp
dem_fps
dem_fps.index(fp)
len(dem_fps)
dem_fps[dem_fps.index(fp)]
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
                    )                except Exception as err:
                    print(
                        '[EXCEPTION] Exception on HUC12: ' +
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
                        '[EXCEPTION] Exception on HUC12: ' +
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
src_files_to_mosaic
src_files_to_mosaic = pd.DataFrame(data={
                    'Files':src_files_to_mosaic,
                    'min(resolution)':src_res_min_to_mosaic,
                    'max(resolution)':src_res_max_to_mosaic
                })
src_files_to_mosaic.sort_values(by=['min(resolution)','max(resolution)'],inplace=True)
mosaic, out_trans = merge(list(src_files_to_mosaic['Files']),res=(max(src_x_to_mosaic),max(src_y_to_mosaic)))
for src in src_files_to_mosaic['Files']:
                    src.close()
\
out_meta = src.meta.copy()
out_meta.update({
                    "driver": 'GTiff',
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": dst_crs
                })
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
keyvalue
memfile.items()
keyvalue
memfile.items().index(keyvalue)
list(memfile.items()).index(keyvalue)
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
for keyvalue in memfile.items():
                    keyvalue[1].close()
                    Path(keyvalue[1].name).unlink(missing_ok=True)
for keyvalue in memfile.items():
    try:
        keyvalue[1].close()
        Path(keyvalue[1].name).unlink(missing_ok=True)
    except:
        pass
mosaic_tuple = (mosaic,out_meta)
break_hu
raster = arguments[4].dissolve(by=['HUC12'])
raster.reset_index(inplace=True)
raster.drop(
                        columns = {'index','index_left','index_right'},
                        inplace = True,
                        errors = 'ignore'
                    )
raster.geometry = raster.buffer(.8).buffer(-.8)
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
my_file.unlink(missing_ok=True)
hu_buff.to_file(str(out_path))
out_path = os.path.join(subdirectory, 'raster.shp')
my_file = Path(out_path)
my_file.unlink(missing_ok=True)
raster.to_file(str(out_path))
len(gpd.sjoin(hu_buff,raster.to_crs(hu_buff.crs),op='within',how='inner').index) == 0
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
path_elevation = os.path.join(subdirectory, 'Elevation.tif')
file_elevation = Path(path_elevation)
file_elevation.unlink(missing_ok=True)
break_hu = False
with rasterio.open(path_elevation,"w",**out_meta) as dst:
                dst.write(out_image)
import readline
readline.write_history_file('DEM2basin-Bridges-history.py')
exit()
with open('DEM2basin-1m.pickle', 'rb') as input:
    flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
import pickle
with open('DEM2basin-1m.pickle', 'rb') as input:
    flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
flows_keys
len(flows_keys)
np.unique(np.array(flows_keys))
import numpy as np
np.unique(np.array(flows_keys))
np.unique(np.array(flows_keys)).shape[0]
flowshu12shape
type(avail_hu12catchs_grouped)
len(avail_hu12catchs_grouped)
import pandas as pd
pd.concat(avail_hu12catchs_grouped)
pd.concat(avail_hu12catchs_grouped)['geometry']
gpd.GeoDataFrame(pd.concat(avail_hu12catchs_grouped))
import geopandas as gdp
import geopandas as gpd
del(gdp)
gpd.GeoDataFrame(pd.concat(avail_hu12catchs_grouped))
gpd.GeoDataFrame(pd.concat(avail_hu12catchs_grouped)).to_file('avail_hu12catchs_grouped.geojson',driver='GeoJSON')
dir()
gpd.GeoDataFrame(pd.concat(hu12catchs)).to_file('hu12catchs.geojson',driver='GeoJSON')
os.getcwd()
import os
os.getcwd()
os.listdir()
flows_keys
'111403070102' in flows_keys
flows_keys.index'111403070102')
flows_keys.index('111403070102')
flows_keys[flows_keys.index('111403070102')]
flowshu12shape[flows_keys.index('111403070102')]
catchshu12shape[flows_keys.index('111403070102')]
flowshu12shape[flows_keys.index('111403070102')].columns
avail_hu12catchs_grouped[flows_keys.index('111403070102')].columns
avail_hu12catchs_grouped[flows_keys.index('111403070102')]['stampede2name']
avail_hu12catchs_grouped[flows_keys.index('111403070102')]['stampede2name'].unique()
os.getcwd()
import dem2basin
dem2basin
args = type('', (), {})()
import os
os.getenv('SCRATCH')
path_dem2basin = os.path.join(os.getenv('SCRATCH'),'GeoFlood','DEM2basin')
path_tnris = os.path.join('scratch','projects','tnris')
path_1m = os.path.join(path_tnris,'dhl-flood-modelling','GeoFlood','DEM2basin-1m')
args.nhd = os.path.join(path_dem2basin,'NFIEGeo_TX.gdb')
args.huc12 = os.path.join(path_dem2basin,'WBD-HU12-TX.shp')
args.shapefile = args.huc12
args.raster = os.path.join(path_tnris,'tnris-lidardata')
args.buffer = 500.
args.availability = os.path.join(os.getenv('WORK'),'GeoFlood','preprocessing','data','TNRIS-LIDAR-Availability-20210525.shp')
args.directory = os.path.join(path_1m,'Output')
args.restart =  os.path.join(path_1m,'DEM2basin-1m.pickle')
args.tempdir = os.path.join(path_1m,'tmp')
args.percent_free_mem = 25
args.percent_free_disk = 10
import argparse
import pandas as pd
import fiona
import geopandas as gpd
import utm
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
import pickle
import gc
import tempfile
import shutil
import uuid
import traceback
flowshu12shape,catchshu12shape = dem2basin.flows_catchs()
args.
args
global args
flowshu12shape,catchshu12shape = dem2basin.flows_catchs()
import readline
readline.write_history_file('dem2basin-history.py')
global args
args = type('', (), {})()
path_dem2basin = os.path.join(os.getenv('SCRATCH'),'GeoFlood','DEM2basin')
path_tnris = os.path.join('scratch','projects','tnris')
path_1m = os.path.join(path_tnris,'dhl-flood-modelling','GeoFlood','DEM2basin-1m')
args.nhd = os.path.join(path_dem2basin,'NFIEGeo_TX.gdb')
args.huc12 = os.path.join(path_dem2basin,'WBD-HU12-TX.shp')
args.shapefile = args.huc12
args.raster = os.path.join(path_tnris,'tnris-lidardata')
args.buffer = 500.
args.availability = os.path.join(os.getenv('WORK'),'GeoFlood','preprocessing','data','TNRIS-LIDAR-Availability-20210525.shp')
args.directory = os.path.join(path_1m,'Output')
args.restart =  os.path.join(path_1m,'DEM2basin-1m.pickle')
args.tempdir = os.path.join(path_1m,'tmp')
args.percent_free_mem = 25
args.percent_free_disk = 10
a]
lowshu12shape,catchshu12shape = dem2basin.flows_catchs()
flowshu12shape,catchshu12shape = dem2basin.flows_catchs()
import dem2basin
flowshu12shape,catchshu12shape = dem2basin.flows_catchs()
    shape = gpd.read_file(args.shapefile)
   shape = gpd.read_file(args.shapefile)
shape = gpd.read_file(args.shapefile)
shape.drop(
    columns = ['Shape_Leng','Shape_Area'],
    inplace = True,
    errors = 'ignore'
)
shape.rename(
    columns = {'HUC12':'HUC12_shapefile'},
    inplace = True,
    errors = 'ignore'
)
hu12 = gpd.read_file(args.huc12,mask=shape)
hu12 = hu12[['HUC12','geometry']]
flows = gpd.read_file(args.nhd,layer='Flowline',mask=hu12)
flows.drop(
    columns = [
        'Shape_Length',
        'Shape_Area',
        'AreaSqKM',
        'index_left',
        'index_right'
    ],
    inplace = True,
    errors = 'ignore'
)
flows.reset_index(inplace=True)
flows.set_index('COMID',inplace=True)
flows.sort_index(inplace=True)
flows_rep = flows.copy()
flows_rep['geometry'] = flows.representative_point()
flowshu12shape_rep = gpd.sjoin(
    flows_rep,
    hu12.to_crs(flows_rep.crs),
    op = 'intersects',
    how = 'inner'
)
flowshu12shape_rep.drop(columns=['index_right'],inplace=True)
catchs = gpd.read_file(args.nhd,layer='Catchment',mask=hu12)
catchs.drop(
    columns=[
        'Shape_Length',
        'Shape_Area',
        'AreaSqKM',
        'index_left',
        'index_right'
    ],
    inplace = True,
    errors = 'ignore'
)
catchs.reset_index(inplace=True)
catchs.set_index('FEATUREID',inplace=True)
catchs.sort_index(inplace=True)
catchshu12shape = catchs[catchs.index.isin(flowshu12shape_rep.index)]
flowshu12shape = flows[flows.index.isin(flowshu12shape_rep.index)]
flowshu12shape = flowshu12shape[
    flowshu12shape.index.isin(catchshu12shape.index)
]]
flowshu12shape = flows[flows.index.isin(flowshu12shape_rep.index)]
flowshu12shape = flowshu12shape[
    flowshu12shape.index.isin(catchshu12shape.index)
]
flowshu12shape['HUC12'] = flowshu12shape_rep.loc[
    flowshu12shape.index,
    'HUC12'
]
catchshu12shape['HUC12'] = flowshu12shape_rep.loc[
    catchshu12shape.index,
    'HUC12'
]
flowshu12shape.loc[flowshu12shape['StreamOrde']==0,'Roughness'] = .99
flowshu12shape.loc[flowshu12shape['StreamOrde']==1,'Roughness'] = .2
flowshu12shape.loc[flowshu12shape['StreamOrde']==2,'Roughness'] = .1
flowshu12shape.loc[flowshu12shape['StreamOrde']==3,'Roughness'] = .065
flowshu12shape.loc[flowshu12shape['StreamOrde']==4,'Roughness'] = .045
flowshu12shape.loc[flowshu12shape['StreamOrde']==5,'Roughness'] = .03
flowshu12shape.loc[flowshu12shape['StreamOrde']==6,'Roughness'] = .01
flowshu12shape.loc[flowshu12shape['StreamOrde']==7,'Roughness'] = .025
catchshu12shape.geometry = catchshu12shape.buffer(0)
flowshu12shape
flowshu12shape.to_file(
flowshu12shape.to_file('flowshu12shape.geojson',driver='GeoJSON')
catchshu12shape.to_file('catchshu12shape.geojson',driver='GeoJSON')
os.getcwd()
shape
hu12catchs = catchshu12shape.dissolve(by='HUC12')
hu12catchs.reset_index(inplace=True)
shape = hu12catchs.copy()
shape.to_crs('epsg:4326',inplace=True)
uniq = shape.representative_point().apply(lambda p: utm.latlon_to_zone_number(p.y,p.x)).value_counts().idxmax()
uniq
hu12catchs.crs.datum.name
crs = 'epsg:6343'
hu12catchs['geometry'] = hu12catchs.to_crs(crs).buffer(args.buffer)
hu12catchs.crs = crs
flowshu12shape.to_crs(crs,inplace=True)
catchshu12shape.to_crs(crs,inplace=True)
hu12catchs
hu12catchs.geometry
type(hu12catchs)
hu12catchs.to_file('hu12catchs_postbuffer.geojson',driver='GeoJSON')
avail = gpd.read_file(args.availability)
avail_hu12catchs = gpd.sjoin(avail,hu12catchs.to_crs(avail.crs),op='intersects',how='inner')
avail_hu12catchs
fnexts = ['.dem','.img']
for fnext in fnexts:
        avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
avail_hu12catchs
avail_hu12catchs.columns
avail_hu12catchs['demname']
avail_hu12catchs['demname'].isna()
avail_hu12catchs['demname'].isna().sum()
avail_hu12catchs['dirname'].isna().sum()
for dirname in avail_hu12catchs['dirname'].unique():
        stampede2names = []
        #raster = '/scratch/projects/tnris/tnris-lidardata'
        basename = os.path.join(args.raster,dirname,'dem')+os.sep
        for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
        direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
        ## If more than one vector image extension found in a DEM project,
        ##  then figure out each file's extension individually
        ## TODO: Test this against stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro
        if len(direxts) > 1:
            for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,x+fnext))
                    if glob.glob(stampede2name.iloc[0]):
                        break
                avail_hu12catchs.loc[truth,'stampede2name'] = stampede2name
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
avail_hu12catchs
avail_hu12catchs.columns
stampede2name
avail_hu12catchs['demname']
avail_hu12catchs['dirname']
avail_hu12catchs['dirname'].unique().shape[0]
dirname
stampede2names
basename
os.path.join(os.sep,args.raster,dirname,'dem')+os.sep
args.raster
args.raster = '/scratch/projects/tnris/tnris-lidardata'
for dirname in avail_hu12catchs['dirname'].unique():
        stampede2names = []
        #raster = '/scratch/projects/tnris/tnris-lidardata'
        basename = os.path.join(args.raster,dirname,'dem')+os.sep
        for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
        direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
        ## If more than one vector image extension found in a DEM project,
        ##  then figure out each file's extension individually
        ## TODO: Test this against stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro
        if len(direxts) > 1:
            for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,x+fnext))
                    if glob.glob(stampede2name.iloc[0]):
                        break
                avail_hu12catchs.loc[truth,'stampede2name'] = stampede2name
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
avail_hu12catchs.columns
avail_hu12catchs['stampede2name']
avail_hu12catchs['stampede2name'].isna().sum()
avail_hu12catchs.groupby(by=['dirname']).apply(lambda gdf: gdf['stampede2name'].unique().shape[0])
avail_hu12catchs.groupby(by=['dirname']).apply(lambda gdf: gdf['stampede2name'].unique())
avail_hu12catchs.groupby(by=['dirname']).apply(lambda gdf: gdf['stampede2name'].unique().shape[0])
dirname_lens = avail_hu12catchs.groupby(by=['dirname']).apply(lambda gdf: gdf['stampede2name'].unique().shape[0])
type(dirname_lens)
pd.DataFrame(dirname_lens)
pd.DataFrame(dirname_lens,columns={0:'dirname_len'})
pd.DataFrame(dirname_lens).rename(columns={0:'dirname_len'})
pd.DataFrame(dirname_lens).rename(columns={0:'dirname_len'}).reset_index()
dirname_lens = pd.DataFrame(dirname_lens).rename(columns={0:'dirname_len'}).reset_index()
avail_hu12catchs.merge(dirname_lens,on=['dirname'])
avail_hu12catchs
avail_hu12catchs['stampede2name'].isna()
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna()]
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna()]['dirname']
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname']
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique().shape[0]
avail_hu12catchs['dirname'].unique().shape[0]
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs[avail_hu12catchs['dirname']=='usgs-2019-70cm-hurricane']
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='usgs-2019-70cm-hurricane','stampede2name']
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='usgs-2019-70cm-hurricane','stampede2name'].unique()
stampede2name
stampede2names
dirname
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name']
dirname = 'usgs-2019-70cm-hurricane'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
basename
Path(basename).is_dir()
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
Path(basename).listdir()
Path(basename).glob('*')
list(Path(basename).glob('*'))
Path('/scratch/projects/tnris/tnris-lidardata').rglob(os.path.*/dem/*
PurePath
from pathlib import PurePath
PurePath('*','dem','*')
PurePath('*','dem','*').str
PurePath('*','dem','*').string
purepath = PurePath('*','dem','*')
purepath.name
purepath.parents
purepath.parents()
purepath.parts
purepath.root
purepath.stem
str(purepath)
Path('/scratch/projects/tnris/tnris-lidardata').rglob(str(PurePath('*','dem','*'))
Path('/scratch/projects/tnris/tnris-lidardata').rglob(str(PurePath('*','dem','*')))
list(Path('/scratch/projects/tnris/tnris-lidardata').rglob(str(PurePath('*','dem','*'))))
tnris_lidardata = list(Path('/scratch/projects/tnris/tnris-lidardata').rglob(str(PurePath('*','dem','*'))))
len(tnris_lidardata)
tnris_lidardata[0]
tnris_lidardata[0].suffix
suffix
np.unique(np.array([suffix.suffix for suffix in tnris_lidardata]))
fnexts = ['.dem','.img','.tif']
for fnext in fnexts:
        avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
for dirname in avail_hu12catchs['dirname'].unique():
        stampede2names = []
        #raster = '/scratch/projects/tnris/tnris-lidardata'
        basename = os.path.join(args.raster,dirname,'dem')+os.sep
        for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
        direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
        ## If more than one vector image extension found in a DEM project,
        ##  then figure out each file's extension individually
        ## TODO: Test this against stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro
        if len(direxts) > 1:
            for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,x+fnext))
                    if glob.glob(stampede2name.iloc[0]):
                        break
                avail_hu12catchs.loc[truth,'stampede2name'] = stampede2name
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
avail_hu12catchs
avail_hu12catchs['stampede2name'].isna().sum()
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna(),'dirname']
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna()]
avail_hu12catchs.columns
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname']
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2020-50cm-north-central-texas','stampede2name'].unique()
name
for name in avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique():
    avail_hu12catchs.loc[avail_hu12catchs['dirname']==name,'stampede2name'].unique()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
dirname='stratmap-2020-50cm-north-central-texas'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
basename
dirname = 'hgac-2008-1m'
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
direxts
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,x+list(direxts)[0]))
stampede2name
basenam
stampede2name.unique()
stampede2name.drop_duplicates(inplace=True)
stampede2name
stampede2name.shape[0]
p = Path(basename)
p
for subp in p.rglob('*'):
        if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampedename
stampede2name
stampede2name.shape[0[
stampede2name.shape[0]
stampede2name.unique()
stampede2name = stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])]
stampede2name
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,x+list(direxts)[0]))
stampede2name.drop_duplicates(inplace=True)
p = Path(basename)
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name
stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])
[subp.as_posix() for subp in list(p.rglob('*'))]
stampede2name
stampede2name.unique()
[subp.as_posix() for subp in list(p.rglob('*'))]
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] 
avail_hu12catchs['demname']
stampede2name.unique()
avail_hu12catchs['demname']
[subp.as_posix() for subp in list(p.rglob('*'))]
stampede2name.unique()
avail_hu12catchs['demname']
demname
name
dem
avail_hu12catchs['demname'].apply(lambda dem: dem.split(sep='-')[0])
avail_hu12catchs['demname'].apply(lambda dem: dem.split(sep='-')[0]).unique()
avail_hu12catchs[avail_hu12catchs['demname']=='No Data Exist']
avail_hu12catchs[avail_hu12catchs['demname']=='No Data Exist']['dirname'].unique()
avail_hu12catchs[avail_hu12catchs['dirname']=='fema-2011-1m-liberty']['demname'].unique()
avail_hu12catchs[avail_hu12catchs['demname']=='No Data Exist']
avail_hu12catchs[avail_hu12catchs['demname']!='No Data Exist']
avail_hu12catchs = avail_hu12catchs[avail_hu12catchs['demname']!='No Data Exist']
avail_hu12catchs['dirname']
avail_hu12catchs['dirname']
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs['demname'].apply(lambda dem: dem.split(sep='_')[-1]).unique()
avail_hu12catchs['demname'].apply(lambda dem: dem.split(sep='_')[-1]).unique().shape[0]
avail_hu12catchs['demname'].apply(lambda dem: len(dem.split(sep='_')[-1])).unique().shape[0]
avail_hu12catchs['demname'].apply(lambda dem: len(dem.split(sep='_')[-1])).unique()
avail_hu12catchs['demname'].apply(lambda dem: len(dem.split(sep='_')[-1]))
avail_hu12catchs['demname_split_len'] = avail_hu12catchs['demname'].apply(lambda dem: len(dem.split(sep='_')[-1]))
avail_hu12catchs.loc[avail_hu12catchs['demname_split_len']==12,'dirname']
avail_hu12catchs.loc[avail_hu12catchs['demname_split_len']==12,'demname']
import re
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)[-2]))
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)))
avail_hu12catchs['demname'].apply(lambda dem: re.split('-|_',dem))
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)))
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem))).unique()
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)[1]))
avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)[1])).unique()
avail_hu12catchs['demname'] = avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)[1]))
avail_hu12catchs['demname_split_len'] = avail_hu12catchs['demname'].apply(lambda dem: len(re.split('-|_',dem)[1]))
vail_hu12catchs['demname_split_len']
avail_hu12catchs['demname_split_len']
avail_hu12catchs.loc[avail_hu12catchs['demname_split_len']==5,'demname']
avail_hu12catchs.loc[avail_hu12catchs['demname_split_len']==4,'demname']
subp
stampede2name
basename
p
subp
subp.prefix
subp.prefixes
subp.parents
subp.parent
subp.parts
Path(subp.parts)
Path(*subp.parts)
Path(*subp.parts[:-1])
Path(*subp.parts[:-1],re.split(subp.parts[-1])
re.split('-|_',subp.parts[-1])[1])
re.split('-|_',subp.parts[-1])[1]
[re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]
np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))
resolution
avail_hu12catchs
avail_hu12catchs['stampede2name'].apply(lambda fn: fn)
avail_hu12catchs['stampede2name'].apply(lambda fn: type(fn))
avail_hu12catchs['stampede2name'].apply(lambda fn: os.splitext(fn)[-1])
avail_hu12catchs['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1] if type(fn)==str)
avail_hu12catchs['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1] if isinstance(fn,str))
avail_hu12catchs['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1] if isinstance(fn,str))
avail_hu12catchs['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs['stampede2name'].isna(.apply(lambda fn:fn)
avail_hu12catchs[~avail_hu12catchs['stampede2name'].isna()]
avail_hu12catchs[~avail_hu12catchs['stampede2name'].isna()].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs[~avail_hu12catchs['stampede2name'].isna()].apply(lambda fn: fn)
avail_hu12catchs.loc[~avail_hu12catchs['stampede2name'].isna(),'stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs.loc[~avail_hu12catchs['stampede2name'].isna()]
avail_hu12catchs_notna = avail_hu12catchs.loc[~avail_hu12catchs['stampede2name'].isna()]
avail_hu12catchs_notna['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs_notna['stampede2name_ext'] = avail_hu12catchs_notna['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1])
avail_hu12catchs_notna['stampede2name_ext']
avail_hu12catchs_notna
avail_hu12catchs_notna.groupby(['dirname'])
avail_hu12catchs_notna.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name_ext].unique())
avail_hu12catchs_notna.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name_ext'].unique())
avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro'
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name']]
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name']
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].unique()
for fn in avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].unique():
    fn
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name']
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'][-3]
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-3]
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-3:]
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-4:]
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-4:]=='.tif'
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'][avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-4:]=='.tif']
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'][avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-4:]=='.tif'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'][avail_hu12catchs.loc[avail_hu12catchs['dirname']=='stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro','stampede2name'].str[-4:]=='.dem'].unique()
avail_hu12catchs[avail_hu12catchs['stampede2name'].isna(),'stampede2name'].unique()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'stampede2name'].unique()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
stampede2names = []
dirname = 'fema-2006-140cm-coastal'
basename = os.path.join(args.raster,dirname,'dem')+os.sep
basename
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
direxts
p = Path(basename)
resolution = np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))[0]
resolution
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+list(direxts)[0]))
stampede2name
stampede2name.unique9)
stampede2name.unique()
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name
stampede2name.unique()
stampede2name = stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])]
stampede2name
stampede2name.unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] 
stampede2name.drop_duplicates()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] = stampede2name
stampede2name
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name']
p.rglob('*')
list(p.rglob('*'))
len(list(p.rglob('*')))
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+list(direxts)[0]))
stampede2name
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name
stampede2name = stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])]
stampede2name
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+list(direxts)[0]))
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])]
stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])].index
stampede2name_index = stampede2name[stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])].index
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] = stampede2name
avail_hu12catchs = avail_hu12catchs[avail_hu12catchs.index]
avail_hu12catchs[avail_hu12catchs.index]
avail_hu12catchs.loc[avail_hu12catchs.index]
avail_hu12catchs.iloc[avail_hu12catchs.index]
avail_hu12catchs
avail_hu12catchs.iloc[stampede2name_index]
stampede2name_drop = stampede2name[~stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])].index
stampede2name_drop
avail_hu12catchs.drop(index=stampede2name_drop,inplace=True,errors='ignore'])
avail_hu12catchs.drop(index=stampede2name_drop,inplace=True,errors='ignore')
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'dirname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'].isna().sum()
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
dirname = 'usgs-2016-70cm-neches-river-basin'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2name
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
dir
direxts
p = Path(basename)
resolution = np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))[0]
resolution
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+list(direxts)[0]))
stampede2name
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name
stampede2name_drop = stampede2name[~stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])].index
stampede2name_drop
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] = stampede2name
avail_hu12catchs.drop(index=stampede2name_drop,inplace=True,errors='ignore')
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
dirname = 'stratmap-2020-50cm-north-central-texas'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2name
stampede2names
basename
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
p = Path(basename)
resolution = np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))[0]
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
dirname = 'hgac-2008-1m'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
len(stampede2names)
if len(stampede2names) == 0:
            break
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
direxts
p = Path(basename)
resolution = np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))[0]
resolution
stampede2name = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+list(direxts)[0]))
for subp in p.rglob('*'):
                if len(stampede2name[stampede2name.str.lower()==str(subp).lower()].index)>0:
                    stampede2name.loc[stampede2name[stampede2name.str.lower()==subp.as_posix().lower()].index[0]] = subp.as_posix()
stampede2name
stampede2names
len(stampede2names)
stampede2name_drop = stampede2name[~stampede2name.isin([subp.as_posix() for subp in list(p.rglob('*'))])].index
stampede2name_drop
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name'] = stampede2name
avail_hu12catchs.drop(index=stampede2name_drop,inplace=True,errors='ignore')
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
avail_hu12catchs.dropna(subset=['stampede2name'],inplace=True)
avail_hu12catchs
avail_hu12catchs.loc[avail_hu12catchs['stampede2name'].isna(),'dirname'].unique()
hu12catchs
avail_hu12catchs
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name']
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: 
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name'].apply(lambda fn: fn.suffix))
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name'].apply(lambda fn: os.splitext(fn)[-1]))
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1]))
avail_hu12catchs.groupby(['dirname']).apply(lambda gdf: gdf['stampede2name'].apply(lambda fn: os.path.splitext(fn)[-1]).unique())
dirname = 'stratmap-2013-50cm-ellis-henderson-hill-johnson-navarro'
stampede2names = []
basename = os.path.join(args.raster,dirname,'dem')+os.sep
for fnext in fnexts:
            avail_hu12catchs['demname'] = avail_hu12catchs['demname'].str.replace(fnext+'$','')
            stampede2names.extend(glob.glob(basename+'*'+fnext))
stampede2names
if len(stampede2names) == 0:
            break
direxts = set([os.path.splitext(os.path.basename(name))[1] for name in stampede2names])
direxts
fnexts
p = Path(basename)
resolution = np.unique(np.array([re.split('-|_',subp.parts[-1])[1] for subp in p.rglob('*')]))[0]
resolution
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique().shape[0]
demname = avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique()[0]
truth_dirname = avail_hu12catchs['dirname']==dirname
truth_demname = avail_hu12catchs['demname']==demname
for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+fnext))
fnext
stampede2name
fnexts
direxts
fnext
avail_hu12catchs.loc[truth,'demname']
truth
truth.sum()
avail_hu12catchs.loc[truth,'demname']
avail_hu12catchs
avail_hu12catchs.loc[truth,'demname']
avail_hu12catchs.index
avail_hu12catchs.index.unique()
truth.index
truth.index.unique()
truth_dirname.index
truth_demname.index
avail_hu12catchs
truth.index
truth = np.logical_and(truth_dirname,truth_demname)
truth.index
for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+fnext))
stampede2name
avail_hu12catchs.loc[truth,'demname']
glob.glob(stampede2name.iloc[0]
stampede2name.iloc[0]
stampede2name.unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].shape[0]
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique().shape[0]
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique().apply
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname']
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].apply(
avail_hu12catchs
avail_hu12catchs[avail_hu12catchs['demname']!='No Data Exist']
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique()
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique().shape[0]
for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+fnext))
stampede2name
dirname
for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():                                                                       truth_dirname = avail_hu12catchs['dirname']==dirname                            truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+fnext))
                    if glob.glob(stampede2name.iloc[0]):
                        break
for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():                                                                       truth_dirname = avail_hu12catchs['dirname']==dirname                            truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
for demname in avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'demname'].unique():
                truth_dirname = avail_hu12catchs['dirname']==dirname
                truth_demname = avail_hu12catchs['demname']==demname
                truth = np.logical_and(truth_dirname,truth_demname)
                for fnext in fnexts:
                    stampede2name = avail_hu12catchs.loc[truth,'demname'].apply(lambda x: os.path.join(basename,re.split('-|_',x)[0]+'-'+resolution+'_'+re.split('-|_',x)[2]+fnext))
                avail_hu12catchs.loc[truth,'stampede2name_exts'] = stampede2name
avail_hu12catchs['stampede2name_exts']
avail_hu12catchs[~avail_hu12catchs['stampede2name_exts']
avail_hu12catchs['stampede2name_exts']
avail_hu12catchs.loc[avail_hu12catchs['dirname']==dirname,'stampede2name_exts']
import os
import shutil
args.tempdir
shutil.disk_usage(args.tempdir)
os.sep+args.tempdir
args.tempdir = os.sep+args.tempdir
os.sep+args.tempdir
shutil.disk_usage(args.tempdir)
dir
dir(
dir()
os.getcwd()
import readline
readline.write_history_file('history0.py')
ls
dir()
args.restart
with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
os.sep
os.sep + args.restart
args.restart = s.sep + args.restart
args.restart = os.sep + args.restart
args.restart
with open(args.restart, 'rb') as input:
                flows_keys, flowshu12shape, catchshu12shape, hu12catchs, avail_hu12catchs_grouped, crs = pickle.load(input)
type(avail_hu12catchs_grouped)
gpd.GeoDataFrame(pd.concat(avail_hu12catchs_grouped)).to_file(DEM2basin-1m-rasters_by_HUC12.geojson'
gpd.GeoDataFrame(pd.concat(avail_hu12catchs_grouped)).to_file('DEM2basin-1m-rasters_by_HUC12.geojson',driver='GeoJSON')
gpd.GeoDataFrame(pd.concat(hu12catchs)).to_file('DEM2basin-1m-buffered_HUC12s.geojson',driver='GeoJSON')
exit()
ls
import geopandas as gpd
screen- ls
screen -ls
import os
os.listdir()
csv = gpd.read_file('USGS-DEMs-10m.csv')
csv
csv.columns
csv.head()
csv.iloc[0]
csv
csv['field_17']
csv['field_17'].iloc[0]
csv['field_15'].iloc[0]
csv['field_14'].iloc[0]
csv['field_14'].unique()
csv['field_15'].iloc[0]
csv['field_15']
csv.columns
LS
ls
dir()
csv
exit()
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin-dhardestylewis import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin_dhardestylewis import dem2basin
exit()
from dem2basin import dem2basin
exit()
import dem2basin
exit()
ls
from dem2basin import dem2basin
exit()
from demb2basin import dem2basin
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
exit()
from dem2basin import dem2basin
dem2basin.
help(dem2basin.get_hucs_by_shapefile)
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
dem2basin.get_hucs_by_shapefile(shapefile, hucs_file)
exit()
from dem2basin import dem2basin
dem2basin.__path_
dem2basin.__path__
dem2basin.__file__
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
reload(dem2basin)
from importlib import reload
reload(dem2basin)
dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
reload(dem2basin)
dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
reload(dem2basin)
dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)[0]
hucs,huc_level = dem2basin.get_hucs_by_shapefile(shapefile,hucs_file
hucs,huc_level = dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
hucs
huc_level
hucs
reload(dem2basin)
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
nhd_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/NFIEGeo_TX.gdb')
nhd_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/NFIEGeo_TX.gdb'
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
type(hucs)
hucs.to_crs
reload(dem2basin)
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
reload(dem2basin)
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
reload(dem2basin)
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
import readline
readline.write_history_file('dem2basin-history.py')
exit()
from dem2basin import dem2basin
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
hucs,huc_level = dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
nhd_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/NFIEGeo_TX.gdb'
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
os.getcwd()
import os
os.getcwd()
exit
exit()
from dem2basin import dem2basin
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/WBD-HU12-TX.shp'
hucs,huc_level = dem2basin.get_hucs_by_shapefile(shapefile,hucs_file)
nhd_file = '/scratch/04950/dhl/GeoFlood/DEM2basin/NFIEGeo_TX.gdb'
dem2basin.get_flowlines_by_huc(hucs,nhd_file)
pwd
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines
flowline_representative_points
flowline_representative_points.columns
flowlines
flowlines.columns
flowlines
flowlines.columns
from importlib import reload
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines.columns
flowline_representative_points
flowline_representative_points.columns
flowlines.columns
flowlines
flowline_representative_points
flowline_representative_points.columns
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines.columns
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines
del(dem2basin)
from dem2basin import dem2basin
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines
flowlines.columns
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines
flowlines.columns
flowline_representative_points.columns
flowlines
flowline_representative_points
hucs
flowline_representative_points.sort_index(inplace=True)
flowline_representative_points
flowlines
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
reload(dem2basin)
flowlines,flowline_representative_points = dem2basin.get_flowlines_by_huc(hucs,nhd_file)
flowlines
flowline_representative_points
dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points)
dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points).sort_index()
reload(dem2basin)
dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points)
reload(dem2basin)
dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points)
dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points).columns
catchments = dem2basin.get_catchments_by_huc(hucs,nhd_file,flowline_representative_points)
dem2basin.index_flowlines_by_catchments(flowlines,catchments)
dem2basin.index_flowlines_by_catchments(flowlines,catchments).columns
flowlines = dem2basin.index_flowlines_by_catchments(flowlines,catchments)
dem2basin.buffer_hucs(hucs)
import gpd
import geopandas as gpd
gpd.read_file('
gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/legacy/nwm_flows.gpkg')
gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/legacy/nwm_catchs.gpkg')
gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/legacy/nwm_catchments.gpkg')
from osgeo import ogr
import ogr
import gdal
exit()
import ogr
import gdal
from osgeo import ogr
os.getcwd()
import os
os.getcwd()
driver = ogr.GetDriverByName('GPKG')
dataSource = driver.Open('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/legacy/nwm_catchments.gpkg')
dataSource
layer = dataSource.GetLayer()
layer
layer.GetMetadata()
layer.GetFeature()
layer.chema
layer.schema
dir()
38000.*.25
import fiona
nwm_catchments = fiona.open('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/legacy/nwm_catchments.gpkg')
nwm_catchments
nwm_catchments.schema
nwm_catchments.values()
list(nwm_catchments.values())
nwm_catchments.next()
list(nwm_catchments.values())
nwm_catchments.next()
nwm_catchments.profile
nwm_catchments.meta
nwm_catchments.env
nwm_catchments.crs
nwm_catchments.bounds
nwm_catchments.crs_wkt
nwm_catchments.schema
os.getcwd()
reload(dem2basin)
from importlib import reload
reload(dem2basin)
from dem2basin import dem2basin
filename = 
filename = '/scratch/projects/tnris/dhl-flood-modelling/yan_liu/fim3/inputs/nwm_hydrofabric/nwm_catchments.gpkg'
converted_filename = '/scratch/projects/tnris/dhl-flood-modelling/yan_liu-converted/nwm_hydrofabric/nwm_catchments.gpkg'
bounding_box_vector_file = '/work/04950/dhl/stampede2/GeoFlood/preprocessing/data/WBD_National_GDB/WBD_National_GDB.gdb'
import geopandas as gpd
bounding_box_vector = gpd.read_file(bounding_box_vector_file)
bounding_box_vector
bounding_box_vector = fiona.Open(bounding_box_vector_file)
bounding_box_vector = fiona.open(bounding_box_vector_file)
bounding_box_vector.naem
bounding_box_vector.name
bounding_box_vector = fiona.listlayers(bounding_box_vector_file)
fiona.listlayers(bounding_box_vector_file)
bounding_box_vector = fiona.open(bounding_box_vector_file,layer='WBDHU4')
bounding_box_vector
bounding_box_vector = gpd.read_file(bounding_box_vector_file,layer='WBDHU4')
bounding_box_vector
bounding_box_vector.total_bounds
tuple(bounding_box_vector.total_bounds)
filename
nwm_catchments.crs
nwm_catchments.meta
nwm_catchments.crs_wkt
type(nwm_catchments.crs_wkt)
nwm_catchments.crs_wkt
import pprint
pretty_printer = pprint.PrettyPrinter()
pretty_printer.pprint(nwm_catchments.crs_wkt)
import json
json.loads(nwm_catchments.crs_wkt)
nwm_catchments.crs_wkt
bounding_box_vector.total_bounds
from shapely.geometry import box
*bounding_box_vector.total_bounds
box(*bounding_box_vector.total_bounds)
print(box(*bounding_box_vector.total_bounds))
box(*bounding_box_vector.total_bounds)
gpd.GeoDataFrame(geometry=box(*bounding_box_vector.total_bounds))
gpd.GeoDataFrame(geometry=[box(*bounding_box_vector.total_bounds)])
bounding_box = gpd.GeoDataFrame(geometry=[box(*bounding_box_vector.total_bounds)])
bounding_box
bounding_box.to_crs(nwm_catchments.crs_wkt)
bounding_box_vector.cr
bounding_box_vector.crs
bounding_box = gpd.GeoDataFrame(geometry=[box(*bounding_box_vector.total_bounds)],crs=bounding_box_vector.crs)
bounding_box.to_crs(nwm_catchments.crs_wkt)
bounding_box.to_crs(nwm_catchments.crs_wkt,inplace=True)
bounding_box.iloc[0,'geometry']
bounding_box.iloc[0[]'geometry']
bounding_box.iloc[0]['geometry']
bounding_box.total_bounds
*bounding_box.total_bounds
tuple(*bounding_box.total_bounds)
tuple(bounding_box.total_bounds)
bounding_box = tuple(bounding_box.total_bounds)
bounding_box
converted_filename
os.path.splitext(converted_filename)
os.path.splitext(converted_filename)[0]
os.path.splitext(converted_filename)[0] + '.gdb'
converted_filename = os.path.splitext(converted_filename)[0] + '.gdb'
meta
nwm_catchments.meta
meta = nwm_catchments.meta
meta['driver']
meta['driver'] = 'FileGDB'
nwm_catchments.meta
meta
dem2basin.crop_and_convert_large_vector_file(filename,converted_filename,meta=meta,bounding_box=bounding_box)
meta['driver'] = 'OpenFileGDB'
dem2basin.crop_and_convert_large_vector_file(filename,converted_filename,meta=meta,bounding_box=bounding_box)
meta['driver'] = 'FileGDB'
dem2basin.crop_and_convert_large_vector_file(filename,converted_filename,meta=meta,bounding_box=bounding_box)
from dem2basin import dem2basin
hucs = dem2basin.get_hucs_by_shape(args.shapefile,args.hucs)
shapefile = '/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/HUC12s-Arctur.geojson'
hucs = '/scratch/projects/tnris/dhl-flood-modelling/WBD_National_GDB.gdb'
hucs_file = hucs.copy()
hucs_file = hucs
del(hucs)
hucs = get_hucs_by_shape(shapefile,hucs_file)
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs_file,hucs_layer='WBDHU12')
reload(dem2basin)
from importlib import reload
reload(dem2basin)
hucs = dem2basin.get_hucs_by_shape(shapefile,hucs_file,hucs_layer='WBDHU12')
this
dir()
os.getcwd()
import os
os.getcwd()
import geopandas as gpd
nwm_flows = gpd.read_file('nwm_flows.gdb')
exit()
python3
exit()
import fiona
flows = fiona.open("nwm_flows.gdb")
flows.schema
flows.schema[1:]
pd.DataFrame(flows.schema).index[1:]
import pandas as pd
pd.DataFrame(flows.schema).index[1:]
pd.DataFrame(flows.schema).index[1:].to_list()
ignore_fields = pd.DataFrame(flows.schema).index[1:].to_list()
flows_ids = fiona.open("nwm_flows.gdb",ignore_fields=ignore_fields)
flows_ids
flows_ids.next()
type(flows_ids)
flows_ids
flows_ids.ignore_
flows_ids.ignore_fields
fiona.__version__
next(iter(flows_ids))
import geopandas as gpd
gpd.read_file('nwm_lakes.gdb',ignore_geometry=True)
gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/gdb.d/fim3/inputs/nwm_hydrofabric/nwm_lakes.gdb',ignore_geometry=True)
gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/gdb.d/fim3/inputs/nwm_hydrofabric/nwm_flows.gdb',ignore_geometry=True)
nwm_flows = gpd.read_file('/scratch/projects/tnris/dhl-flood-modelling/yan_liu/gdb.d/fim3/inputs/nwm_hydrofabric/nwm_flows.gdb',ignore_geometry=True)
dir()
exit()
from dem2basin import dem2basin
770000.*.
770000.*.1/5.
770000./5.
770000.*.1
770000./5.
161184.
161184.*.1
dir()
os.getcwd()
import os
os.getcwd()
dem2basin
dem2basin.
hucs = get_hucs_by_shape(args.shapefile,args.hucs)
hucs = dem2basin.get_hucs_by_shape(args.shapefile,args.hucs)
shapefile = 'HUC12s-Arctur.geojson'
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
temporary_huc_file
hucs.to_file('/scratch/04950/dhl/tmp/hucs.geojson')
hucs.to_file('/scratch/04950/dhl/tmp/hucs.geojson',driver='GeoJSON')
hucs_file_to_check = '/scratch/04950/dhl/tmp/hucs.geojson'
hucs_file_to_check
hucs_file_to_check = Path(hucs_file_to_check)
hucs_file_to_check
hucs.to_file('/scratch/04950/dhl/tmp/hucs.geojson',driver='GeoJSON')
hucs['HUC'].astype(str).isin(output_hucs_existing)
hucs[25]
hucs.iloc[25]
dir()
hucs_test_file = '/scratch/04950/dhl/tmp/hucs.geojson'
Path(hucs_test_file)
hucs_test_file = Path(hucs_test_file)
hucs_test_file
hucs_test_file.parent.joinpath(catchments.geojson)
hucs_test_file.parent.joinpath('catchments.geojson')
catchments_test_file = hucs_test_file.parent.joinpath('catchments.geojson')
catchments_test_file
catchments.to_file(catchments_test_file)
flowlines
flowlines_test_file = hucs_test_file.parent.joinpath('flowlines.geojson')
flowlines.to_file(flowlines_test_file)
catchments.to_file(catchments_test_file,driver='GeoJSON')
flowlines.to_file(flowlines_test_file,driver='GeoJSON')
ls
pwd
dir()
lidar_index_test_file = hucs_test_file.parent.joinpath('lidar_index.geojson')
lidar_index.to_file(lidar_index_test_file,driver='GeoJSON')
huc = hucs[25]
huc = hucs.iloc[25]
huc
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique().shape[0]]
huc = hucs.iloc[[25]]
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique().shape[0]]
output_raster_filename
Path(output_raster_filename).parent.parent
Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+huc['HUC'].unique().shape[0]).joinpath('Elevation.tif')
Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+str(huc['HUC'].unique().shape[0])).joinpath('Elevation.tif')
Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+str(huc['HUC'].unique())).joinpath('Elevation.tif')
Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+str(huc['HUC'].unique()[0])).joinpath('Elevation.tif')
output_raster_filename = Path(output_raster_filename).parent.parent.joinpath('HUC12s-Arctur-'+str(huc['HUC'].unique()[0])).joinpath('Elevation.tif')
output_raster_filename
parent_temporary_directory
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
reload(dem2basin)
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
reload(dem2basin)
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
lidar_index_by_huc
lidar_index_by_huc = lidar_index[lidar_index['HUC']==str(huc['HUC'].unique().shape[0])]
lidar_index_by_huc
lidar_index
lidar_index['HUC'].unique()
huc
dir()
huc_fi
hucs_file_to_check
huc_id
huc_prefix
huc
huc['HUC'].unique()
huc['HUC'].unique()[0]
type(huc['HUC'].unique()[0])
lidar_index.dtypes
lidar_index_by_huc = lidar_index[lidar_index['HUC']==huc['HUC'].unique()[0]]
lidar_index_by_huc
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
filenames
src_dems_dir = '/scratch/04950/dhl/tmp/src_dems'
[shutil.copyfile(copy,Path(str(src_dems_dir)).joinpath(Path(str).name)) for copy in filenames]
import shutil
[shutil.copyfile(copy,Path(str(src_dems_dir)).joinpath(Path(str).name)) for copy in filenames]
[shutil.copyfile(copy,str(Path(str(src_dems_dir)).joinpath(Path(str).name))) for copy in filenames]
[shutil.copyfile(str(copy),str(Path(str(src_dems_dir)).joinpath(Path(str).name))) for copy in filenames]
[(str(copy),str(Path(str(src_dems_dir)).joinpath(Path(str).name)) for copy in filenames]
[str(copy),str(Path(str(src_dems_dir)).joinpath(Path(str).name) for copy in filenames]
[str(copy),str(Path(str(src_dems_dir)).joinpath(Path(str).name)) for copy in filenames]
filenames
filenames[0]
type(filenames[0])
[Path(str(filename)).name for filename in filenames]
[Path(str(src_dems_dir)).joinpathPath(str(filename)).name for filename in filenames]
[Path(str(src_dems_dir)).joinpath(Path(str(filename)).name) for filename in filenames]
src_dems = [Path(str(src_dems_dir)).joinpath(Path(str(filename)).name) for filename in filenames]
[(fns[0],fns[1]) for fns in zip(filenames,src_dems)]
[shutil.copyfile(fns[0],fns[1]) for fns in zip(filenames,src_dems)]
lidar_index_by_huc['lidar_file']
len(filenames)
filenames = lidar_index_by_huc['lidar_file'].to_list()
src_dems = [Path(str(src_dems_dir)).joinpath(Path(str(filename)).name) for filename in filenames]
[shutil.copyfile(fns[0],fns[1]) for fns in zip(filenames,src_dems)]
import pyprog
import pyproj
pyproj.CRS('EPSG:4326')
pyproj.CRS('EPSG:4326').accuracy
from pyproj import from_crs
from pyproj import Proj
p = Proj()
p = Proj('EPSG:4326')
p.accuracy
p.crs
p.from_crs('EPSG:6344').accuracy
p.from_crs('EPSG:6344',p.crs).accuracy
filenames
lidar_index_by_huc
lidar_index_by_huc['dirname']
lidar_index_by_huc['dirname'].unique()
lidar_index_by_huc.groupby(['dirname'])
lidar_index_by_huc.groupby(['dirname']).first()
lidar_index_by_huc.groupby(['dirname']).first()['lidar_file'].apply(lambda fn: gdal.Open(fn).crs)
import gdal
from osgeo import gdal
import gdal
lidar_index_by_huc.groupby(['dirname']).first()['lidar_file'].apply(lambda fn: gdal.Open(fn).crs)
lidar_index_by_huc.groupby(['dirname']).first()['lidar_file'].apply(lambda fn: gdal.Open(fn).srs)
lidar_index_by_huc.groupby(['dirname']).first()['lidar_file'].iloc[0]
lidar_file = '/scratch/projects/tnris/tnris-lidardata/usgs-2017-70cm-brazos-freestone-robertson/dem/usgs17-1m_14RPV690620.img'
lidar_file_obj =  gdal.Open(lidar_file)
lidar_file_obj
lidar_file_obj.GetProjection()
projection = lidar_file_obj.GetProjection()
projection
lidar_index_by_huc.groupby(['dirname']).groupby()
lidar_index_by_huc.groupby(['dirname']).count()
lidar_index_by_huc.count(['dirname'])
lidar_index_by_huc.count(['dirname']).size()
lidar_index_by_huc.groupby(['dirname']).size()
lidar_index_by_huc.groupby(['dirname']).size().sort()
lidar_index_by_huc.groupby(['dirname']).size().sort_values()
lidar_index_by_huc.groupby(['dirname']).size().sort_values().index
lidar_index_by_huc.groupby(['dirname']).size()
test = lidar_index_by_huc.groupby(['dirname']).size()
test.append({'test':1})
test.append(data={'test':1})
test.append({'test':1})
lidar_index_by_huc
test = lidar_index_by_huc.copy()
test.iloc[-1]
test.iloc[-1].index
test.iloc[-1].name
test.iloc[-1].name+1
test.iloc[test.iloc[-1].name+1]['dirname'] = 'test'
test.append({'dirname':'test'})
test.append({'dirname':'test'},ignore_index=True)
test = test.append({'dirname':'test'},ignore_index=True)
test
lidar_index_by_huc.groupby(['dirname']).size().sort_values()
test.groupby(['dirname']).size().sort_values()
test.groupby(['dirname']).size()
test.groupby(['dirname']).size().sort_values(ascending=False)
lidar_index_by_huc
lidar_index_by_huc.groupby('dirname')
test = lidar_index_by_huc.groupby('dirname')
[test.get_group(x) for x in test.groups]
by_project = [test.get_group(x) for x in test.groups]
by_project
[project.shape[0] for project in by_project]
[{project['dirname'].unique()[0]:project.shape[0]} for project in by_project]
pd.DataFrame([{project['dirname'].unique()[0]:project.shape[0]} for project in by_project])
pd.DataFrame([{project['dirname'].unique()[0]:project.shape[0]} for project in by_project],axis=1)
pd.DataFrame([{project['dirname'].unique()[0]:project.shape[0]} for project in by_project])
pd.DataFrame(data=[{project['dirname'].unique()[0]:project.shape[0]} for project in by_project])
[{project['dirname'].unique()[0]:project.shape[0]} for project in by_project]
[[project['dirname'].unique()[0],project.shape[0]] for project in by_project]
dataframe_data = [[project['dirname'].unique()[0],project.shape[0]] for project in by_project]
pd.DataFrame(dataframe_data)
pd.DataFrame(dataframe_data,columns='dirname','count')
pd.DataFrame(dataframe_data,columns=['dirname','count'])
lidar_projects_with_counts.sort_values(by=['count'])
lidar_projects_with_counts = pd.DataFrame(dataframe_data,columns=['dirname','count'])
lidar_projects_with_counts.sort_values(by=['count'])
lidar_projects_with_counts.sort_values(by=['count'],ascending=False)
lidar_projects_with_counts.sort_values(by=['count'],ascending=False,inplace=True)
lidar_index_by_huc.groupby('dirname').first()
lidar_index_by_huc.groupby('dirname').first()[['dirname','lidar_file']]
lidar_index_by_huc.groupby('dirname').first()[['lidar_file']]
lidar_index_by_huc.groupby('dirname').first()[['lidar_file']].reset_index()
lidar_projects_with_info_tile = lidar_index_by_huc.groupby('dirname').first()[['lidar_file']].reset_index()
lidar_projects_with_info_tile
lidar_projects_with_info_tile.merge(lidar_projects_with_counts,on=['dirname'])
lidar_projects_with_info_tile.merge(lidar_projects_with_counts,on=['dirname'],inplace=True)
lidar_projects_with_info_tile.merge(lidar_projects_with_counts,on=['dirname'])
lidar_projects_with_counts = lidar_projects_with_info_tile.merge(lidar_projects_with_counts,on=['dirname'])
lidar_projects_with_counts.sort_values(
        by = ['count'],
        ascending = False,
        inplace = True
    )
lidar_projects_with_counts['lidar_file'].apply(lambda fn: gdal.Open(fn).GetProjection())
lidar_projects_with_counts['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts['lidar_file'].apply(lambda fn: pyproj.CRS.from_proj4(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts.iloc[0]['lidar_file'].apply(lambda fn: gdal.Open(fn).GetProjection())
lidar_projects_with_counts['crs'] = lidar_projects_with_counts.iloc[0]['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts['crs'] = lidar_projects_with_counts['lidar_file'].apply(lambda fn: pyproj.CRS.from_wkt(gdal.Open(fn).GetProjection()))
lidar_projects_with_counts
:w
lidar_projects_with_counts
filenames
filenames.shape[0]
len(filenames)
vrts_to_composite
vrts_to_composite = []
for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            huc_prefix +
            '-' +
            project +
            '.vrt'
        ))
for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            huc_prefix +
            '-' +
            project +
            '.vrt'
        ))
type(huc_prefix)
type(project)
    for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
    for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
    for project in lidar_projects_with_counts['dirname'].to_list():
for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
vrts_to_composite
from itertools import cycle
filenames
vrts_to_composite
for filenames,vrts_to_composite in zip(cycle(filenames),vrts_to_composite):
        build_vrt(filenames,vrts_to_composite)
for filenames,vrts_to_composite in zip(cycle(filenames),vrts_to_composite):
    dem2basin.build_vrt(filenames,vrts_to_composite)
filenames_repeated = [filenames] * len(vrts_to_composite)
vrts_to_composite
vrts_to_composite = []
for project in lidar_projects_with_counts['dirname'].to_list():
#        filenamess.append(lidar_projects_with_counts[
#            lidar_projects_with_counts['dirname'] == project
#        ]['lidar_file'].to_list())
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
vrts_to_composite
filenames_repeated = [filenames] * len(vrts_to_composite)
filenames_repeated
filenames
filenames = lidar_index_by_huc['lidar_file'].to_list()
filenames
filenames_repeated = [filenames] * 2
filenames_repeated
for filenames_inner,vrts_to_composite in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        build_vrt(filenames_inner,vrts_to_composite)
for filenames_inner,vrts_to_composite in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        dem2basinbuild_vrt(filenames_inner,vrts_to_composite)
for filenames_inner,vrts_to_composite in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        dem2basin.build_vrt(filenames_inner,vrts_to_composite)
vrts_to_composite
vrts_to_composite = []
    for project in lidar_projects_with_counts['dirname'].to_list():
for project in lidar_projects_with_counts['dirname'].to_list():
        vrts_to_composite.append(Path(str(temporary_directory)).joinpath(
            str(huc_prefix) +
            '-' +
            str(project) +
            '.vrt'
        ))
vrts_to_composite
filenames_repeated = [filenames] * len(vrts_to_composite)
filenames
filenames_repeated
for filenames_inner,vrts_to_composite_inner in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        build_vrt(filenames_inner,vrts_to_composite_inner)
for filenames_inner,vrts_to_composite_inner in zip(
        filenames_repeated,
        vrts_to_composite
    ):
        dem2basin.build_vrt(filenames_inner,vrts_to_composite_inner)
vrts_to_composite
reprojected_vrts_filenames = [
        Path(str(vrt)).parent.joinpath(
            Path(
                os.splitext(str(Path(str(vrt)).name))[0] +
                'reprojected.vrt',
            )
        )
        for vrt
        in vrts_to_composite
    ]
reprojected_vrts_filenames = [
        Path(str(vrt)).parent.joinpath(
            Path(
                os.path.splitext(str(Path(str(vrt)).name))[0] +
                'reprojected.vrt',
            )
        )
        for vrt
        in vrts_to_composite
    ]
reprojected_vrts_filenames
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
reprojected_vrts_filenames
for vrt,reprojected_vrt,crs in zip(
        vrts_to_composite,
        reprojected_vrts_filenames,
        lidar_projects_with_counts['crs'].to_list()
    ):
        reproject_raster(
            vrts_to_composite,
            reprojected_vrts_filenames,
            dst_crs = crs
        )
for vrt,reprojected_vrt,crs in zip(
        vrts_to_composite,
        reprojected_vrts_filenames,
        lidar_projects_with_counts['crs'].to_list()
    ):
        reproject_raster(
            vrts,
            reprojected_vrt,
            dst_crs = crs
        )
for vrt,reprojected_vrt,crs in zip(
        vrts_to_composite,
        reprojected_vrts_filenames,
        lidar_projects_with_counts['crs'].to_list()
    ):
        reproject_raster(vrt,reprojected_vrt,dst_crs=crs)
for vrt,reprojected_vrt,crs in zip(
        vrts_to_composite,
        reprojected_vrts_filenames,
        lidar_projects_with_counts['crs'].to_list()
    ):
        dem2basin.reproject_raster(vrt,reprojected_vrt,dst_crs=crs)
temporary_vrt_file = temporary_directory.joinpath(
        str(huc_prefix) + '.vrt'
    )
build_vrt(reprojected_vrts_filenames,temporary_vrt_file)
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
out_raster_filename
output_raster_filename
reload(dem2basin)
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
import pyproj
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
reload(dem2basin)
import pyproj
dem2basin._get_mosaic_and_output_raster(lidar_index_by_huc,huc,output_raster_filename,parent_temporary_directory)
flowline_filenames = []
subdirectory
hucs
hucs['HUC'].unique()
hucs['HUC'].unique().to_list()
hucs['HUC'].unique()
huc_ids = list(hucs['HUC'].unique())
huc_ids
dir()
subdirectory
subdirectory.parent
Path(str(subdirectory)).parent
[Path(str(subdirectory)).parent.join_path('HUC12s-Arctur-'+str(huc_id)) for huc_id in huc_ids]
[Path(str(subdirectory)).parent.joinpath('HUC12s-Arctur-'+str(huc_id)) for huc_id in huc_ids]
subdirectories = [Path(str(subdirectory)).parent.joinpath('HUC12s-Arctur-'+str(huc_id)) for huc_id in huc_ids]
[subdirectory.joinpath('Flowline.shp') for subdirectory in subdirectories]
flowlines_files = [subdirectory.joinpath('Flowline.shp') for subdirectory in subdirectories]
roughnesses_files = [subdirectory.joinpath('Roughness.csv') for subdirectory in subdirectories]
catchments_files = [subdirectory.joinpath('Catchment.shp') for subdirectory in subdirectories]
elevations_files = [subdirectory.joinpath('Elevation.tif') for subdirectory in subdirectories]
flowlines.groupby(['Flowline.shp'])
flowlines.groupby(['HUC'])
flowlines.groupby(['HUC']).apply(list)
flowlines.groupby(['HUC']).apply(list).groupby(level=0)
flowlines.groupby(['HUC']).apply(list).groupby(level=0).apply(list)
flowlines.groupby(['HUC'])
flowlines
flowlines['COMID'].unique().shape0[0]
flowlines['COMID'].unique().shape[0]
flowlines
[huc for _,huc in flowlines.groupby('HUC')
[huc for _,huc in flowlines.groupby('HUC')]
flowlines
flowlines_by_huc = [huc for _,huc in flowlines.groupby('HUC')]
catchments_by_huc = [huc for _,huc in catchments.groupby('HUC')]
lidar_indices_by_huc = [huc for _,huc in lidar_index.groupby('HUC')]
roughnesses_by_huc = [huc for _,huc in roughnesses.groupby('HUC')]
flowlines_files
import readline
readline.write_history_file('final_product-history.py')
