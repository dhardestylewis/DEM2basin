# DEM to basin preprocessing
This preprocessing library takes source 1-meter digital elevation model (DEM) data and splits, crops, buffers, and reprojects it to individual hydrologic basins (identified by their unique identifier, the "HUC12" ID).

This preprocessing script also produces ancillary data products corresponding to each new HUC12 DEM raster to describe their sub-basins (ie "catchments"), their streams  (ie "flowlines"), and the roughness of each streambed.

Taken together, these are the major inputs needed to run [GeoFlood](https://github.com/passaH2O/GeoFlood), which creates short-term flood projections.


## Alpha-release Python library
The underlying Python library is available on PyPi and can be installed by:
```
pip install dem2basin
```


## Main Python script
The recommended way to run `dem2basin.py`:
```
python3 dem2basin.py \
    --shapefile study_area_polygon.shp \
    --huc12 WBD-HUC12s.shp \
    --nhd NHD_catchments_and_flowlines.gdb/ \
    --raster TNRIS-LIDAR-Datasets/ \
    --availability TNRIS-LIDAR-Dataset_availability.shp \
    --directory HUC12-DEM_outputs/ \
    --restart dem2basin-study_area.pickle
```

## Required source data inputs
There are 5 required inputs.
* `--shapefile` _Study area polygon vector GIS file:_ a vector GIS file of a single polygon which defines the study area
* `--huc12` _[USGS Watershed Boundary Dataset (WBD)](https://www.usgs.gov/core-science-systems/ngp/national-hydrography/watershed-boundary-dataset):_ a HUC12 vector GIS file from USGS's WBD or a subset of the WBD
* `--nhd` _[NHD Medium Resolution (MR)](https://www.usgs.gov/core-science-systems/ngp/hydrography/about/nhd-medium-resolution):_ a GeoDataBase of NHD MR catchments and flowlines
* `--raster` _[TNRIS Lidar](https://tnris.org/stratmap/elevation-lidar/):_ the parent directory to a collection of TNRIS Lidar datasets
* `--availability` _[TNRIS Lidar availability](https://data.tnris.org/5751f066-28be-46af-b795-08387a27da6e/resources/tnris-lidar_48_vector.zip):_ a vector GIS file of TNRIS Lidar availability, provided by TNRIS [here](https://tnris.org/stratmap/elevation-lidar/)

## Optional parameters
* `--directory` _Outputs directory:_ a directory to store outputs, which will each be sorted by HUC12
* `--restart` _Restart file:_ a [Python Pickle file](https://docs.python.org/3/library/pickle.html) from which you can restart the preprocessing if it's interrupted
* `--overwrite` _Overwrite flag:_ optional flag to overwrite all files found in output directory
* `--overwrite_rasters` _Overwrite rasters flag:_ optional flag to overwrite just the raster outputs
* `--overwrite_flowlines` _Overwrite flowlines flag:_ optional flag to overwrite just the flowline outputs
* `--overwrite_catchments` _Overwrite catchments flag:_ optional flag to overwrite just the catchment outputs
* `--overwrite_roughnesses` _Overwrite roughness table flag:_ optional flag to overwrite the roughness table
* `--log` _Log file:_ a file to store runtime log

## Description of outputs
There are 4 outputs per HUC12.
* _Cropped & buffered DEM:_
    * buffered 500m
    * cropped to each HUC12 intersecting the study area
    * at least 1m resolution
    * mosaicked with preference for lowest resolution tiles
    * reprojected to the study area's projections
* _corresponding NHD MR flowlines:_
    * subset of NHD MR flowlines
    * each flowline's median point along the line lies within the HUC12
    * reprojected to the study area's projections
* _corresponding NHD MR catchments:_
    * subset of NHD MR catchments
    * correspond with the NHD MR flowlines above
    * reprojected to the study area's projections
* _Manning's n roughness table:_
    * organized by flowline using their [ComIDs](https://nhd.usgs.gov/userGuide/Robohelpfiles/NHD_User_Guide/Feature_Catalog/Data_Dictionary/Data_Dictionary.htm) 
    * vary by stream order

Here is an example of these outputs, originally visualized by [Prof David Maidment](https://www.caee.utexas.edu/faculty/directory/maidment).
![Example outputs](https://github.com/dhardestylewis/DEM2basin/blob/master/images/DEM-HUC12-Outputs_example.jpg)

## Already preprocessed DEMs
Already preprocessed DEMs are now available for the vast majority of Texas's HUC12s if you are a [TACC user](https://portal.tacc.utexas.edu/). You can request a TACC account [here](https://portal.tacc.utexas.edu/account-request).
### Notes about preprocessed DEMs
* The DEMs are not provided for any HUC12s that have any gap in 1m resolution data.
* All of the DEMS are reprojected to [WGS 84 / UTM 14N](https://epsg.io/32614), even if the HUC12 is outside of UTM 14.
### Where to find them
The DEMs are located on [Stampede2](https://www.tacc.utexas.edu/systems/stampede2) at `/scratch/projects/tnris/dhl-flood-modelling/TX-HUC12-DEM_outputs`.
### If you run into trouble
Please [submit a ticket](https://portal.tacc.utexas.edu/tacc-consulting) if you have trouble accessing this data. You may also contact me directly at [@dhardestylewis](https://github.com/dhardestylewis) or <dhl@tacc.utexas.edu>
### Available preprocessed HUC12s
These HUC12 DEMs are available right now on [Stampede2](https://www.tacc.utexas.edu/systems/stampede2).
![Available HUC12 DEMs](https://github.com/dhardestylewis/DEM2basin/blob/master/images/DEM-HUC12-Availability.png)
### Confirmed successfully preprocessed HUC12s
These HUC12 DEMs have been successfully preprocessed in the past, and will soon be available once again on [Stampede2](https://www.tacc.utexas.edu/systems/stampede2). If you need any of these _right now_, please contact me.
![Confirmed HUC12 DEMs](https://github.com/dhardestylewis/DEM2basin/blob/master/images/DEM-HUC12-Confirmed.png)

## Preprocessing workflow
If you would like an understanding of the preprocessing workflow, I provide a simplified but representative example in this [Jupyter notebook](https://github.com/dhardestylewis/DEM2basin/blob/master/dem2basin.ipynb). This Jupyter notebook was presented at the inaugural [TACC Institute on Planet Texas 2050 Cyberecosystem Tools](https://bridgingbarriers.utexas.edu/pt2050-tacc-institute/) in August, 2020. Please contact me if you would like a recording.

