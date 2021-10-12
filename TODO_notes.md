Good!

Here's a small set of outputs on Stampede2:

/scratch/projects/tnris/dhl-flood-modelling/GeoFlood/DEM2basin-1m/Output-Arctur

As you will see, there are several defects with the mosaicked rasters produced here. Nearly all of these defects can be categorized as follows:
    - Could not locate underlying Lidar filename during LidarIndex() -- this will be corrected as we improve out metadata of the 40TB Lidar dataset
    - My current raster mosaicking algorithm did not include underlying specific Lidar files in mosaic because:
        - projection "did not match":
            - often the projection is functionally the same (ie reports the same EPSG) but GDAL considers different WKT strings different projections
            - or the projection is functionally the same for our purposes -- Tim Whiteaker has determined that the GeoFlood HAND algorithm is not significantly impacted by adjacent UTM reprojections (ie using UTM 14 in a UTM 15 zone, or UTM 14 in a UTM 13 zone)
            - or the projection actually didn't match for whatever reason
        - GDAL reported underlying Lidar file is not projected in a "SingleCRS" -- haven't yet researched how to resolve this one
        - Memory error because the raster is too large -- because we are strictly producing HUC12s right now, if this happens we can only log it for that particular HUC12 and skip that HUC12 for the time being

So there are two parallel tracks forward right now:

    1) improve LidarIndex in a variety of ways:
        - generalize the internal routines to apply to Lidar, Fathom, NED, or MERIT-DEM, or so that at least it is much easier to derive them for these other datasets, in particular:
            - create a single GeoDataFrame (or PostgreSQL) representation of these multiple datasets, which includes an attribute recording the data source
        - enable LidarIndex to be updated on subsequent runs, rather than having to rerun it each time to rebuild the database of file names and file metadata, hopefully by:
            - converting LidarIndex to a proper class method, rather than a function method in a class wrapper
            - marking individual Lidar projects as "complete" when we are confident each filename within one has been successfully identified and its metadata recorded
            - store the Lidar metadata in a PostgreSQL database (to be stored under PT2050 DataX PATH location or PATH VM from Je'aime), and write up the Python methods to retrieve the data from this database / generalize the use of the 'lidar_index' GeoDataFrame throughout to support either PostgreSQL or GeoDataFrame inputs
        - when locating files in the Lidar dataset, record the following extra information:
            - file size
            - CRS (ie WKT projection)
            - EPSG
            - Geographic CRS
        - recording on a per-HUC basis:
            - whether or not all of the associated Lidar availability polygons intersecting that HUC were found, or whether the ones that were found covered the HUC

    2) Improve the raster mosaicking algorithm and QA/QC:
        - Create a database per HUC to log every run attempted for that HUC and various aspects of that run:
            - whether the output raster succeeds against various unit tests (ie QA/QC), such as:
                - Did we use all of the data available? If not, what was excluded?
                - Is there any no-data within the produced raster?
            - what kind of raster was ultimately produced?
                - if multiple data sources were used, which data sources were ultimately included? (mosaicking at the highest resolution available usually means older Lidar projects are excluded)
                - create a vector image depicting using polygons the ultimate data sources used
                - what is the reprojection accuracy? (can used pyproj.Transformer.accuracy to determine)
           - Timing and memory profiling each HUC raster mosaicking attempt, recording which raster mosaicking algorithm was used for that attempt, getting a rough metric of memory_used-to-tile_filesizes for each algorithm

To do this:
