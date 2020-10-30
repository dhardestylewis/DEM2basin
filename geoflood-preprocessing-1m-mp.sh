#!/bin/bash
#
#-----------------------------------------------------------------------------
# This Stampede-2 job script is designed to create preprocess GeoFlood raw inputs
#
# To submit the job, issue: "sbatch <sbatch_script>.sh" 
#
# For more information, please consult the User Guide at: 
#
# https://portal.tacc.utexas.edu/user-guides/stampede2
#-----------------------------------------------------------------------------
#
#SBATCH -J geoflood-preproc-1m-mp                  # Job name
#SBATCH -o geoflood-preproc-1m-mp.o%j              # Name of stdout output file (%j expands to jobId)
#SBATCH -e geoflood-preproc-1m-mp.e%j              # Name of stdout output file (%j expands to jobId)
#SBATCH -p skx-normal                # Queue name
#SBATCH -N 1                          # Total number of nodes requested (68 cores/node)
#SBATCH -n 48                         # Total number of mpi tasks requested
#SBATCH -t 48:00:00                   # Run time (hh:mm:ss) - 4 hours
#SBATCH -A PT2050-DataX

#--------------------------------------------------------------------------
# ---- You normally should not need to edit anything below this point -----
#--------------------------------------------------------------------------

set -x

module unload python2
#export PYTHONPATH="$WORK/miniconda3/envs/hand-taudem/lib/python3.8/site-packages:$PYTHONPATH"
#conda env list
#conda init
eval "$(conda shell.bash hook)" ## Properly initialise non-interactive shell
conda activate geoflood-pre

PREPROC_WORK='${PREPROC_WORK}'
PREPROC_SCRATCH='/scratch/04950/dhl/GeoFlood-preprocessing'
PREPROC_TNRIS='/scratch/projects/tnris'
python3 "${PREPROC_SCRATCH}/geoflood-preprocessing-1m-mp-utm.py" \
    --shapefile "${PREPROC_SCRATCH}/TXDoT-TX_boundary.shp/TXDoT-TX_boundary.shp" \
    --huc12 "${PREPROC_WORK}/WBD-HU12-TX.shp/WBD-HU12-TX.shp" \
    --nhd "${PREPROC_WORK}/NFIEGeo_TX.gdb" \
    --raster "${PREPROC_TNRIS}/tnris-lidardata" \
    --availability "${PREPROC_WORK}/TNRIS-LIDAR-Availability-20200219.shp/TNRIS-LIDAR-Availability-20200219.shp" \
    --directory "${PREPROC_TNRIS}/dhl-flood-modelling/TX-HUC12-DEM_outputs" \
    --restart "${PREPROC_SCRATCH}/geoflood-preprocessing-TX.pickle"

#if [ "x$TACC_RUNTIME" != "x" ]; then
#    # there's a runtime limit, so warn the user when the session will die
#    # give 5 minute warning for runtimes > 5 minutes
#    H=`echo $TACC_RUNTIME | awk -F: '{print $1}'`
#    M=`echo $TACC_RUNTIME | awk -F: '{print $2}'`
#    S=`echo $TACC_RUNTIME | awk -F: '{print $3}'`
#    if [ "x$S" != "x" ]; then
#        # full HH:MM:SS present
#        H=$(($H * 3600))
#        M=$(($M * 60))
#        TACC_RUNTIME_SEC=$(($H + $M + $S))
#    elif [ "x$M" != "x" ]; then
#        # only HH:MM present, treat as MM:SS
#        H=$(($H * 60))
#        TACC_RUNTIME_SEC=$(($H + $M))
#    else
#        TACC_RUNTIME_SEC=$S
#    fi
#
#    if [ $TACC_RUNTIME_SEC -gt 300 ]; then
#        sleep $(($TACC_RUNTIME_SEC - 300)) && echo "$USER's VNC session on $VNC_DISPLAY will end in 5 minutes.  Please save your work now." | wall &
#    fi
#fi
#
