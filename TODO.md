# TODO file
File template shamelessly stolen from [todo.md](https://raw.githubusercontent.com/todomd/todo.md/master/TODO.md)

### ToDo

- [ ] Add Conda environment description to `README.md`
- [ ] Add Conda `environment.yml`
- [ ] Add testing information (eg Stampede2, Ubuntu VM, Linux, Unix, etc) to `README.md`
- [ ] Refer to Jupyter notebook in `README.md`
- [ ] Include Tim Whiteaker write-up about cross-UTM, attribute him, "not definitive, but guiding"
- [ ] Include Alec Carruthers write-up about Manning's values, attribute him, attribute him for original workflow design
- [ ] Reference presentation to inaugural annual TACC Institute PT2050 Cyberecosystem Tools: make recording presenting the Jupyter notebook
- [ ] Make section referring to downstream projects such as HAND-TauDEM and Flooded_lots ("where you can go from here")
- [ ] Reference outgoing president [AAAI's address](https://vimeo.com/400177695) of downstream Flooded_lots work (at time 39:18)
- [ ] Change catchment images to HUC12 images
- [ ] Combine confirmed and available HUC12 images when they are they same
- [ ] Change name from "GeoFlood-preprocessing" to "DEM preprocessing" or "source to basin DEM preprocessing"
- [ ] Provide basic description or abstract of this tool, dataset. Include resolution
- [ ] Sample citation
- [ ] Add sponsor info (TACC, CWE, NOAA (and award number), Planet Texas (and Paola's project name), etc)
- [ ] For available preprocessed DEMs, refer to exact LIDAR availability date (2020/02/19)
- [ ] Unify name of Jupyter notebook and scripts
- [ ] Provide ancillary scripts which do the same thing for the US (10m) and for the World (90m)
- [ ] Prepare paper outline ("data paper" about "data collection"): here's a [template](https://par.nsf.gov/biblio/10143795)
- [ ] Prepare for submission to TDR, OpenTopography, or other digital repositories
- [ ] Produce a choropleth of missing or present HUC12s with each color corresponding to reason why missing
- [ ] Specify output format, maybe give option to select formats
- [ ] Possibly include each detail used by [OpenTopography](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.092020.2193.2) for its datasets
- [ ] Include [AAAI slides](https://www.dropbox.com/scl/fi/0opj6ff7cjyh0gpc2ettm/HANDEdits_PA-MINT.pptx?dl=0&rlkey=70qk5cxlyz04qvr74uwedwj1d) (they are draft) (slides 13-15 in this draft)
- [ ] Replace AAAI draft slides with final from Yolanda Gil
- [ ] Include [video and slides](https://csdms.colorado.edu/wiki/Presenters-0473) from Dr Pierce's keynote speech at CSDMS
- [ ] Check for slivers of nodata in HUC12 DEMs outside of UTM 14N -- report back results to the CWE
- [ ] For buffered HUC12s entirely outside of UTM 14N, replace the UTM 14N HUC12 DEMs with HUC12 DEMs in their respective UTMs (13N & 15N)
- [ ] Regroup this repository into CWE group's
- [ ] Link against existing CWE repositories where appropriate
- [ ] In QAQC error-file generation: distinguish nodata from >1m data instances
- [ ] Produce visualization of where I will not be providing preprocessed HUC12s because of known errors or constraints imposed on this project

### In Progress

- [ ] 

### Done ✓

- [x] Add image of outputs to `README.md`
- [x] Add image of preprocessed HUC12 DEM availability for all of Texas to `README.md`
