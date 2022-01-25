# segment-classify-track

This repository contains all the steps necessary for the annotation and tracking of a homogenous or heterozygous population of epithelial cells in time lapse microscopy data.

Each step has an associated Jupyter notebook which should be run in the following sequential order:

1. Image alignment (align.ipynb)
2. Cellular segmentation (cellpose_segmentation.ipynb/stardist_segmentation.ipynb)
3. Phenotype classification and object tracking (cellx_classify_and_btrack.ipynb)
4. Viewer (napari_viewer.ipynb)

The repository is designed to work with the raw output of any timelapse microscopy data set that is structured with the following path pattern:

`/expt_ID/PosID/img_channel000_position000_time000000000_z000.tif/`


i.e. `Nathan/ND0000/Pos0/Pos0_aligned/img_channel000_position...z000.tif`

This pipeline was originally designed to label a 2-class, mixed population cell competition images (a wild-type cell versus a mutant), but now has been modified to be more broadly applicable to a homogenous population of cell types, with the 2-class pipeline in a separate subdirectory.

Image loading is done via the DaskOctopusLiteLoader function from https://github.com/lowe-lab-ucl/octopuslite-reader

Image: 3D (x,y,t) rendering of single-cell tracks belonging to a competitive MDCK epithelium. 

![](tracks_hd.png)