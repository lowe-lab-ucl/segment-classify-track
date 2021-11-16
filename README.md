# segment-classify-track

This repository contains all the steps necessary for the annotation and tracking of the CellX project's time lapse microscopy data.

Each step has an associated jupyter notebook which should be run in the following sequential order:

1. Image alignment (align.ipynb)
2. Cellular segmentation (cellpose_segmentation.ipynb/stardist_segmentation.ipynb)
3. Phenotype classification (cellx_classify.ipynb)
4. Object tracking (btrack_tracking.ipynb)
5. Viewer (viewer.ipynb)

The repository is designed to work with the raw output of any timelapse microscopy data set that is structured with the following path structure:

`/expt_ID/PosID/img_channel000_position000_time000000000_z000.tif/`
                                                         
