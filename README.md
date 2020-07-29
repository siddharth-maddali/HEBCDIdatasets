# Phys Rev X Manuscript XP10416: Supplementary material

This repo contains the collected BCDI data and the processing scripts in the form of Jupyter notebooks.

## Data files
These are the BCDI data, after pre-processing, background reduction and removal of stray scattering. 
They are in the `.mat` format, and can be opened in Matlab, Octave or Pyhon (using the `scipy.io` module).

### File `111-A.mat`
BCDI data set corresponding to the $\left[111\right]$ reflection of the Bragg peak Friedel pair.

### File `111-B.mat` 
BCDI data set corresponding to the $\left[\bar{1}\bar{1}\bar{1}\right]$ reflection of the Bragg peak Friedel pair.

### File `200-D.mat` 
BCDI data set corresponding to the $\left[200\right]$ reflection of the second grain.

### File `stdSample.mat` 
BCDI data set corresponding to the $\left[111\right]$ reflection of the dewetted gold nanoparticle.

## Miscellaneous (`misc/*`)
Contains the calculated bases of sampling vectors for each reconstruction (real- and reciprocal-space), estimates of spatially varying strain, and isosurface data.


## Computation scripts

### Notebooks
Jupyter notebooks were used for the phase retrieval reconstructions and and a few subsequent computations for results seen in the manuscript. 
Dependencies include Tensorflow 1.5 for the GPU code for the reconstruction code.
The notebooks are named correspondingly to the data files described above.

### Python scripts
These were used for computations other than those included above.


