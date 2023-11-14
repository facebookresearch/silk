# COLMAP

We've developed a tool to easily import SiLK matches to [COLMAP](https://colmap.github.io/).

## How to run it ?

Our tool will read image sequences present in `assets/colmap/sequences/*`, and generate reconstructions in `var/colmap/*`.
We currently provide three sequences (from [Co3D](https://ai.meta.com/datasets/CO3D-dataset/)) as examples.

To start the reconstruction, simply run : `./bin/run_colmap_examples`

## How to visualize the results ?

We provide a jupyter notebook (`./notebook/viz-colmap.ipynb`) to read the COLMAP reconstruction and visualize the generated point cloud, as well as the predicted camera poses. Simply change the `tag` variable at the beginning of the notebook to select the reconstruction to visualize.

## How to run it on my own sequence ?

To run SiLK/COLMAP on your own sequence, please add your image sequence as a new folder to `assets/colmap/sequences/`.
Images should be chronologically ordered and have the `.jpg` extension.

Also, add your newly created folder name and camera intrinsics to `assets/colmap/sequences/intrinsics.txt` in order to be processed by `./bin/run_colmap_examples`.
Each line correspond to a reconstruction and has the following format : `<sequence_name> <fx,fy> <cx,cy>` where `<fx,fy>` are the focal parameters `<cx,cy>` are the principal point. 
