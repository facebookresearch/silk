# Setup

The following will create a new environment and install all dependencies required for SiLK.

```bash
make conda_env_init
conda activate silk
make dev_install
make conda_update
```

Then, assets (cached datasets, pre-trained model, ...) have to be downloaded using
```bash
./bin/public-pull-assets
```

Asset file which are not required for download can be removed from the `assets/public-assets.txt` file before running the above command.

The assets are structured as such.
* `assets/datasets` contains datasets metadata, and cached hpatches datasets.
* `assets/models` contains the model checkpoints used for all experiments mentioned in the paper.
    * `assets/models/silk/analysis/alpha/pvgg-4.ckpt` is the default model weights.
    * `assets/models/silk/analysis/*/` folders correspond to different result tables from the paper.
    * `assets/models/superpoint/*.ckpt` contains our reproduced SuperPoint model weights.
* `assets/results` contains json files of the results of every experiments mentioned in the paper.
* `assets/tests` are files required to run the unit tests and check procedures.


Once the environement is ready, please run the following to test if things works fine.

```bash
./bin/run_tests
./bin/run_check_procedures
```
