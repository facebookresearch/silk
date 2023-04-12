# Evaluation

## HPatches

### In-framework model

If the model to evaluate is already in the SiLK framework, then the evaluation can be run using `./bin/silk-cli mode=run-hpatches-tests-silk`. Make sure the model specified in `etc/mode/run-hpatches-tests-silk.yaml` point to the right model you want to test.

### Out-of-framework model

If the model is outside the SiLK framework (and would be too much work to integrate), then we provide an alternative way. First of all, we provide several versions of the HPatches dataset (found in `./assets/datasets/cached-hpatches/`, for different sizes and color space) that are ready to be used using our `CachedDataset` class. Second, we provide one template and several examples scripts (`./scripts/eval/run-on-hpatches-*.py`) used to "augment" an input dataset with keypoints obtained by an out-of-framework model (e.g. DISK, R2D2, LoFTR).

Those scripts generate a new dataset that contains the original input HPatches, and the keypoints found by the respective models. Once the output dataset has been generated, create a new configuration file `./etc/test-cached-<model_name>.yaml` that points to the newly generated dataset. You also have to create a new configuration file `./etc/run-hpatches-tests-cached-<model_name>.yaml` (inspired from existing `./etc/run-hpatches-tests-cached-*.yaml` files) and run it using the command line `./bin/silk-cli mode=run-hpatches-tests-cached-<model_name>`.

## ScanNet

The ScanNet evaluation has been run using the [Unsupervised R&R](https://github.com/mbanani/unsupervisedRR) implementation. We provide our own [fork](https://github.com/gleize/unsupervisedRR) as a git submodule in this SiLK repository. The modification we've made are only to accomodate the running of SiLK, SuperPoint & LoFTR. The evaluation protocol itself has not been changed (for fair comparison).

To pull the ScanNet eval pipeline, please run
```bash
git submodule update --init
```
then follow the setup instructions from [Unsupervised R&R](https://github.com/mbanani/unsupervisedRR).

To run the evaluation pipeline (using VGG-4), use the following commands
```bash
cd external/URR
python evaluate.py --checkpoint "<silk_path>/assets/models/silk/analysis/alpha/pvgg-4.ckpt" --boost_alignment --img_dim 146 --encoder silk PCReg
```
