# SuperPoint Model

## Paper https://arxiv.org/abs/1712.07629

## Training

### Locally (using 2 GPUs).

1. Train MagicPoint model (see [here](MAGICPOINT.md)).
2. Save model checkpoint path to `etc/models/magicpoint-frozen-best.yaml`.
3. Homographically augment COCO dataset (phase 1)
```bash
./bin/silk-cli -m mode=cache-dataset-homographically-adapted-coco-phase-1 'datasets/coco@mode.loader.dataset=training,validation'
```
4. Save newly created datasets path to `etc/datasets/homographically-adapted-coco/{phase-1-training.yaml,phase-1-validation.yaml}`.
5. Train superpoint (phase 1)
```bash
./bin/silk-cli mode=train-superpoint-phase-1
```
6. Save model checkpoint path to `etc/models/superpoint-frozen-phase-1-best.yaml`.
7. Homographically augment COCO dataset (phase 2)
```bash
./bin/silk-cli -m mode=cache-dataset-homographically-adapted-coco-phase-2 'datasets/coco@mode.loader.dataset=training,validation'
```
8. Save newly created datasets path to `etc/datasets/homographically-adapted-coco/{phase-2-training.yaml,phase-2-validation.yaml}`.
9. Train superpoint (phase 2)
```bash
./bin/silk-cli mode=train-superpoint-phase-2
```
10. Save model checkpoint path to `etc/models/superpoint-frozen-phase-2-best.yaml`.

All results can be found in `var/silk-cli/run/training/<date>/<time>`.

### On SLURM (using 4 GPUs).

Take any above local command as `$CMD` and run

```bash
srun --gres=gpu:$NGPU --partition=pixar --time=9999 --pty --cpus-per-task $NCPU $CMD hydra=slurm
```

All results can be found in `/checkpoint/$USER/silk-cli/run/{training,cache_dataset}/<date>/<time>`.
