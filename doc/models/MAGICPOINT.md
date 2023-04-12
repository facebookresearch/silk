# MagicPoint Model

## Paper https://arxiv.org/abs/1712.07629

## Training

### Locally (using 2 GPUs).

```bash
./bin/silk-cli mode=train-magicpoint mode.trainer.gpus=2
```

The results will be found in `var/silk-cli/run/training/<date>/<time>`.

### On SLURM (using 4 GPUs).

```bash
srun --gres=gpu:4 --partition=pixar --time=9999 --pty --cpus-per-task 48 ./bin/silk-cli mode=train-magicpoint mode.trainer.gpus=4 hydra=slurm
```

The results will be found in `/checkpoint/$USER/silk-cli/run/training/<date>/<time>`.

## Benchmarking

### Locally

```bash
./bin/silk-cli mode=benchmark-magicpoint mode.model.checkpoint_path=<path_to_ckpt_file>
```

You might have to put the last argument under quotes if the path contains uncommon characters (e.g. `'mode.model.checkpoint_path="/weird/path/param=2,lr=0.1/checkpoint.ckpt"'`).
