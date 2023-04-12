# Setup

## First Time Setup

```bash
make conda_env_init
conda activate silk
make dev_install
make conda_update
```

## How to push update/add a PIP dependency

1. Add the new dependency to `requirements.txt` (with version constraint).
1. Run `make dev_install` to re-install the package in dev mode.
1. Run `make conda_export` to update the conda environment (don't forget to commit it).

## How to push update/add a CONDA dependency

1. Run your conda command `conda install ...`.
1. Run `make conda_export` to update the conda environment (don't forget to commit it).

## How to update your existing environment with pulled dependencies ?

```bash
make conda_update
```

# Unit Tests

## How to run all unit tests

```bash
./bin/run_tests
```

## How to add a unit test

1. Add a file named `<module>_test.py` next to the module `<module>.py` that you want to test.
1. Implement the tests in that new file using the [unittest](https://docs.python.org/3/library/unittest.html) module.
1. Run your tests like you would any other python file (`python <path>/<module>_test.py`) to make sure it works well.
1. By adopting this naming convention, this new test file will be executed when running `./bin/run_tests`.

# Linter

Please run `./bin/linter` before committing your code.

# Code Structure

## Library

All common python code will be in `lib/` :
* Models should be in `lib/models`.
* Custom pytorch modules should be in `lib/layers`.
* Custom transforms (non-differentialble) should be in `lib/transforms`.
* Datasets should be in `lib/datasets`.
* Data Classes should be in `lib/data`.
* Serialization/File/Networking should be in `lib/io`.
* Training loop stuff (pytorch lightning) will be in `lib/train`.
* 3D environment specific code should be in `lib/env3d`.
* Visual Query specific code should be in `lib/visual_query`.

## Binaries

When implementing a binary command line, it should be in `bin/`.

Example : We might need to build a simple python tool that extract keypoints from a list of images.
This could become a python script like this `bin/extract_keypoints`, and this script would be using the `silk` library.

# Documentation

## What documentation format should we use ?

We do use [pdoc3](https://pdoc3.github.io/pdoc/) for automatically generating the documentation. `pdoc3` is compatible with several [formats](https://pdoc3.github.io/pdoc/doc/pdoc/#supported-docstring-formats) (markdown, numpydoc, Google-style docstrings, ...).

When commenting our code, we should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format as it is easy to read and fairly exhaustive.

## How to generate the documentation ?

```bash
make doc
```

# VS Code

## When debugging, I cannot get the debugger call stack when `./bin/silk-cli` crashes raising an exception.

That's because `silk-cli` catches all exceptions by default. To avoid that problem, run `silk-cli` with this additional argument : `debug=true`.

Example : `./bin/silk-cli [...] debug=true`
