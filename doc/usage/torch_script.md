# Torch Script

We provide a simple script `./scripts/examples/silk-torch-script.py` to convert SiLK to a TorchScript model, saved to disk. The advantage of doing so is that the saved model can be run in any python environment that uses PyTorch. The disadvantage is that the provided parameters (threshold, top-k, ...) are frozen during the tracing and cannot be changed afterwards. Anytime a change of parameter is required, the tracing has to be re-run, and a new TorchScript model will be created.
