
.PHONY: doc

doc: doc/silk/index.html

doc/silk/index.html: $(shell find -L silk -type f -iname "*.py")
	pdoc --html -o doc --force silk

dev_install:
	pip install --upgrade pip
	pip install -e . -f https://download.pytorch.org/whl/cu113/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # `-f ...` required to find pytorch / jax packages
	python -c "import silk; print(silk.__name__)"

conda_env_init:
	conda create -n silk python==3.8.12

conda_export:
	conda env export > envs/dev.yml
	sed -i '/prefix: /d' envs/dev.yml # remove line with "prefix: /..."
	cat envs/dev.yml

conda_update:
	conda env update --name silk --file envs/dev.yml --prune

clean_pycaches:
	find . -type d -name __pycache__ -exec rm -rf {} +