.ONESHELL:
.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_DEACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda deactivate ; conda deactivate
DEV_ENV_NAME=rxt-dev


install: 
	@echo "Creating development environment for Rxitect..."
	mamba env create -f env-dev.yaml
	@echo "Installing Rxitect in editable mode..."
	$(CONDA_ACTIVATE) $(DEV_ENV_NAME)
	pip install -e .

env:
	@echo "Activating Rxitect's development environment..."
	$(CONDA_ACTIVATE) $(DEV_ENV_NAME)

delete_env:
	@echo "Deleting the current environment..."
	$(CONDA_DEACTIVATE) deactivate
	mamba env remove -n $(DEV_ENV_NAME)

reinstall_env:
	@echo "Reinstalling Rxitect environment..."
	delete_env
	install

data:
	@echo "Pulling data from DVC..."
	$(CONDA_ACTIVATE) $(DEV_ENV_NAME)
	dvc pull -r origin

test:
	pytest

view_docs:
	@echo Loading API documentation... 
	pdoc src --http localhost:8080

save_docs:
	@echo Saving documentation...
	rm -rf docs
	pdoc src -o docs

clean:
	@echo "Deleting all compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache