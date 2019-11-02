lint:
	flake8 --show-source deepmatcher/
	isort --check-only -rc deepmatcher/ --diff

	flake8 --show-source tests/
	isort --check-only -rc tests/ --diff

	flake8 --show-source setup.py
	isort --check-only setup.py --diff

test:
	python -m unittest

install:
	pip install -r requirements/ci.txt
	pip install -e .
	pre-commit install
