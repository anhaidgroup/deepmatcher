lint:
	# flake8 --show-source deepmatcher/
	isort --check-only -rc deepmatcher/ --diff

	# flake8 --show-source test/
	isort --check-only -rc test/ --diff

	flake8 --show-source setup.py
	isort --check-only setup.py --diff

test:
	python -m nosetests -v

install:
	pip install -r requirements/ci.txt
	pip install -e .
	pre-commit install
