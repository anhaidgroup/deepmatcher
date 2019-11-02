lint:
	isort --check-only -rc deepmatcher/ --diff
	isort --check-only -rc test/ --diff
	isort --check-only setup.py --diff

test:
	python -m nosetests -v

install:
	pip install -r requirements/ci.txt
	pip install -e .
	pre-commit install
