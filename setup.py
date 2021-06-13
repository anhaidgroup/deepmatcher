import io
import os
import re

from setuptools import setup


def read(*names, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *names)) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('deepmatcher', '__init__.py')
long_description = read('README.rst')

# Deepmatcher lists "fasttextmirror" as a dependency because the official "fasttext"
# release on PyPI is out of date and has not been updated in over a year.
# "fasttextmirror" is a clone of the "fasttext" repository as of June 5 2018 and is hosted
# at https://github.com/sidharthms/fastText.
setup(
    name='deepmatcher',
    description='A deep learning package for entity matching',
    long_description=long_description,
    version=VERSION,
    author='Sidharth Mudgal, Han Li',
    author_email='uwmagellan@gmail.com',
    url='http://deepmatcher.ml',
    license='BSD',
    packages=['deepmatcher', 'deepmatcher.data', 'deepmatcher.models'],
    python_requires='>=3.5',
    install_requires=[
        'torch>=1.0', 'tqdm', 'pyprind', 'six', 'Cython', 'torchtext>=0.9',
        'nltk>=3.2.5', 'fasttext', 'pandas', 'dill', 'scikit-learn'
    ])
