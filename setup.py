import io
import os
import re

from setuptools import setup


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('deepmatcher', '__init__.py')
long_description = read('README.md')

setup(
    name='deepmatcher',
    description='A deep learning package for entity matching',
    long_description=long_description,
    version=VERSION,
    author='Sidharth Mudgal, Han Li',
    url='http://bit.do/deepmatcher',
    license='BSD',
    packages=['deepmatcher', 'deepmatcher.data', 'deepmatcher.models'],
    install_requires=[
        'torch>=0.3.1', 'tqdm', 'pyprind', 'six', 'Cython', 'torchtext', 'nltk>=3.2.5',
        'fasttextgithub==0.1.1', 'pandas'
    ],
    dependency_links=[
        'http://github.com/facebookresearch/fastText/tarball/master#egg=fasttextgithub-0.1.1'
    ])
