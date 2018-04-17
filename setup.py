from setuptools import setup, find_packages

setup(
    name='deepmatcher',
    description='A deep learning package for entity matching',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['torch>=0.3.1', 'tqdm', 'pyprind', 'six', 'Cython', 'fasttext'],
    dependency_links=[
        'https://github.com/facebookresearch/tarball/master#fasttext-0.1.1'
    ])
