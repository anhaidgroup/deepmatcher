from setuptools import setup

setup(
    name='deepmatcher',
    description='A deep learning package for entity matching',
    version='0.0.1',
    packages=['deepmatcher'],
    install_requires=['torch>=0.3.1', 'tqdm', 'six', 'fasttext', 'spacy'],
    dependency_links=[
        'https://github.com/facebookresearch/tarball/master#fasttext-0.1.1'
    ])
