from setuptools import setup

setup(
    name='deepmatcher',
    description='A deep learning package for entity matching',
    version='0.0.1',
    packages=['deepmatcher', 'deepmatcher.data', 'deepmatcher.models'],
    install_requires=['torch>=0.3.1', 'tqdm', 'pyprind', 'six', 'Cython', 'torchtext', 'nltk>=3.2.5', 'fasttextgithub==0.1.1', 'pandas'],
    dependency_links=[
        'http://github.com/facebookresearch/fastText/tarball/master#egg=fasttextgithub-0.1.1'
    ])
