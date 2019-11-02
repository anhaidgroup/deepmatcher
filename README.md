# DeepMatcher

[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![image](https://travis-ci.org/anhaidgroup/deepmatcher.svg?branch=master)](https://travis-ci.org/anhaidgroup/deepmatcher)
[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

DeepMatcher is a Python package for performing entity and text matching
using deep learning. It provides built-in neural networks and utilities
that enable you to train and apply state-of-the-art deep learning models
for entity matching in less than 10 lines of code. The models are also
easily customizable - the modular design allows any subcomponent to be
altered or swapped out for a custom implementation.

As an example, given labeled tuple pairs such as the
following:

![image](https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/docs/source/_static/match_input_ex.png)

DeepMatcher uses labeled tuple pairs and trains a neural network to
perform matching, i.e., to predict match / non-match labels. The trained
network can then be used to obtain labels for unlabeled tuple pairs.

## Paper and Data

For details on the architecture of the models used, take a look at our
paper [Deep Learning for Entity
Matching](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)
(SIGMOD '18). All public datasets used in the paper can be downloaded
from the [datasets page](Datasets.md).

## Quick Start: DeepMatcher in 30 seconds

There are four main steps in using DeepMatcher:

1.  Data processing: Load and process labeled training, validation and
    test CSV data.

<!-- end list -->

```python
import deepmatcher as dm
train, validation, test = dm.data.process(path='data_directory',
    train='train.csv', validation='validation.csv', test='test.csv')
```

2.  Model definition: Specify neural network architecture. Uses the
    built-in hybrid model (as discussed in section 4.4 of
    <span class="title-ref">our paper
    \<http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf\></span>\_\_)
    by default. Can be customized to your heart's desire.

<!-- end list -->

```python
model = dm.MatchingModel()
```

3.  Model training: Train neural network.

<!-- end list -->

```python
model.run_train(train, validation, best_save_path='best_model.pth')
```

4.  Application: Evaluate model on test set and apply to unlabeled data.

<!-- end list -->

```python
model.run_eval(test)

unlabeled = dm.data.process_unlabeled(path='data_directory/unlabeled.csv', trained_model=model)
model.run_prediction(unlabeled)
```

## Installation

We currently support only Python versions 3.5 and 3.6. Installing using
pip is recommended:

```bash
$ pip install deepmatcher
```

Note that during installation you may see an error message that says
"Failed building wheel for fasttextmirror". You can safely ignore this -
it does NOT mean that there are any problems with installation.

## Tutorials

**Using DeepMatcher:**

1.  [Getting
    Started](https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/getting_started.ipynb):
    A more in-depth guide to help you get familiar with the basics of
    using DeepMatcher.
2.  [Data
    Processing](https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/data_processing.ipynb):
    Advanced guide on what data processing involves and how to customize
    it.
3.  [Matching
    Models](https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/matching_models.ipynb):
    Advanced guide on neural network architecture for entity matching
    and how to customize it.

**Entity Matching Workflow:**

[End to End Entity
Matching](https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/end_to_end_em.ipynb):
A guide to develop a complete entity matching workflow. The tutorial
discusses how to use DeepMatcher with
[Magellan](https://sites.google.com/site/anhaidgroup/projects/magellan)
to perform blocking, sampling, labeling and matching to obtain matching
tuple pairs from two tables.

**DeepMatcher for other matching tasks:**

[Question Answering with
DeepMatcher](https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/question_answering.ipynb):
A tutorial on how to use DeepMatcher for question answering.
Specifically, we will look at
[WikiQA](https://aclweb.org/anthology/D15-1237), a benchmark dataset for
the task of Answer Selection.

## API Reference

API docs [are here](https://anhaidgroup.github.io/deepmatcher/html/).

## Support

Take a look at [the FAQ](FAQ.md) for common issues. If you run into any
issues or have questions not answered in the FAQ, please [file GitHub
issues](https://github.com/anhaidgroup/deepmatcher/issues) and we will
address them asap.

## The Team

DeepMatcher was developed by University of Wisconsin-Madison grad
students Sidharth Mudgal and Han Li, under the supervision of Prof.
AnHai Doan and Prof. Theodoros Rekatsinas.
