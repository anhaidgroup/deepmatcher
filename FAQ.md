# FAQ

### I get a `TypeError` with a stacktrace similar to [this](https://gist.github.com/sidharthms/2698c2c9fc072036371166c5972b48f9) when I try to train the model.

This is a known issue and will be fixed in the next release. As a workaround, you can simply change the name of the attribute causing the problem, e.g., you can change "left_type" and "right_type" to "left_entity_type" and "right_entity_type" respectively.

### What is `fasttextmirror` and why does `deepmatcher` list it as a dependency?

This is because the official `fasttext` [release on PyPI](https://pypi.org/project/fasttext/) is out of date and has not been updated in over a year. `fasttextmirror` is a [github fork](https://github.com/sidharthms/fastText) of the official `fasttext` [github repository](https://github.com/facebookresearch/fastText) as of June 5 2018.

### I do not see any output from `dm.data.process` after the very first time I run it.

This is because during the first time `dm.data.process` is called on a dataset, it builds a cache of the data. In subsequent calls, it simply reuses the cache so as to load faster.

### Why do I see a message saying "Loading best model..." at the end of training?

After each epoch of training `deepmatcher` computes the validation set accuracy of the latest model. If the validation accuracy is better than the best validation accuracy seen so far, the model is saved to disk. After some epochs the accuracy may stop improving and instead may start declining. So at the end of training, `deepmatcher` restores the best model from disk. This is done by modifying the model in-place.
