import os

from torchtext.vocab import FastText

test_dir_path = os.path.dirname(os.path.realpath(__file__))

embeddings = FastText()
