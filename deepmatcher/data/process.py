import io
import os
import pdb

import torchtext
from torchtext import data
from torchtext.utils import unicode_csv_reader

from . import torchtext_extensions as text


def _check_header(header, id_attr, left_prefix, right_prefix, label_attr):
    # assert id_attr in header
    assert label_attr in header

    for attr in header:
        if attr not in (id_attr, label_attr):
            assert attr.startswith(left_prefix) or attr.startswith(right_prefix)

    num_left = sum(attr.startswith(left_prefix) for attr in header)
    num_right = sum(attr.startswith(right_prefix) for attr in header)
    assert num_left == num_right


def _make_fields(header, id_attr, label_attr, ignore_columns, lower, include_lengths):
    text_field = text.MatchingField(
        lower=lower,
        init_token='<<<',
        eos_token='>>>',
        batch_first=True,
        include_lengths=include_lengths)
    numeric_field = text.MatchingField(
        sequential=False, preprocessing=lambda x: int(x), use_vocab=False)
    id_field = text.MatchingField(sequential=False, use_vocab=False, id=True)

    fields = []
    for attr in header:
        if attr == id_attr:
            fields.append((attr, id_field))
        elif attr == label_attr:
            fields.append((attr, numeric_field))
        elif attr in ignore_columns:
            fields.append((attr, None))
        else:
            fields.append((attr, text_field))
    return fields


def process(path,
            train=None,
            validation=None,
            test=None,
            unlabeled=None,
            cache='cacheddata.pth',
            check_cached_data=True,
            auto_rebuild_cache=False,
            shuffle_style='bucket',
            lowercase=True,
            embeddings='fasttext.en.bin',
            embeddings_cache_path='~/.vector_cache',
            ignore_columns=(),
            include_lengths=True,
            id_attr='id',
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            pca=False):

    a_dataset = train or validation or test or unlabeled
    with io.open(os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    _check_header(header, id_attr, left_prefix, right_prefix, label_attr)
    fields = _make_fields(header, id_attr, label_attr, ignore_columns, lowercase,
                          include_lengths)

    column_naming = {
        'id': id_attr,
        'left': left_prefix,
        'right': right_prefix,
        'label': label_attr if not unlabeled else None
    }

    return text.MatchingDataset.splits(
        path,
        train,
        validation,
        test,
        unlabeled,
        fields,
        embeddings,
        embeddings_cache_path,
        column_naming,
        cache,
        check_cached_data,
        auto_rebuild_cache,
        train_pca=pca)
