import os
import shutil
import unittest

from deepmatcher.data.iterator import create_matching_splits
from deepmatcher.data.process import process
from tests import embeddings, test_dir_path


class ClassMatchingIteratorTestCases(unittest.TestCase):
    def test_splits_1(self):
        vectors_cache_dir = ".cache"
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        data_dir = os.path.join(test_dir_path, "test_datasets")
        train_path = "sample_table_large.csv"
        valid_path = "sample_table_large.csv"
        test_path = "sample_table_large.csv"
        cache_file = "cache.pth"
        cache_path = os.path.join(data_dir, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)

        datasets = process(
            data_dir,
            train=train_path,
            validation=valid_path,
            test=test_path,
            cache=cache_file,
            embeddings=embeddings,
            id_attr="_id",
            left_prefix="ltable_",
            right_prefix="rtable_",
            embeddings_cache_path="",
            pca=False,
        )

        splits = create_matching_splits(datasets, batch_size=16)
        self.assertEqual(splits[0].batch_size, 16)
        self.assertEqual(splits[1].batch_size, 16)
        self.assertEqual(splits[2].batch_size, 16)
        splits_sorted = create_matching_splits(
            datasets, batch_sizes=[16, 32, 64], sort_in_buckets=False
        )
        self.assertEqual(splits_sorted[0].batch_size, 16)
        self.assertEqual(splits_sorted[1].batch_size, 32)
        self.assertEqual(splits_sorted[2].batch_size, 64)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        if os.path.exists(cache_path):
            os.remove(cache_path)

    def test_create_batches_1(self):
        vectors_cache_dir = ".cache"
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        data_dir = os.path.join(test_dir_path, "test_datasets")
        train_path = "sample_table_large.csv"
        valid_path = "sample_table_large.csv"
        test_path = "sample_table_large.csv"
        cache_file = "cache.pth"
        cache_path = os.path.join(data_dir, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)

        datasets = process(
            data_dir,
            train=train_path,
            validation=valid_path,
            test=test_path,
            cache=cache_file,
            embeddings=embeddings,
            id_attr="_id",
            left_prefix="ltable_",
            right_prefix="rtable_",
            embeddings_cache_path="",
            pca=False,
        )

        splits = create_matching_splits(datasets, batch_size=16)
        batch_splits = [split.create_batches() for split in splits]
        assert batch_splits

        sorted_splits = create_matching_splits(
            datasets, batch_sizes=[16, 32, 64], sort_in_buckets=False
        )
        batch_sorted_splits = [
            sorted_split.create_batches() for sorted_split in sorted_splits
        ]
        assert batch_sorted_splits

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        if os.path.exists(cache_path):
            os.remove(cache_path)
