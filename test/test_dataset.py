from nose.tools import *

import os
import pandas as pd
import torch
import unittest
from deepmatcher.data.dataset import *
from deepmatcher.data.field import MatchingField
from deepmatcher.data.process import process

from test import test_dir_path

class ClassMatchingDatasetTestCases(unittest.TestCase):
    def test_init_1(self):
        fields = [('left_a', MatchingField()), ('right_a', MatchingField())]
        col_naming = {'id':'id', 'label':'label', 'left':'left',
                      'right':'right'}
        path = os.path.join(test_dir_path, 'test_datasets', 'sample_table_small.csv')
        md = MatchingDataset(fields, col_naming, path=path)
        self.assertEqual(md.id_field, 'id')
        self.assertEqual(md.label_field, 'label')
        self.assertEqual(md.all_left_fields, ['left_a'])
        self.assertEqual(md.all_right_fields, ['right_a'])
        self.assertEqual(md.all_text_fields, ['left_a', 'right_a'])
        self.assertEqual(md.canonical_text_fields, ['_a'])

class DataframeSplitTestCases(unittest.TestCase):
    def test_split_1(self):
        labeled_path = os.path.join(test_dir_path, 'test_datasets', 'sample_table_large.csv')
        labeled_table = pd.read_csv(labeled_path)
        ori_cols = list(labeled_table.columns)
        out_path = os.path.join(test_dir_path, 'test_datasets')
        train_prefix = 'train.csv'
        valid_prefix = 'valid.csv'
        test_prefix = 'test.csv'
        split(labeled_table, out_path, train_prefix, valid_prefix, test_prefix)

        train_path = os.path.join(out_path, train_prefix)
        valid_path = os.path.join(out_path, valid_prefix)
        test_path = os.path.join(out_path, test_prefix)

        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)

        self.assertEqual(list(train.columns), ori_cols)
        self.assertEqual(list(valid.columns), ori_cols)
        self.assertEqual(list(test.columns), ori_cols)

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(valid_path):
            os.remove(valid_path)
        if os.path.exists(test_path):
            os.remove(test_path)

    def test_split_2(self):
        labeled_path = os.path.join(test_dir_path, 'test_datasets', 'sample_table_large.csv')
        labeled_table = pd.read_csv(labeled_path)
        ori_cols = list(labeled_table.columns)
        out_path = os.path.join(test_dir_path, 'test_datasets')
        train_prefix = 'train.csv'
        valid_prefix = 'valid.csv'
        test_prefix = 'test.csv'
        split(labeled_path, out_path, train_prefix, valid_prefix, test_prefix)

        train_path = os.path.join(out_path, train_prefix)
        valid_path = os.path.join(out_path, valid_prefix)
        test_path = os.path.join(out_path, test_prefix)

        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)

        self.assertEqual(list(train.columns), ori_cols)
        self.assertEqual(list(valid.columns), ori_cols)
        self.assertEqual(list(test.columns), ori_cols)

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(valid_path):
            os.remove(valid_path)
        if os.path.exists(test_path):
            os.remove(test_path)

# class GetRawTableTestCases(unittest.TestCase):
#     def test_get_raw_table(self):
#         train = process(
#             path=os.path.join('.', 'test_datasets'),
#             train='sample_table_large.csv',
#             id_attr='_id',
#             left_prefix='ltable_',
#             right_prefix='rtable_',
#             ignore_columns=('ltable_id', 'rtable_id'))
#         train_raw = train.get_raw_table()
#         ori_train = pd.read_csv(os.path.join('.', 'test_datasets', 'sample_table_large.csv'))
#         self.assertEqual(list(train_raw.columns), list(ori_train.columns))
