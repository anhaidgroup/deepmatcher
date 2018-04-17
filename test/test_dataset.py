from nose.tools import *

import os
import pandas as pd
import torch
import unittest
from deepmatcher.data.dataset import *
from deepmatcher.data.field import MatchingField

class ClassMatchingDatasetTestCases(unittest.TestCase):
    def test_init_1(self):
        fields = [('left_a', MatchingField()), ('right_a', MatchingField())]
        col_naming = {'id':'id', 'label':'label', 'left':'left',
                      'right':'right'}
        path = os.path.join('.', 'test_datasets', 'sample_table_small.csv')
        md = MatchingDataset(fields, col_naming, path=path)
        self.assertEqual(md.id_field, 'id')
        self.assertEqual(md.label_field, 'label')
        self.assertEqual(md.all_left_fields, ['left_a'])
        self.assertEqual(md.all_right_fields, ['right_a'])
        self.assertEqual(md.all_text_fields, ['left_a', 'right_a'])
        self.assertEqual(md.canonical_text_fields, ['_a'])

    def test_init_2(self):
        pass
