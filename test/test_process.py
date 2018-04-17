from nose.tools import *

import io
import os
import unittest
from torchtext.utils import unicode_csv_reader
from deepmatcher.data.process import _check_header, _make_fields, process

class CheckHeaderTestCases(unittest.TestCase):
    def test_check_header_1(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_small.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'left_a', 'right_a', 'label'])
        id_attr= 'id'
        label_attr = 'label'
        left_prefix = 'left'
        right_prefix = 'right'
        _check_header(header, id_attr, left_prefix,right_prefix, label_attr, [])

    @raises(AssertionError)
    def test_check_header_2(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_small.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'left_a', 'right_a', 'label'])
        id_attr= 'id'
        label_attr = 'label'
        left_prefix = 'left'
        right_prefix = 'bb'
        _check_header(header, id_attr, left_prefix,right_prefix, label_attr, [])

    @raises(AssertionError)
    def test_check_header_3(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_small.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'left_a', 'right_a', 'label'])
        id_attr= 'id'
        label_attr = 'label'
        left_prefix = 'aa'
        right_prefix = 'right'
        _check_header(header, id_attr, left_prefix,right_prefix, label_attr, [])

    @raises(AssertionError)
    def test_check_header_5(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_small.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'left_a', 'right_a', 'label'])
        id_attr= 'id'
        label_attr = ''
        left_prefix = 'left'
        right_prefix = 'right'
        _check_header(header, id_attr, left_prefix,right_prefix, label_attr, [])

    @raises(AssertionError)
    def test_check_header_6(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_small.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'left_a', 'right_a', 'label'])
        header.pop(1)
        id_attr= 'id'
        label_attr = 'label'
        left_prefix = 'left'
        right_prefix = 'right'
        _check_header(header, id_attr, left_prefix,right_prefix, label_attr, [])


class MakeFieldsTestCases(unittest.TestCase):
    def test_make_fields_1(self):
        path = os.path.join('.', 'test_datasets')
        a_dataset = 'sample_table_large.csv'
        with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))
        self.assertEqual(header, ['id', 'label', 'left_id', 'left_title',
                                  'left_manufacturer', 'left_price', 'right_id',
                                  'right_title', 'right_manufacturer', 'right_price'])
        id_attr= 'id'
        label_attr = 'label'
        fields = _make_fields(header, id_attr, label_attr,
                              ['left_price', 'right_price'], True, True)
        self.assertEqual(len(fields), 10)
        counter = {}
        for tup in fields:
            if tup[1] not in counter:
                counter[tup[1]] = 0
            counter[tup[1]] += 1
        self.assertEqual(sorted(list(counter.values())), [1, 1, 2, 6])

class ProcessTestCases(unittest.TestCase):
    def test_process_1(self):
        data_dir = os.path.join('.', 'test_datasets')
        train_path = 'sample_table_large.csv'
        valid_path = 'sample_table_large.csv'
        test_path = 'sample_table_large.csv'
        pass
