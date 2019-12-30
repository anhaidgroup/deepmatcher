from nose.tools import *

import io
import os
import shutil
import pandas as pd
import torch
import unittest

from deepmatcher import attr_summarizers
from deepmatcher.data.field import MatchingField, FastText
from deepmatcher.data.process import process, process_unlabeled
from deepmatcher import MatchingModel
from urllib.parse import urljoin
from urllib.request import pathname2url

from test import test_dir_path

class ModelTrainSaveLoadTest(unittest.TestCase):
    def setUp(self):
        self.vectors_cache_dir = '.cache'
        if os.path.exists(self.vectors_cache_dir):
            shutil.rmtree(self.vectors_cache_dir)

        self.data_cache_path = os.path.join(test_dir_path, 'test_datasets',
            'train_cache.pth')
        if os.path.exists(self.data_cache_path):
            os.remove(self.data_cache_path)

        vec_dir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(vec_dir)) + os.path.sep
        ft = FastText(filename, url_base=url_base, cache=self.vectors_cache_dir)

        self.train, self.valid, self.test = process(
            path=os.path.join(test_dir_path, 'test_datasets'),
            cache='train_cache.pth',
            train='test_train.csv',
            validation='test_valid.csv',
            test='test_test.csv',
            embeddings=ft,
            embeddings_cache_path='',
            ignore_columns=('left_id', 'right_id'))

    def tearDown(self):
        if os.path.exists(self.data_cache_path):
            os.remove(self.data_cache_path)

        if os.path.exists(self.vectors_cache_dir):
            shutil.rmtree(self.vectors_cache_dir)

    def test_sif(self):
        model_save_path = 'sif_model.pth'
        model = MatchingModel(attr_summarizer='sif')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)
        s1 = model.run_eval(self.test)

        model2 = MatchingModel(attr_summarizer='sif')
        model2.load_state(model_save_path)
        s2 = model2.run_eval(self.test)

        self.assertEqual(s1, s2)

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_rnn(self):
        model_save_path = 'rnn_model.pth'
        model = MatchingModel(attr_summarizer='rnn')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)
        s1 = model.run_eval(self.test)

        model2 = MatchingModel(attr_summarizer='rnn')
        model2.load_state(model_save_path)
        s2 = model2.run_eval(self.test)

        self.assertEqual(s1, s2)

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_attention(self):
        model_save_path = 'attention_model.pth'
        model = MatchingModel(attr_summarizer='attention')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        s1 = model.run_eval(self.test)

        model2 = MatchingModel(attr_summarizer='attention')
        model2.load_state(model_save_path)
        s2 = model2.run_eval(self.test)

        self.assertEqual(s1, s2)

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_hybrid(self):
        model_save_path = 'hybrid_model.pth'
        model = MatchingModel(attr_summarizer='hybrid')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        s1 = model.run_eval(self.test)

        model2 = MatchingModel(attr_summarizer='hybrid')
        model2.load_state(model_save_path)
        s2 = model2.run_eval(self.test)

        self.assertEqual(s1, s2)

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_hybrid_self_attention(self):
        model_save_path = 'self_att_hybrid_model.pth'
        model = MatchingModel(
            attr_summarizer=attr_summarizers.Hybrid(
                word_contextualizer='self-attention'))

        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        s1 = model.run_eval(self.test)

        model2 = MatchingModel(
            attr_summarizer=attr_summarizers.Hybrid(
                word_contextualizer='self-attention'))
        model2.load_state(model_save_path)
        s2 = model2.run_eval(self.test)

        self.assertEqual(s1, s2)

        if os.path.exists(model_save_path):
            os.remove(model_save_path)


class ModelPredictUnlabeledTest(unittest.TestCase):
    def setUp(self):
        self.vectors_cache_dir = '.cache'
        if os.path.exists(self.vectors_cache_dir):
            shutil.rmtree(self.vectors_cache_dir)

        self.data_cache_path = os.path.join(test_dir_path, 'test_datasets',
            'train_cache.pth')
        if os.path.exists(self.data_cache_path):
            os.remove(self.data_cache_path)

        vec_dir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(vec_dir)) + os.path.sep
        ft = FastText(filename, url_base=url_base, cache=self.vectors_cache_dir)

        self.train, self.valid, self.test = process(
            path=os.path.join(test_dir_path, 'test_datasets'),
            cache='train_cache.pth',
            train='test_train.csv',
            validation='test_valid.csv',
            test='test_test.csv',
            embeddings=ft,
            embeddings_cache_path='',
            ignore_columns=('left_id', 'right_id'))

    def tearDown(self):
        if os.path.exists(self.data_cache_path):
            os.remove(self.data_cache_path)

        if os.path.exists(self.vectors_cache_dir):
            shutil.rmtree(self.vectors_cache_dir)

    def test_sif(self):
        model_save_path = 'sif_model.pth'
        model = MatchingModel(attr_summarizer='sif')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        unlabeled = process_unlabeled(
            path=os.path.join(test_dir_path, 'test_datasets', 'test_unlabeled.csv'),
            trained_model=model,
            ignore_columns=('left_id', 'right_id'))

        pred_test = model.run_eval(self.test, return_predictions=True)
        pred_unlabeled = model.run_prediction(unlabeled)

        self.assertEqual(sorted([tup[1] for tup in pred_test]),
                         sorted(list(pred_unlabeled['match_score'])))

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_rnn(self):
        model_save_path = 'rnn_model.pth'
        model = MatchingModel(attr_summarizer='rnn')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        unlabeled = process_unlabeled(
            path=os.path.join(test_dir_path, 'test_datasets', 'test_test.csv'),
            trained_model=model,
            ignore_columns=('left_id', 'right_id', 'label'))

        pred_test = model.run_eval(self.test, return_predictions=True)
        pred_unlabeled = model.run_prediction(unlabeled)

        self.assertEqual(sorted([tup[1] for tup in pred_test]),
                         sorted(list(pred_unlabeled['match_score'])))

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_attention(self):
        model_save_path = 'attention_model.pth'
        model = MatchingModel(attr_summarizer='attention')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        unlabeled = process_unlabeled(
            path=os.path.join(test_dir_path, 'test_datasets', 'test_unlabeled.csv'),
            trained_model=model,
            ignore_columns=('left_id', 'right_id'))

        pred_test = model.run_eval(self.test, return_predictions=True)
        pred_unlabeled = model.run_prediction(unlabeled)

        self.assertEqual(sorted([tup[1] for tup in pred_test]),
                         sorted(list(pred_unlabeled['match_score'])))

        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def test_hybrid(self):
        model_save_path = 'hybrid_model.pth'
        model = MatchingModel(attr_summarizer='hybrid')
        model.run_train(
            self.train,
            self.valid,
            epochs=1,
            batch_size=8,
            best_save_path= model_save_path,
            pos_neg_ratio=3)

        unlabeled = process_unlabeled(
            path=os.path.join(test_dir_path, 'test_datasets', 'test_unlabeled.csv'),
            trained_model=model,
            ignore_columns=('left_id', 'right_id'))

        pred_test = model.run_eval(self.test, return_predictions=True)
        pred_unlabeled = model.run_prediction(unlabeled)

        self.assertEqual(sorted([tup[1] for tup in pred_test]),
                         sorted(list(pred_unlabeled['match_score'])))

        if os.path.exists(model_save_path):
            os.remove(model_save_path)
