import os
import pdb
import tempfile

import deepmatcher as dm

train, validation, test = dm.process(
    path='~/emdata/structured_data_rm_long_attrs/Amazon-Google/',
    train='train_dm.csv',
    validation='valid_dm.csv',
    test='test_dm.csv',
    ignore_columns=('left_id', 'right_id'),
    include_lengths=True,
    auto_rebuild_cache=True,
    pca=True)

em_model = dm.MatchingModel(attr_summarizer='sif')

tmpdir = tempfile.mkdtemp()
save_path = os.path.join(tmpdir, 'wa')

unlabeled = dm.process(
    path='~/emdata/structured_data_rm_long_attrs/Amazon-Google/',
    unlabeled='valid_dm.csv',
    ignore_columns=('left_id', 'right_id'),
    include_lengths=True,
    pca=True)
raw_table = unlabeled.get_raw_table()

em_model.run_train(
    train,
    validation,
    label_smoothing=0.05,
    batch_size=16,
    epochs=1,
    best_save_path=save_path,
    pos_weight=1.3)
# em_model.load_state(save_path)
# em_model.run_eval(test)
predictions = em_model.run_prediction(validation)
pdb.set_trace()

os.remove(save_prefix + '_best.pth')
os.rmdir(tmpdir)
