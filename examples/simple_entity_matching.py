import deepmatcher as dm
import tempfile
import os

train, validation, test = dm.process(
    path='~/emdata/structured_data_rm_long_attrs/Amazon-Google/',
    train='train_dm.csv',
    validation='valid_dm.csv',
    test='test_dm.csv',
    ignore_columns=('left_id', 'right_id'),
    include_lengths=True,
    auto_rebuild_cache=True)

em_model = dm.MatchingModel(attr_summarizer='hybrid')

tmpdir = tempfile.mkdtemp()
save_prefix = os.path.join(tmpdir, 'wa')

em_model.run_train(train, validation, label_smoothing=0.05, batch_size=16, epochs=10,
                   save_prefix=save_prefix, pos_weight=1.3)
em_model.load_state(save_prefix + '_best.pth')
em_model.run_eval(test)

os.remove(save_prefix + '_best.pth')
os.rmdir(tmpdir)
