import deepmatcher as dm

train, validation, test = dm.process(
    path='~/emdata/structured_data_rm_long_attrs/Walmart-Amazon/',
    train='train_dm.csv',
    validation='valid_dm.csv',
    test='test_dm.csv',
    ignore_columns=('left_id', 'right_id'),
    include_lengths=False)

em_model = dm.MatchingModel()
em_model.train(train, validation)
