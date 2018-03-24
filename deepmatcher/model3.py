import dm

### Simple use cases
# Data loading / preprocessing
train, validation, test = dm.preprocess(
    path='/home/user/xyz',
    train='train.csv',
    validation='validation.csv',
    test='test.csv')

# Simplest model creation (defaults to Hybrid)
em_model = dm.MatchingModel()

# Picking attr summarizer v1:
em_model1 = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.SIF())
em_model2 = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.RNN())
em_model3 = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Attention())
em_model4 = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid())

# Picking attr summarizer v2:
em_model1 = dm.MatchingModel(attr_summarizer='sif')
em_model2 = dm.MatchingModel(attr_summarizer='rnn')
em_model3 = dm.MatchingModel(attr_summarizer='attention')
em_model4 = dm.MatchingModel(attr_summarizer='hybrid')

em_model.train(train, validation)

scores = em_model.evaluate(test)
print(scores)

unlabeled = dm.preprocess(unlabeled='unlabeled.csv')
predictions = em_model.predict(unlabeled)

preprocessed = unlabeled.get_raw_table()
for row in preprocessed.rows:
    print(row, 'Match prob = ', predictions[row['__id']])

### Detailed use cases
# Detailed data loading / preprocessing:
train, valid, test = dm.preprocess(
    path='/home/user/xyz',
    train='train.csv',
    validation='validation.csv',
    test='test.csv',

    ### Other default args:
    cache='cacheddata.pth',
    check_cached_data=True,  # Disable cache content verification
    # Automatically rebuild cache if data files or other preprocess
    # configs change.
    auto_rebuild_cache=False,
    column_spec=dm.data.Field(lower=False, tokenize='moses'),
    # Whether to randomly shuffle ('random'), sort by length ('sort'),
    # or to sort within randomly shuffled buckets ('bucket').
    shuffle_style='bucket')

# Detailed MatchingModel:
em_model = dm.MatchingModel(
    attr_summarizer=dm.attr_summarizers.Hybrid(hidden_size=1024),
    attr_comparator='concat_diff',
    classifier=dm.nn.NonLinearTransform(),
    loss=dm.nn.Loss(pos_weight=1.2))

# Detailed MatchingModel custom modules:
em_model = dm.MatchingModel(
    attr_summarizer=lambda s: my_attr_summarizer(s),
    attr_comparator=lambda s: my_attr_comparator(s),
    classifier=lambda s: my_fancy_classifier(s),
    loss=nn.NLLLoss(weight=[10, 0.1], weigtht_average=False))

# Detailed Hybrid Attr Summarizer:
hybrid_attr_summarizer = dm.attr_summarizers.Hybrid(
    word_contextualizer=dm.word_contextualizers.RNN(lstm=True, layers=3),
    word_comparator=dm.word_comparators.Attention(raw_alignment=True),
    word_aggregator=dm.word_aggregators.AttentionWithRNN(rnn_hidden_size=128))

# Detailed Attention Word Comparator:
attention_word_comparator = dm.word_comparators.Attention(
    heads=2,
    raw_alignment=False,
    alignment_network=dm.nn.AlignmentNetwork(type='decomposable'),
    value_transform_network=dm.nn.NonLinearTransform(),
    comparison_merge='concat',
    comparison_network=dm.nn.NonLinearTransform())

# Detailed Attention Word Comparator custom modules:
attention_word_comparator = dm.word_comparators.Attention(
    alignment_network=lambda s: my_alignment_network(s),
    value_transform_network=lambda s: my_transform_network(s),
    comparison_merge=lambda s: my_comparison_merge(s),
    comparison_network=lambda s: my_comparison_network(s))
