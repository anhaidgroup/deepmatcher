import deepmatcher as dm
import torch

train, validation, test = dm.process(
    path='~/emdata/structured_data_rm_long_attrs/Amazon-Google/',
    train='train_dm.csv',
    validation='valid_dm.csv',
    test='test_dm.csv',
    ignore_columns=('left_id', 'right_id'),
    include_lengths=True,
    auto_rebuild_cache=True,
    pca=True)

def my_attr_comparator(left, right):
    return torch.cat((left,
                      right,
                      torch.abs(left - left),
                      left * right), left.dim() - 1)

custom_model = dm.MatchingModel(
    attr_summarizer=dm.attr_summarizers.Hybrid(
        word_contextualizer=dm.word_contextualizers.RNN(unit_type='LSTM',
                                                        layers=2,
                                                        dropout=0.2,
                                                        bypass_network='highway'),
        word_comparator=dm.word_comparators.Attention(
            input_dropout=0.3,
            alignment_network=dm.modules.AlignmentNetwork(
                    style='concat_dot',
                    transform_network='2-layer-residual-tanh'),
            comparison_network=dm.modules.Transform('3-layer-highway',
                                                    hidden_size=400,
                                                    output_size=200)),
        word_aggregator=dm.word_aggregators.AttentionWithRNN(
            input_dropout=0.2,
            alignment_network='decomposable',
            rnn=dm.modules.RNN(unit_type='GRU',
                               layers=3,
                               dropout=0.1,
                               bypass_network='residual'),
            rnn_pool_style='max')),
    attr_comparator=lambda: dm.modules.Lambda(my_attr_comparator),
    classifier=dm.Classifier(transform_network='1-layer-residual-glu',
                             hidden_size=300),
    finetune_embeddings=True)

custom_model.run_train(
    train,
    validation,
    label_smoothing=0.05,
    batch_size=16,
    epochs=10,
    best_save_path='/tmp/custom_model.pth',
    pos_weight=1.3)
custom_model.load_state('/tmp/custom_model.pth')
custom_model.run_eval(test)
