import copy
from collections import Iterable, Mapping

import pdb
import six

import deepmatcher as dm
import torch
import torch.nn as nn

from . import _utils
from ..runner import Runner
from ..data import AttrTensor

class MatchingModel(nn.Module):

    def __init__(self,
                 attr_summarizer='hybrid',
                 attr_comparator='concat-diff',
                 attr_merge='concat',
                 classifier='2-layer-highway',
                 hidden_size=300,
                 finetune_embeddings=False):
        """
        Create a Hybrid Entity Matching Model (see arxiv.org/...)

        Args:
        attr_summarizer (dm.AttrSummarizer):
            The neural network to summarize a sequence of words into a
            vector. Defaults to `dm.attr_summarizers.Hybrid()`.
        attr_comparator (dm.AttrComparator):
            The neural network to compare two attribute summary vectors.
            Default is selected based on `attr_summarizer` choice.
        classifier (dm.Classifier):
            The neural network to perform match / mismatch classification
            based on attribute similarity representations.
            Defaults to `dm.Classifier()`.
        """
        super(MatchingModel, self).__init__()

        self.attr_summarizer = attr_summarizer
        self.attr_comparator = attr_comparator
        self.attr_merge = attr_merge
        self.classifier = classifier

        self.hidden_size = hidden_size
        self.finetune_embeddings = finetune_embeddings
        self._train_buffers = set()
        self._initialized = False

    def initialize(self, train_dataset):
        if self._initialized:
            return
        self.train_dataset = train_dataset
        self.text_fields = train_dataset.text_fields
        self.all_text_fields = train_dataset.all_text_fields
        self.all_left_fields = train_dataset.all_left_fields
        self.all_right_fields = train_dataset.all_right_fields
        self.corresponding_field = train_dataset.corresponding_field
        self.canonical_text_fields = train_dataset.canonical_text_fields

        self.attr_summarizers = dm.modules.ModuleMap()
        if isinstance(self.attr_summarizer, Mapping):
            for name, summarizer in self.attr_summarizer.items():
                self.attr_summarizers[name] = AttrSummarizer.create(
                    summarizer, hidden_size=self.hidden_size)
            assert len(
                set(self.attr_summarizers.keys()) ^ set(self.canonical_text_fields)) == 0
        else:
            self.attr_summarizer = AttrSummarizer.create(
                self.attr_summarizer, hidden_size=self.hidden_size)
            for name in self.canonical_text_fields:
                self.attr_summarizers[name] = copy.deepcopy(self.attr_summarizer)

        self.attr_comparators = dm.modules.ModuleMap()
        if isinstance(self.attr_comparator, Mapping):
            for name, comparator in self.attr_comparator.items():
                self.attr_comparators[name] = AttrComparator.create(comparator)
            assert len(
                set(self.attr_comparators.keys()) ^ set(self.canonical_text_fields)) == 0
        else:
            self.attr_comparator = AttrComparator.create(self.attr_comparator)
            for name in self.canonical_text_fields:
                self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        self.attr_merge = dm.modules._merge_module(self.attr_merge)
        self.classifier = dm.modules._transform_module(
            self.classifier, hidden_size=self.hidden_size, output_size=2)

        self.embed = dm.modules.ModuleMap()
        field_embeds = {}
        for name in self.all_text_fields:
            field = train_dataset.fields[name]
            if field not in field_embeds:
                vectors_size = field.vocab.vectors.shape
                embed = nn.Embedding(vectors_size[0], vectors_size[1])
                embed.weight.data.copy_(field.vocab.vectors)
                embed.weight.requires_grad = self.finetune_embeddings
                field_embeds[field] = embed
            self.embed[name] = field_embeds[field]

        self._initialized = True

    def forward(self, input):
        embeddings = {}
        for name in self.all_text_fields:
            attr_input = getattr(input, name)
            embedded = self.embed[name](attr_input.data)
            embeddings[name] = AttrTensor.from_old_metadata(embedded, attr_input)

        attr_comparisons = []
        for name in self.canonical_text_fields:
            left, right = self.text_fields[name]
            left_summary, right_summary = self.attr_summarizers[name](embeddings[left],
                                                                      embeddings[right])
            attr_comparisons.append(self.attr_comparators[name](left_summary.data,
                                                                right_summary.data))

        entity_comparison = self.attr_merge(*attr_comparisons)
        return self.classifier(entity_comparison)

    def register_train_buffer(self, name, value=None):
        self._train_buffers.add(name)
        setattr(self, name, value)

    def save_state(self, path):
        state = {'model': self.state_dict()}
        for k in self._train_buffers:
            state[k] = getattr(self, k)
        torch.save(state, path)

    def load_state(self, path):
        state = torch.load(path)
        for k, v in six.iteritems(state):
            if k != 'model':
                self._train_buffers.add(k)
                setattr(self, k, v)
        self.load_state_dict(state['model'])

    def train(self, *args, **kwargs):
        Runner.train(self, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        Runner.evaluate(self, *args, **kwargs)

    def train_mode(self):
        super(MatchingModel, self).train()

    def eval_mode(self):
        super(MatchingModel, self).eval()


class AttrSummarizer(dm.modules.LazyModule):

    def _init(self,
              word_contextualizer,
              word_comparator,
              word_aggregator,
              hidden_size=None):
        self.word_contextualizer = WordContextualizer.create(word_contextualizer,
                                                             hidden_size)
        self.word_comparator = WordComparator.create(word_comparator, hidden_size)
        self.word_aggregator = WordAggregator.create(word_aggregator, hidden_size)

    def _forward(self, left_input, right_input):
        pdb.set_trace()
        left_contextualized = self.word_contextualizer(left_input)
        right_contextualized = self.word_contextualizer(right_input)

        left_compared = self.word_comparator(left_contextualized, right_contextualized)
        right_compared = self.word_comparator(right_contextualized, left_contextualized)

        left_aggregated = self.word_aggregator(left_compared)
        right_aggregated = self.word_aggregator(right_compared)

        return left_aggregated, right_aggregated

    @classmethod
    def create(cls, arg, hidden_size=None):
        assert arg is not None
        if isinstance(arg, six.string_types):
            type_map = {
                'sif': dm.attr_summarizers.SIF(),
                'rnn': dm.attr_summarizers.RNN(),
                'attention': dm.attr_summarizers.Attention(),
                'hybrid': dm.attr_summarizers.Hybrid()
            }
            if arg in type_map.keys():
                asr = type_map[arg]
            else:
                raise ValueError('Unknown Attribute Summarizer name.')
        else:
            asr = _utils.get_module(AttrSummarizer, arg)

        asr.expect_signature('[AxBxC, AxDxC] -> [AxE, AxE]')
        return asr


class AttrComparator(dm.modules.LazyModule):

    @classmethod
    def create(cls, arg):
        assert arg is not None
        module = dm.modules._merge_module(arg)
        module.expect_signature('[AxB, AxB] -> [AxC]')
        return module


class WordContextualizer(dm.modules.LazyModule):

    @classmethod
    def create(cls, arg, hidden_size=None):
        wc = dm.modules.Identity()
        if isinstance(arg, six.string_types):
            if arg in dm.word_contextualizers.RNN.supported_styles:
                wc = dm.word_contextualizers.RNN(arg)
            elif arg == 'selfattention':
                wc = dm.word_contextualizers.SelfAttention()
        else:
            wc = _utils.get_module(WordContextualizer, arg, hidden_size=hidden_size)
        wc.expect_signature('[AxBxC] -> [AxBxD]')
        return wc


class WordComparator(dm.modules.LazyModule):

    @classmethod
    def create(cls, arg, hidden_size=None):
        wc = dm.modules.Identity()
        if isinstance(arg, six.string_types):
            parts = arg.split('-')
            if (parts[1] == 'attention' and
                    parts[0] in dm.modules.AlignmentNetwork.supported_styles):
                wc = dm.word_comparators.Attention(
                    alignment_network=parts[0], hidden_size=hidden_size)
            else:
                raise ValueError('Unknown Word Comparator name.')
        else:
            wc = _utils.get_module(WordComparator, arg)

        if wc is not None:
            wc.expect_signature('[AxBxC, AxDxC] -> [AxBxE]')

        return wc


class WordAggregator(dm.modules.LazyModule):

    @classmethod
    def create(cls, arg, hidden_size=None):
        assert arg is not None
        if isinstance(arg, six.string_types):
            parts = arg.split('-')
            if (parts[-1] == 'pool' and
                    '-'.join(parts[0:-1]) in dm.word_aggregators.Pool.supported_styles):
                wa = dm.word_aggregators.Pool(style=parts[0])
            elif ('-'.join(parts[-3:]) == 'attention-with-rnn' and
                  '-'.join(parts[0:-3]) in dm.modules.AlignmentNetwork.supported_styles):
                wa = dm.word_aggregators.AttentionWithRNN(
                    alignment_network=parts[0], hidden_size=hidden_size)
            else:
                raise ValueError('Unknown Word Aggregator name.')
        else:
            wa = _utils.get_module(WordAggregator, arg)

        wa.expect_signature('[AxBxC] -> [AxD]')
        return wa
