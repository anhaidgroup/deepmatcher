import copy
from collections import Mapping

import deepmatcher as dm
import torch
import torch.nn as nn

from . import _utils


class MatchingModel(nn.Module):

    def __init__(self,
                 hidden_size=300,
                 attr_summarizer='average',
                 attr_comparator='concat-diff',
                 attr_merge='concat',
                 classifier='3-layer',
                 siamese=True,
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
        self.attr_summarizer = attr_summarizer
        self.attr_comparator = attr_comparator
        self.attr_merge = attr_merge
        self.classifier = classifier

        self.hidden_size = hidden_size
        self.siamese = siamese
        self.finetune_embeddings = finetune_embeddings
        self._initialized = False

    def set_metadata(self, train_dataset):
        self.dataset = train_dataset
        self.text_fields = self.dataset.text_fields
        self.all_left_fields = self.dataset.all_left_fields
        self.all_right_fields = self.dataset.all_right_fields
        self.corresponding_field = self.dataset.corresponding_field
        self.canonical_text_fields = self.dataset.canonical_text_fields

        self.attr_summarizers = dm.modules.ModuleMap()
        if isinstance(self.attr_summarizer, Mapping):
            for name, summarizer in self.attr_summarizer.items():
                if name in self.text_fields:
                    self.attr_summarizers[name] = summarizer
                    assert self.corresponding_field[name] in self.attr_summarizer
                elif name in self.canonical_text_fields:
                    left, right = self.text_fields[name]
                    assert left not in self.attr_summarizer
                    assert right not in self.attr_summarizer
                    self.attr_summarizers[left] = summarizer
                    if self.siamese:
                        self.attr_summarizers[right] = summarizer
                    else:
                        self.attr_summarizers[right] = copy.deepcopy(summarizer)
        else:
            self.attr_summarizer = _utils.get_attr_summarizer(
                self.attr_summarizer, hidden_size=self.hidden_size)

            for name in self.all_left_fields:
                self.attr_summarizers[name] = copy.deepcopy(self.attr_summarizer)
                corr_name = self.corresponding_field[name]
                if self.siamese:
                    self.attr_summarizers[corr_name] = self.attr_summarizers[name]
                else:
                    self.attr_summarizers[corr_name] = copy.deepcopy(self.attr_summarizer)

        self.attr_comparators = dm.modules.ModuleMap()
        if isinstance(self.attr_comparator, Mapping):
            for name, comparator in self.attr_comparator.items():
                self.attr_comparators[name] = comparator
        else:
            self.attr_comparator = _utils.get_attr_comparator(self.attr_comparator)
            for name in self.all_text_fields:
                self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        self.attr_merge = _utils.get_merge_module(self.attr_merge)
        self.classifier = _utils.get_transform_module(
            self.classifier, hidden_size=self.hidden_size, output_size=2)

        self.embed = dm.modules.ModuleMap()
        field_embeds = {}
        for name, field in train_dataset.fields.items():
            if field not in field_embeds:
                embed = nn.Embedding()
                embed.weight.data.copy_(field.vocab.vectors)
                embed.weight.requires_grad = self.finetune_embed
                field_embeds[field] = embed
            self.embed[name] = field_embeds[field]
        print('len field_embeds', len(field_embeds))

        self._initialized = True

    def forward(self, input):
        embeddings = {}
        for name in self.text_fields:
            embeddings[name] = self.embed[name](getattr(input, name))

        attr_summaries = {}
        for name in self.text_fields:
            corr_name = self.corresponding_field[name]
            attr_summaries[name] = self.attr_summarizers[name](embeddings[name],
                                                               embeddings[corr_name])

        attr_comparisons = []
        for name in self.canonical_text_fields:
            left, right = self.text_fields[name]
            attr_comparisons.append(self.attr_comparators[name](attr_summaries[left],
                                                                attr_summaries[right]))

        entity_comparison = self.attr_merge(attr_comparisons)
        return self.classifier(entity_comparison)
