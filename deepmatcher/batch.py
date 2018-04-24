from collections import namedtuple

import torch

AttrTensor_ = namedtuple('AttrTensor', ['data', 'lengths', 'word_probs', 'pc'])


class AttrTensor(AttrTensor_):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 0:
            return super(AttrTensor, cls).__new__(cls, *args)
        else:
            name = kwargs['name']
            attr = kwargs['attr']
            train_info = kwargs['train_info']
            if isinstance(attr, tuple):
                data = attr[0]
                lengths = attr[1]
            else:
                data = attr
                lengths = None
            word_probs = None
            if 'word_probs' in train_info.metadata:
                raw_word_probs = train_info.metadata['word_probs'][name]
                word_probs = torch.Tensor(
                    [[raw_word_probs[w] for w in b] for b in data.data])
                if data.is_cuda:
                    word_probs = word_probs.cuda()
            pc = None
            if 'pc' in train_info.metadata:
                pc = torch.Tensor(train_info.metadata['pc'][name])
                if data.is_cuda:
                    pc = pc.cuda()
            return AttrTensor(data, lengths, word_probs, pc)

    @staticmethod
    def from_old_metadata(data, old_attrtensor):
        return AttrTensor(data, *old_attrtensor[1:])


class MatchingBatch(object):

    def __init__(self, input, train_info):
        copy_fields = train_info.all_text_fields
        for name in copy_fields:
            setattr(self, name,
                    AttrTensor(
                        name=name, attr=getattr(input, name),
                        train_info=train_info))
        for name in [train_info.label_field, train_info.id_field]:
            if name is not None and hasattr(input, name):
                setattr(self, name, getattr(input, name))
