from collections import namedtuple

import torch

AttrTensor_ = namedtuple('AttrTensor', ['data', 'lengths', 'word_probs', 'pc'])


class AttrTensor(AttrTensor_):
    """A wrapper around the batch tensor for a specific attribute.

    The purpose of having a wrapper around the tensor is to include attribute specific
    metadata along with it. Metadata include the following:

    * ``lengths``: Lengths of each sequence (attribute value) in the batch.
    * ``word_probs``: For each sequence in the batch, a list of word probabilities
      corresponding to words in the sequence.
    * ``pc``: The first principal component of the sequence embeddings for all values of
      this attribute. For details on how this is computed refer documentation for
      :meth:`~deepmatcher.data.MatchingDataset.compute_metadata`. This is used for
      implementing the SIF model proposed in
      `this paper <https://openreview.net/pdf?id=SyK00v5xx>`__.

    This class is essentially a :class:`namedtuple`. The tensor containing the data and
    the associated metadata described above can be accessed as follows::

        name_attr = AttrTensor(data, lengths, word_probs, pc)
        assert(name_attr.data == data)
        assert(name_attr.lengths == lengths)
        assert(name_attr.word_probs == word_probs)
        assert(name_attr.pc == pc)
    """

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
                    [[raw_word_probs[int(w)] for w in b] for b in data.data])
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
        """Wrap a PyTorch :class:`torch.Tensor` into an :class:`AttrTensor`.

        The metadata information is (shallow) copied from a pre-existing
        :class:`AttrTensor`. This is useful when the data for an attribute is
        transformed by a neural network and we wish the wrap the result into an
        :class:`AttrTensor` for further processing by another module that requires
        access to metadata.

        Args:
            old_attrtensor (:class:`AttrTensor`):
                The pre-existing :class:`AttrTensor` to copy metadata from.
        """
        return AttrTensor(data, *old_attrtensor[1:])


class MatchingBatch(object):
    """A batch of data and associated metadata for a text matching task.

    Consists of one :class:`AttrTensor` (containing the data and metadata) for each
    attribute. For example, the :class:`AttrTensor` s of a :class:`MatchingBatch` object
    ``mbatch`` for a matching task with two attribtues ``name`` and ``category``, can be
    accessed as follows::

        name_attr = mbatch.name
        category_attr = mbatch.category
    """

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
