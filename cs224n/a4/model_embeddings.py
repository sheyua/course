#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from torch import device
from torch.nn import Module


class ModelEmbeddings(Module):
    """
    Class that converts input words to their embeddings.
    """
    @property
    def device(self) -> device: return device('cpu')

    @property
    def dtype(self) -> type:
        """

        """
        from torch import float32
        return float32

    def __init__(self, embed_size: int, vocab: 'Vocab') -> None:
        """
        Init the Embedding layers.

        @param embed_size: Embedding size (dimensionality)
        @param vocab: Vocabulary object containing src and tgt languages See vocab.py for documentation.
        """
        from torch.nn import Embedding

        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']
        # YOUR CODE HERE (~2 Lines)
        #     self.source (Embedding Layer for source language)
        #     self.target (Embedding Layer for target langauge)
        #
        # Note:
        #     1. `vocab` object contains two vocabularies:
        #            `vocab.src` for source
        #            `vocab.tgt` for target
        #     2. You can get the length of a specific vocabulary by running:
        #             `len(vocab.<specific_vocabulary>)`
        #     3. Remember to include the padding token for the specific vocabulary
        #        when creating your Embedding.
        #
        # Use the following docs to properly initialize these variables:
        #     Embedding Layer:
        #         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        v, e, i = len(vocab.src), self.embed_size, src_pad_token_idx
        self.source = Embedding(num_embeddings=v, embedding_dim=e, padding_idx=i, device=self.device, dtype=self.dtype)
        v, e, i = len(vocab.tgt), self.embed_size, tgt_pad_token_idx
        self.target = Embedding(num_embeddings=v, embedding_dim=e, padding_idx=i, device=self.device, dtype=self.dtype)
        # END YOUR CODE
