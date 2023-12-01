#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from typing import List, Union, Tuple, Generator
import nltk
nltk.download('punkt')


SentType = Union[List[int], List[str]]
SentsType = List[SentType]
BatchType = Generator[Tuple[SentsType, SentsType], None, None]
PadTokenType = Union[int, str]


def pad_sents(sents: SentsType, pad_token: PadTokenType) -> SentsType:
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents: list of sentences, where each sentence is represented as a list of words
    @param pad_token: padding token
    @returns sents_padded: list of sentences where sentences shorter than the max length sentence are padded out with
                           the pad_token, such that each sentences in the batch now has equal length.
    """
    # sents_padded = []

    # YOUR CODE HERE (~6 Lines)
    size = max([len(sent) for sent in sents])
    sents_padded = [sent + [pad_token] * (size - len(sent)) for sent in sents]
    # END YOUR CODE

    return sents_padded


def read_corpus(file_path: str, source: str, vocab_size: int=2500) -> SentsType:
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path: path to file containing corpus
    @param source: "tgt" or "src" indicating whether text is of the source language or target language
    @param vocab_size: number of unique subwords in vocabulary when reading and tokenizing
    """
    from sentencepiece import SentencePieceProcessor

    data = []
    sp = SentencePieceProcessor()
    sp.load('{}.model'.format(source))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)
    return data


def autograder_read_corpus(file_path: str, source: str) -> SentsType:
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path: path to file containing corpus
    @param source: "tgt" or "src" indicating whether text is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data: List[Tuple[SentType, SentsType]], batch_size: int, shuffle: bool=False) -> BatchType:
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data: list of tuples containing source and target sentence
    @param batch_size: batch size
    @param shuffle: whether to randomly shuffle the dataset
    """
    from math import ceil
    from numpy import random

    batch_num = ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

