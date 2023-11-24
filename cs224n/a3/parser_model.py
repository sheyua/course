#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2020-2021: Homework 3
parser_model.py: Feed-Forward Neural Network for Dependency Parsing
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from numpy import ndarray
from torch import dtype, Tensor
from torch.nn import Module, Parameter


class ParserModel(Module):
    """ Feedforward neural network with an embedding layer and two hidden layers.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    PyTorch Notes:
        - Note that "ParserModel" is a subclass of the "nn.Module" class. In PyTorch all neural networks
            are a subclass of this "nn.Module".
        - The "__init__" method is where you define all the layers and parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix. Thus,
            you should add the "self." prefix layers, values, etc. that you want to utilize
            in other ParserModel methods.
        - For further documentation on "nn.Module" please see https://pytorch.org/docs/stable/nn.html.
    """
    @property
    def dtype(self) -> dtype: return self.embeddings.dtype

    def __init__(self, embeddings: ndarray, n_features: int=36, hidden_size: int=200, n_classes: int=3,
                 dropout_prob: float=0.5) -> None:
        """ Initialize the parser model.

        @param embeddings: word embeddings (num_words, embedding_size)
        @param n_features: number of input features
        @param hidden_size: number of hidden units
        @param n_classes: number of output classes
        @param dropout_prob: dropout probability
        """
        from numpy import zeros
        from torch import tensor
        from torch.nn import Dropout
        from torch.nn.init import xavier_uniform_, uniform_, calculate_gain

        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        _, self.embed_size = embeddings.shape
        self.hidden_size = hidden_size
        self.embeddings = Parameter(data=tensor(data=embeddings, requires_grad=False), requires_grad=False)

        # YOUR CODE HERE (~9-10 Lines)
        #     1) Declare `self.embed_to_hidden_weight` and `self.embed_to_hidden_bias` as `nn.Parameter`.
        #        Initialize weight with the `nn.init.xavier_uniform_` function and bias with `nn.init.uniform_`
        #        with default parameters.
        #     2) Construct `self.dropout` layer.
        #     3) Declare `self.hidden_to_logits_weight` and `self.hidden_to_logits_bias` as `nn.Parameter`.
        #        Initialize weight with the `nn.init.xavier_uniform_` function and bias with `nn.init.uniform_`
        #        with default parameters.
        #
        # W matrix
        shape = self.n_features * self.embed_size, self.hidden_size
        data = tensor(data=zeros(shape=shape), requires_grad=True, dtype=self.dtype)
        self.embed_to_hidden_weight = Parameter(data=data, requires_grad=True)
        # b1 bias
        data = tensor(data=zeros(shape=self.hidden_size), requires_grad=True, dtype=self.dtype)
        self.embed_to_hidden_bias = Parameter(data=data, requires_grad=True)
        # dropout
        self.dropout = Dropout(p=self.dropout_prob, inplace=False)
        # U matrix
        shape = self.hidden_size, self.n_classes
        data = tensor(data=zeros(shape=shape), requires_grad=True, dtype=self.dtype)
        self.hidden_to_logits_weight = Parameter(data=data, requires_grad=True)
        # b2 bias
        data = tensor(data=zeros(shape=self.n_classes), requires_grad=True, dtype=self.dtype)
        self.hidden_to_logits_bias = Parameter(data=data, requires_grad=True)
        # init
        gain = calculate_gain(nonlinearity='relu')
        xavier_uniform_(self.embed_to_hidden_weight, gain=gain)
        uniform_(self.embed_to_hidden_bias)
        xavier_uniform_(self.hidden_to_logits_weight, gain=1.)
        uniform_(self.hidden_to_logits_bias)
        # END YOUR CODE

    def embedding_lookup(self, w: Tensor) -> Tensor:
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w: input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        # YOUR CODE HERE (~1-4 Lines)
        #     1) For each index `i` in `w`, select `i`th vector from self.embeddings
        #     2) Reshape the tensor using `view` function if necessary
        x = self.embeddings[w, :].reshape([w.size(0), -1])
        # END YOUR CODE
        return x

    def forward(self, w: Tensor) -> Tensor:
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss
            function nn.CrossEntropyLoss

            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `w` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `w` as follows,
                    the `forward` function would called on `w` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(w) # this calls the forward function
                - For more details checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w: input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        # YOUR CODE HERE (~3-5 lines)
        #     Complete the forward computation as described in write-up. In addition, include a dropout layer
        #     as decleared in `__init__` after ReLU function.
        #
        # Note: We do not apply the softmax to the logits here, because
        # the loss function (torch.nn.CrossEntropyLoss) applies it more efficiently.
        from torch import matmul
        from torch.nn.functional import relu

        x = self.embedding_lookup(w=w)
        h = relu(input=matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias, inplace=False)
        h_drop = self.dropout(h)
        logits = matmul(h_drop, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
        # END YOUR CODE
        return logits


def main() -> None:
    """

    """
    import argparse
    from torch import randint, long
    from numpy import zeros, float32, all

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
    parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
    args = parser.parse_args()

    embeddings = zeros((100, 30), dtype=float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = randint(0, 100, (4, 36), dtype=long)
        selected = model.embedding_lookup(inds)
        assert all(selected.data.numpy() == 0), f'The result of embedding lookup {selected} contains non-zero elements.'

    def check_forward():
        inputs = randint(0, 100, (4, 36), dtype=long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")


if __name__ == "__main__":
    main()
