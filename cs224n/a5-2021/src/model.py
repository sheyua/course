
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from logging import getLogger
from typing import Optional, Tuple
from torch import Tensor, zeros
from torch.nn.functional import cross_entropy
from torch.nn import Module, Embedding, Dropout, Parameter, Sequential, LayerNorm, Linear, GELU
from .config import GPTConfig
from .attention import Block


logger = getLogger(name=__name__)


class GPT(Module):
    """
        the full GPT language model, with a context size of block_size
    """
    @property
    def num_parameter(self) -> int: return sum(value.numel() for value in self.parameters())

    @property
    def init_std(self) -> float: return .02

    def __init__(self, config: GPTConfig) -> None:
        """

        """
        super().__init__()
        self.block_size = config.block_size

        # input embedding stem
        self.token_embedding = Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        size = [1, config.block_size, config.embedding_dim]
        self.position_embedding = Parameter(data=zeros(*size))
        self.dropout = Dropout(p=config.embd_pdrop, inplace=False)

        # transformer
        self.blocks = Sequential(*[Block(config=config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = LayerNorm(normalized_shape=config.embedding_dim)
        self.head = Linear(in_features=config.embedding_dim, out_features=config.vocab_size, bias=False)
        self.apply(self.__init_weights__)
        logger.info(f'number of parameters on the GPT model is {self.num_parameter}')

    def __init_weights__(self, module: Module) -> None:
        """

        """
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(value=1.0)
        else:
            pass
    #
    # def get_block_size(self):
    #     return self.block_size

    def forward(self, inputs: Tensor, target: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        """

        """
        batch_size, block_size = inputs.shape
        if block_size > self.block_size:
            raise ValueError('cannot forward, model block size is exhausted.')
        # forward the GPT model
        # each index maps to a (learnable) vector
        token_embedding = self.token_embedding(input=inputs)
        # each position maps to a (learnable) vector
        position_embedding = self.position_embedding[:, :block_size, :]
        x = self.dropout(input=token_embedding + position_embedding)

        # feed to attention layers
        x = self.blocks(input=x)
        x = self.ln_f(input=x)
        logits = self.head(input=x)

        # if we are given some desired targets also calculate the loss
        if target is not None:
            total_size = batch_size * block_size
            loss = cross_entropy(input=logits.view(total_size, -1), target=target.view(total_size), ignore_index=0)
        else:
            loss = None
        return logits, loss
