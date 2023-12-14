from numpy import inf, sqrt
from torch.nn.functional import softmax
from torch import Tensor, ones, tril, matmul
from torch.nn import Module, LayerNorm, Linear, Dropout, Sequential, GELU
from .config import GPTConfig


class CausalSelfAttention(Module):
    """
        a vanilla multi-head masked self-attention layer with a projection at the end.
        I believe I could have just used torch.nn.MultiheadAttention but their documentation
        is all but absent and code ugly so I don't trust it, rolling my own here.
    """
    def __init__(self, config: GPTConfig) -> None:
        """

        """
        super().__init__()
        assert config.embedding_dim % config.n_head == 0
        # key, query, value projections for all heads
        self.key = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        self.query = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        self.value = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        self.n_head = config.n_head
        # regularization
        self.attn_drop = Dropout(p=config.attn_pdrop, inplace=False)
        self.resid_drop = Dropout(p=config.resid_pdrop, inplace=False)
        # output projection
        self.proj = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        data = ones(config.block_size, config.block_size, dtype=int)
        data = tril(input=data, diagonal=0).reshape([1, 1, config.block_size, config.block_size])
        self.register_buffer(name='mask', tensor=data, persistent=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """

        """
        batch_size, block_size, hidden_size = inputs.shape
        head_dim = hidden_size // self.n_head
        if head_dim * self.n_head != hidden_size:
            raise ValueError('multi-head attention size mismatch')

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        key = self.key(input=inputs).view(batch_size, block_size, self.n_head, head_dim).transpose(dim0=1, dim1=2)
        query = self.query(input=inputs).view(batch_size, block_size, self.n_head, head_dim).transpose(dim0=1, dim1=2)
        value = self.value(input=inputs).view(batch_size, block_size, self.n_head, head_dim).transpose(dim0=1, dim1=2)

        # compute casual attention
        scale = 1 / sqrt(head_dim)
        attn = scale * matmul(query, key.transpose(dim0=2, dim1=3))
        attn = attn.masked_fill(mask=self.mask[:, :, :block_size, :block_size] == 0, value=-inf)
        attn = softmax(input=attn, dim=3)

        # compute output
        ans = matmul(attn, value).transpose(dim0=1, dim1=2).reshape([batch_size, block_size, hidden_size])
        ans = self.proj(input=ans)
        ans = self.resid_drop(input=ans)
        return ans


# class SynthesizerAttention(nn.Module):
#     """
#     Write your SynthesizerAttention below.
#     Hint: paste over the CausalSelfAttention above and modify it minimally.
#     """
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         # NEW learnable weights
#         self.w1 = nn.Linear(config.n_embd, config.n_embd)
#         self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
#                                            config.block_size-1))
#         self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
#         # value projection
#         self.value = nn.Linear(config.n_embd, config.n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(config.attn_pdrop)
#         self.resid_drop = nn.Dropout(config.resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(config.n_embd, config.n_embd)
#         # causal mask to ensure that attention is only applied to the left in
#         #     the input sequence
#         self.register_buffer("mask", torch.tril(
#             torch.ones(config.block_size, config.block_size)).view(
#             1, 1, config.block_size, config.block_size))
#         self.n_head = config.n_head
#         self.block_size = config.block_size
#
#         nn.init.uniform_(self.w2,-0.001,0.001)
#
#     def forward(self, x, layer_past=None):
#         # TODO [part g]: Write your SynthesizerAttention below.
#         #   Do not modify __init__().
#         # Hints:
#         #   - Paste over the CausalSelfAttention above and modify it minimally.
#         #   - Consider especially the parameters self.w1, self.w2 and self.b2.
#         #       How do these map to the matrices in the handout?
#
#         raise NotImplementedError


class Block(Module):
    """
        an unassuming Transformer block
    """
    def __init__(self, config: GPTConfig) -> None:
        """

        """
        super().__init__()
        self.ln1 = LayerNorm(normalized_shape=config.embedding_dim)
        if config.additive:
            raise NotImplementedError
        else:
            self.attn = CausalSelfAttention(config=config)
        # feed-forward
        out_features = config.embedding_dim * config.feed_forward_expand
        self.ln2 = LayerNorm(normalized_shape=config.embedding_dim)
        self.feed_forward = Sequential(
            Linear(in_features=config.embedding_dim, out_features=out_features, bias=True),
            GELU(),
            Linear(in_features=out_features, out_features=config.embedding_dim, bias=True),
            Dropout(p=config.feed_forward_pdrop, inplace=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """

        """
        ln1 = self.ln1(input=inputs)
        attn = self.attn(inputs=ln1)
        # residual connection
        x = inputs + attn
        ln2 = self.ln2(input=x)
        mlp = self.feed_forward(input=ln2)
        # residual connection
        ans = x + mlp
        return ans
