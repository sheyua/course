from numpy import inf, sqrt
from torch import Tensor, ones, zeros, tril, matmul
from torch.nn.functional import softmax, relu
from torch.nn import Module, Parameter, LayerNorm, Linear, Dropout, Sequential, GELU
from torch.nn.init import uniform_
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
        attn = self.attn_drop(input=attn)
        attn = attn.masked_fill(mask=self.mask[:, :, :block_size, :block_size] == 0, value=-inf)
        attn = softmax(input=attn, dim=3)

        # compute output
        ans = matmul(attn, value).transpose(dim0=1, dim1=2).reshape([batch_size, block_size, hidden_size])
        ans = self.proj(input=ans)
        ans = self.resid_drop(input=ans)
        return ans


class SynthesizerAttention(Module):
    """
        Write your SynthesizerAttention below.
        Hint: paste over the CausalSelfAttention above and modify it minimally.
    """
    def __init__(self, config: GPTConfig) -> None:
        """

        """
        super().__init__()
        assert config.embedding_dim % config.n_head == 0
        head_dim = config.embedding_dim // config.n_head

        # new learnable weights
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.w1 = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        # self.w2 = Parameter(zeros(config.n_embd // config.n_head, config.block_size-1))
        self.w2 = Parameter(data=zeros(self.n_head, head_dim, self.block_size - 1), requires_grad=True)
        uniform_(tensor=self.w2, a=-1e-3, b=1e-3)
        # self.b2 = Parameter(zeros(config.block_size-1))
        self.b2 = Parameter(data=zeros(self.block_size - 1), requires_grad=True)
        # value projection
        self.value = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        # regularization
        self.attn_drop = Dropout(p=config.attn_pdrop, inplace=False)
        self.resid_drop = Dropout(p=config.resid_pdrop, inplace=False)
        # output projection
        self.proj = Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)
        # mask to ensure that attention is only applied to the left in the input sequence
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

        # compute attention value
        kernel = self.w1(input=inputs).view(batch_size, block_size, self.n_head, head_dim).transpose(dim0=1, dim1=2)
        kernel = relu(input=kernel, inplace=False)
        attn = matmul(kernel, self.w2) + self.b2
        attn = self.attn_drop(input=attn)
        attn = attn.masked_fill(mask=self.mask[:, :, :block_size, :(self.block_size - 1)] == 0, value=-inf)
        attn = softmax(input=attn[:, :, :, :block_size], dim=3)

        # compute output
        value = self.value(input=inputs).view(batch_size, block_size, self.n_head, head_dim).transpose(dim0=1, dim1=2)
        ans = matmul(attn, value).transpose(dim0=1, dim1=2).reshape([batch_size, block_size, hidden_size])
        ans = self.proj(input=ans)
        ans = self.resid_drop(input=ans)
        return ans


class Block(Module):
    """
        an unassuming Transformer block
    """
    def __init__(self, config: GPTConfig) -> None:
        """

        """
        super().__init__()
        self.ln1 = LayerNorm(normalized_shape=config.embedding_dim)
        if config.attention_type.lower() == 'causal':
            self.attn = CausalSelfAttention(config=config)
        elif config.attention_type.lower() == 'synthesizer':
            self.attn = SynthesizerAttention(config=config)
        else:
            raise NotImplementedError
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
