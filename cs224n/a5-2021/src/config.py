class GPTConfig(object):
    """

    """
    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, n_layer: int, n_head: int,
                 attention_type: str) -> None:
        """

        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.feed_forward_expand = 4
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.feed_forward_pdrop = 0.1
        self.additive = False
        self.attention_type = attention_type


class GPT1Config(GPTConfig):
    """

    """
    def __init__(self, vocab_size: int, block_size: int) -> None:
        """

        """
        super(GPT1Config, self).__init__(vocab_size=vocab_size, embedding_dim=768, block_size=block_size,
                                         n_layer=12, n_head=12, attention_type='causal')


class TrainerConfig(object):
    """
        optimization parameters
    """
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    # only applied on matmul weights
    weight_decay = 0.1
    # checkpoint settings
    checkpoint = None

    def __init__(self, max_epoch: int, batch_size: int, learning_rate: float, lr_decay: bool, warmup_token: int,
                 final_token: int, num_worker: int) -> None:
        """

        """
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.lr_decay = lr_decay
        self.warmup_token = warmup_token
        self.final_token = final_token
        self.num_worker = num_worker

