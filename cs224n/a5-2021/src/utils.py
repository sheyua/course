""" Utilities; we suggest changing none of these functions

but feel free to add your own.
"""
from typing import Optional, List, Tuple
from torch import Tensor
from .model import GPT


def set_seed(value: int) -> None:
    """

    """
    from random import seed
    from numpy import random
    from torch import manual_seed
    from torch.cuda import manual_seed_all

    seed(value)
    random.seed(value)
    manual_seed(value)
    manual_seed_all(value)


def top_k_logits(logits: Tensor, k: int) -> Tensor:
    """

    """
    from torch import topk
    v, ix = topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def sample(model: GPT, x: Tensor, steps: int, temperature: float=1.0, is_greedy: bool=True,
           top_k: Optional[int]=None) -> Tensor:
    """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
    """
    from torch import topk, multinomial, cat
    from torch.nn.functional import softmax

    batch_size, block_size = x.shape

    model.eval()
    for step in range(steps):
        if block_size <= model.block_size:
            x_cond = x
        else:
            x_cond = x[:, -model.block_size:]
        logits, _ = model(inputs=x_cond, target=None)
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits=logits, k=top_k)
        # apply softmax to convert to probabilities
        prob = softmax(logits, dim=1)
        # sample from the distribution or take the most likely
        if is_greedy:
            _, ix = topk(input=prob, k=1, dim=1)
        else:
            ix = multinomial(input=prob, num_samples=1)
        # append to the sequence and continue
        x = cat((x, ix), dim=1)

    return x


def evaluate_places(y_true: List[str], prediction: List[str]) -> Tuple[int, int]:
    """
      Computes percent of correctly predicted birth places.

      Returns: (total, correct), floats
    """
    total, correct = len(y_true), 0
    for y, p in zip(y_true, prediction):
        if y == p:
            correct += 1
    return total, correct
