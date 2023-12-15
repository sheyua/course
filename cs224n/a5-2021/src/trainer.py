"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

We suggest not changing anything in this file.
"""
from numpy import pi, cos
from typing import Optional
from logging import getLogger
from torch import set_grad_enabled, save
from torch.cuda import is_available, current_device
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from .model import GPT
from .config import TrainerConfig


logger = getLogger(__name__)


class Trainer(object):
    """

    """
    def __init__(self, model: GPT, train_dataset: Dataset, test_dataset: Optional[Dataset],
                 config: TrainerConfig) -> None:
        """

        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # take over whatever gpus are on the system
        if is_available():
            self.device = current_device()
            self.model = self.model.to(self.device)
        else:
            self.device = 'cpu'
        logger.info(f'trainer running on device {self.device}')

    def save_checkpoint(self) -> None:
        """

        """
        if self.config.checkpoint is not None:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            logger.info(f'saving {self.config.checkpoint}')
            save(model.state_dict(), self.config.checkpoint)

    def train(self) -> None:
        """

        """
        # create the optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        params_decay, params_nodecay = list(), list()
        for name, param in self.model.named_parameters():
            if any(value in name for value in no_decay):
                params_nodecay.append(param)
            else:
                params_decay.append(param)
        # parameter decay is a regularization technique to pull the weights to zero each epoch
        optim_groups = [
            {'params': params_decay, 'weight_decay': self.config.weight_decay},
            {'params': params_nodecay, 'weight_decay': 0.0},
        ]
        optimizer = AdamW(params=optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        loader = DataLoader(dataset=self.train_dataset, shuffle=False, pin_memory=False,
                            batch_size=self.config.batch_size, num_workers=self.config.num_worker)

        # counter used for learning rate decay
        num_tokens = 0
        self.model.train(mode=True)

        # run through epochs
        for epoch in range(self.config.max_epoch):
            batch_loss = list()
            for it, (x, y) in enumerate(loader):
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # compute loss
                logits, loss = self.model(inputs=x, target=y)
                # collapse all losses if they are scattered on multiple gpus
                loss = loss.mean()
                batch_loss.append(loss.item())
                # back-prop and update the parameters
                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    # number of tokens processed this step (i.e. label is not -100)
                    num_tokens = num_tokens + int((y >= 0).sum())
                    if num_tokens < self.config.warmup_token:
                        # linear warmup, it would speed up learning in most cases
                        lr_ratio = num_tokens / max(1, self.config.warmup_token)
                        lr_ratio = max(1., min(lr_ratio, 5.))
                    else:
                        # cosine learning rate decay
                        value = self.config.final_token - self.config.warmup_token
                        value = (num_tokens - self.config.warmup_token) / max(1, value)
                        lr_ratio = max(.5 * (1 + cos(pi * value)), .1)
                    lr = self.config.learning_rate * lr_ratio
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate
                # report progress
                message = f'epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}'
                logger.info(message)
            # whether to save this epoch
            self.save_checkpoint()
