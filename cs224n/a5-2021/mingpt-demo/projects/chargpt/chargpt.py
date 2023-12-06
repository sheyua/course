from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset


def set_logger() -> None:
    """

    """
    from logging import basicConfig, INFO

    date_fmt = '%m/%d/%Y %H:%M:%S'
    handler_fmt = '%(asctime)s-%(levelname)s-%(name)s - %(message)s'
    basicConfig(format=handler_fmt, datefmt=date_fmt, level=INFO)


def batch_end_callback(trainer: 'Trainer') -> None:
    """

    """
    from logging import getLogger

    if trainer.iter_num % 100 == 0:
        iter_dt = trainer.iter_dt * 1000
        logger = getLogger(__file__)
        logger.info(f'iter_dt {iter_dt:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}')


class CharDataset(Dataset):
    """

    """
    def __init__(self, data: str, block_size: int) -> None:
        """

        """
        from logging import getLogger

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.logger = getLogger(name=__file__)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.logger.info(f'data has {data_size} characters, {self.vocab_size} unique.')

    def __len__(self) -> int: return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """

        """
        from torch import long, tensor

        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = tensor(dix[:-1], dtype=long)
        y = tensor(dix[1:], dtype=long)
        return x, y


def run() -> None:
    """

    """
    from os.path import exists, dirname, abspath
    from torch import save, load, long, tensor
    from mingpt.model import GPT
    from mingpt.utils import sample
    from mingpt.trainer import Trainer

    num_workers = 4
    # spatial extent of the model for its context
    block_size = 128
    # don't worry we won't run out of file handles
    text = open(f'{dirname(abspath(__file__))}/input.txt', 'r').read()
    # one line of poem is roughly 50 characters
    train_dataset = CharDataset(data=text, block_size=block_size)

    # init model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = train_dataset.vocab_size
    model_config.block_size = train_dataset.block_size
    model = GPT(config=model_config)

    # trainer
    # the model we're using is so small that we can go a bit faster
    train_config = Trainer.get_default_config()
    train_config.batch_size = 512
    train_config.learning_rate = 6e-4
    train_config.num_workers = num_workers
    trainer = Trainer(config=train_config, model=model, train_dataset=train_dataset)
    trainer.set_callback(onevent='on_batch_end', callback=batch_end_callback)
    filename = f'{dirname(abspath(__file__))}/model.bin'
    if not exists(filename):
        trainer.run()
        save(model.state_dict(), filename)
    else:
        model.load_state_dict(load(filename))

    model.eval()
    context = 'O God, O God!'
    x = tensor([train_dataset.stoi[s] for s in context], dtype=long)[None, ...].to(trainer.device)
    y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)


def main() -> None:
    """

    """
    from sys import path
    path.append('../..')
    from mingpt.utils import set_seed

    set_seed(3407)
    set_logger()
    run()


if __name__ == '__main__':
    main()
