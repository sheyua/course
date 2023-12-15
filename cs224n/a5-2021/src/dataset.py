from typing import Tuple
from logging import getLogger
from torch import Tensor
from torch.utils.data import Dataset


"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

You don't need to implement anything in NameDataset.
"""

logger = getLogger(name=__name__)


class NameDataset(Dataset):
    """

    """
    def __init__(self, pretraining_dataset: Dataset, data: str) -> None:
        """

        """
        # the double question mark character, for mask
        self.MASK_CHAR = u"\u2047"
        # the empty square character, for pad
        self.PAD_CHAR = u"\u25A1"
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self) -> int: return len(self.data) - 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """

        """
        from torch import tensor, long

        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp) - 1) + x[len(inp):]
        x = tensor([self.stoi[c] for c in x[:-1]], dtype=long)
        y = tensor([self.stoi[c] for c in y], dtype=long)
        return x, y


class CharCorruptionDataset(Dataset):
    """

    """
    def __init__(self, data: str, block_size: int) -> None:
        """

        """
        # the double question mark character, for mask
        self.MASK_CHAR = u'\u2047'
        # the empty square character, for pad
        self.PAD_CHAR = u'\u25A1'

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        data_size, vocab_size = len(data), len(chars)
        self.block_size = block_size
        self.vocab_size = vocab_size
        logger.info(f'data has {data_size} characters, {self.vocab_size} unique.')
        self.data = data.split('\n')

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
            [part e]

            Write a class that yields examples of a simplified span corruption objective.
            Do not change the signature of the __init__ or __getitem__ functions.

            Make sure to implement the full spec for full credit -- we list below the
            criteria that must be satisfied for a full implementation.

            --------------
            Vocabulary Specification

            Your vocabulary is to be accessible via two dictionaries:
              self.stoi: a dictionary from characters in the vocabulary to indices of type
                  int
              self.itos: a dictionary from indices of type int to characters in the
                  vocabulary

            Your vocabulary must have the following form:

              Identifier 0 must be assigned to the unicode element u"\u25A1".
                  This is the empty_square_character.
                  Further, let self.PAD_CHAR = u"\u25A1"
              Identifier 1 must be assigned to the unicode element u"\u2047".
                  This is the doublequestionmark character, which we'll use
                  as a sentinel to represent that text is missing from the input
                  Further, let self.MASK_CHAR = u"\u2047"
              Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
                  that appear in the data argument.

            --------------
            Masking Specification

            The __getitem__ function takes an index and returns a data point (x, y) where
            x and y are Long tensors of length self.block_size. x encodes the input
            sequence, and y encodes the output sequence.

            0. Use the idx argument of __getitem__ to retrieve the element of self.data
            at the given index. We'll call the resulting data entry a document.

            1. Randomly truncate the document to a length no less than 4 characters,
            and no more than int(self.block_size*7/8) characters.

            - IMPORTANT: You are free to decide how to perform this random truncation, but
            make sure that the length is picked _randomly_ (every possible length from 4
            to int(self.block_size*7/8) has a chance of being picked) for full credit.

            2. Now, break the (truncated) document into three substrings:

                [prefix] [masked_content] [suffix]

              In other words, choose three strings prefix, masked_content and suffix
                such that prefix + masked_content + suffix = [the original (truncated?) document].
              The length of [masked_content] should be random, and 1/4 the length of the
                truncated document on average.

            - IMPORTANT: You are free to decide how to perform this operation, but
            make sure that the length is picked _randomly_ (has a chance of being more or
            less than 1/4 the length of the truncated document) for full credit.

            3. Rearrange these substrings into the following form:

                [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]

              This resulting string, denoted masked_string, serves as the output example.
              Here MASK_CHAR is the masking character defined in Vocabulary Specification,
                and [pads] is a string of repeated PAD_CHAR characters chosen so that the
                entire string is of length self.block_size.
              Intuitively, the [masked_content], a string, is removed from the document and
                replaced with MASK_CHAR (the masking character defined in Vocabulary
                Specification). After the suffix of the string, the MASK_CHAR is seen again,
                followed by the content that was removed, and the padding characters.

            4. We now use masked_string to construct the input and output example pair. To
            do so, simply take the input string to be masked_string[:-1], and the output
            string to be masked_string[1:]. In other words, for each character, the goal is
            to predict the next character in the masked string.

            5. Making use of the vocabulary that you defined, encode the resulting input
            and output strings as Long tensors and return the resulting data point.

            ----------------
            Here are some examples of input-output pairs (x, y):

              x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
              y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

              x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
              y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

              x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
              y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


        """
        from torch import tensor, long
        from numpy.random import randint

        # get item
        item = self.data[idx]
        # truncate to get the prefix
        low, high = 4, int(self.block_size * 7 / 8) + 1
        if low <= high <= self.block_size:
            pass
        else:
            raise ValueError(f'cannot truncate the sentence')
        # get item and use a truncation length
        item = item[:randint(low=low, high=high, size=None)]
        # get the size of the mask, random between 1 and size / 2, on average size / 4
        mask_size = randint(low=1, high=int(len(item) / 2), size=None)
        start_index = randint(low=1, high=len(item) - mask_size, size=None)
        end_index = start_index + mask_size
        prefix, mask, suffix = item[:start_index], item[start_index:end_index], item[end_index:]

        # rearrange
        item = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + mask
        num_pad = self.block_size - len(item)
        assert num_pad >= 0
        item = item + self.PAD_CHAR * num_pad

        # output
        x, y = item[:-1], item[1:]
        x = tensor([self.stoi[c] for c in x[:-1]], dtype=long)
        y = tensor([self.stoi[c] for c in y], dtype=long)
        return x, y


def main() -> None:
    """

    """
    from os.path import dirname, abspath
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument('dataset_type', help='type of dataset to sample from. options: namedata, charcorruption.',
                      choices=['namedata', 'charcorruption'])
    args = args.parse_args()
    location = abspath(f'{dirname(abspath(__file__))}/..')
    block_size = 128
    num_test = 4

    if args.dataset_type == 'namedata':
        # even if it hasn't been implemented, we use it to define the vocab
        with open(f'{location}/wiki.txt', 'r') as f:
            corruption_dataset = CharCorruptionDataset(data=f.read(), block_size=block_size)
        # make the name dataset
        with open(f'{location}/birth_places_train.tsv', 'r') as f:
            name_dataset = NameDataset(pretraining_dataset=corruption_dataset, data=f.read())

        for _, example in zip(range(num_test), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))

    elif args.dataset_type == 'charcorruption':
        with open(f'{location}/wiki.txt', 'r') as f:
            corruption_dataset = CharCorruptionDataset(data=f.read(), block_size=block_size)
        for _, example in zip(range(num_test), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(f'Unknown dataset type in command line args: {args.dataset_type}')


if __name__ == '__main__':
    main()
