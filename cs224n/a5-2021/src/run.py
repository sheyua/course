from argparse import ArgumentParser


def set_logger() -> None:
    """

    """
    from logging import basicConfig, INFO

    date_fmt = '%m/%d/%Y %H:%M:%S'
    handler_fmt = '%(asctime)s-%(levelname)s-%(name)s - %(message)s'
    basicConfig(format=handler_fmt, datefmt=date_fmt, level=INFO)


def make_parser() -> ArgumentParser:
    """

    """
    ans = ArgumentParser()
    choices = ['pretrain', 'finetune', 'evaluate']
    text = 'whether tpretrain, finetune or evaluate a model'
    ans.add_argument('--function', dest='function', help=text, choices=choices)
    choices = ['vanilla', 'synthesizer']
    text = 'which variant of the model to run (vanilla or synthesizer)'
    ans.add_argument('--variant', dest='variant', help=text, choices=choices)
    text = 'path of the corpus to pretrain on'
    ans.add_argument('--pretrain-corpus-path', dest='pretrain_corpus_path', help=text, default=None)
    text = 'if specified, path of the model to load before finetuning/evaluation'
    ans.add_argument('--reading-params-path', dest='reading_params_path', help=text, default=None)
    text = 'path to save the model after pretraining/finetuning'
    ans.add_argument('--writing-params-path', dest='writing_params_path', help=text, default=None)
    text = 'path of the corpus to finetune on'
    ans.add_argument('--finetune-corpus-path', dest='finetune_corpus_path', help=text, default=None)
    text = 'path of the corpus to evaluate on'
    ans.add_argument('--eval_corpus_path', help=text, default=None)
    ans.add_argument('--outputs_path', default=None)

    # keep the default arguments
    ans.add_argument('--block-size', dest='block_size', default=128, type=int, help='training block size')
    ans.add_argument('--batch-size', dest='batch_size', default=256, type=int, help='training batch size')
    ans.add_argument('--num-layer', dest='n_layer', default=4, type=int, help='number of layers')
    ans.add_argument('--num-head', dest='n_head', default=8, type=int, help='number of multi-head attention')
    ans.add_argument('--dim-embedding', dest='n_embd', default=256, type=int, help='embedding size')
    return ans


def main() -> None:
    """

    """
    from src.core import core

    set_logger()
    args = make_parser().parse_args()
    core(args=args)


if __name__ == '__main__':
    from sys import path
    from os.path import dirname, abspath

    location = dirname(abspath(__file__))
    path.append(abspath(f'{location}/..'))
    main()
