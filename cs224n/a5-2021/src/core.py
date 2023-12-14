from argparse import Namespace


def core(args: Namespace) -> None:
    """

    """
    from .utils import set_seed
    from .dataset import CharCorruptionDataset, NameDataset
    from .config import GPTConfig, TrainerConfig
    from .model import GPT
    from .trainer import Trainer

    set_seed(value=0)
    # the pretraining corpus always required because we use it to have the same vocabulary, that is,
    # the same mapping from character to integer, and we build the vocab from the pretraining corpus.
    with open(args.pretrain_corpus_path, 'r') as f:
        pretrain_dataset = CharCorruptionDataset(data=f.read(), block_size=args.block_size)

    # the default hyper-parameters are known to work. use them for both the vanilla and the synthesizer models
    model_config = GPTConfig(vocab_size=pretrain_dataset.vocab_size, embedding_dim=args.n_embd,
                             block_size=pretrain_dataset.block_size, n_layer=args.n_layer, n_head=args.n_head)

    # part c and g
    if args.variant == 'vanilla':
        model = GPT(config=model_config)
    elif args.variant == 'synthesizer':
        # TODO [part g]: Make some other model here
        raise NotImplementedError
    else:
        raise NotImplementedError

    # From here on, your code should be identical independent of which
    # variant (vanilla or synthesizer) has been chosen.

    if args.function == 'pretrain':
        assert args.pretrain_corpus_path is not None
        assert args.writing_params_path is not None

        # TODO [part f]:
        # - Given:
        #     1. A corpus specified in args.pretrain_corpus_path
        #     2. An output path args.writing_params_path for the model parameters
        # - Goals:
        #     1. Pretrain the model on this corpus
        #     2. Save the resulting model in args.writing_params_path
        # - Make sure to use the following hyperparameters for pretraining:
        #     max_epochs=650
        #     batch_size=128
        #     learning_rate=6e-3
        #     lr_decay=True
        #     warmup_tokens=512*20
        #     final_tokens=200*len(pretrain_dataset)*block_size
        #     num_workers=4
        raise NotImplementedError

    elif args.function == 'finetune':

        from torch import load, save

        assert args.writing_params_path is not None
        assert args.finetune_corpus_path is not None

        # load fine tune data set
        with open(args.finetune_corpus_path, 'r') as f:
            train_dataset = NameDataset(pretraining_dataset=pretrain_dataset, data=f.read())

        # if model is saved load it up
        final_token = 200 * len(pretrain_dataset) * pretrain_dataset.block_size
        if isinstance(args.reading_params_path, str):
            model.load_state_dict(load(args.reading_params_path))
            # hyper-parameters for fine-tuning WITH a pretrained model:
            #  max_epochs = 10
            #  batch_size = 256
            #  learning_rate = 6e-4
            #  lr_decay = True
            #  warmup_tokens = 512 * 20
            #  final_tokens = 200 * len(pretrain_dataset) * block_size
            #  num_workers = 4
            trainer_config = TrainerConfig(max_epoch=10, batch_size=args.batch_size, learning_rate=6e-4, lr_decay=True,
                                           warmup_token=512 * 20, final_token=final_token, num_worker=4)
        else:
            # hyper-parameters for fine-tuning WITHOUT a pretrained model:
            #  max_epochs = 75
            #  batch_size = 256
            #  learning_rate = 6e-4
            #  lr_decay = True
            #  warmup_tokens = 512 * 20
            #  final_tokens = 200 * len(pretrain_dataset) * block_size
            #  num_workers = 4
            trainer_config = TrainerConfig(max_epoch=75, batch_size=args.batch_size, learning_rate=6e-4, lr_decay=True,
                                           warmup_token=512 * 20, final_token=final_token, num_worker=4)
        # make trainer
        trainer = Trainer(model=model, train_dataset=train_dataset, test_dataset=None, config=trainer_config)
        trainer.train()
        save(model.state_dict(), args.writing_params_path)
        import ipdb
        ipdb.set_trace()
        assert True

        # TODO [part c] [part f]:
        # - Given:
        #     1. A finetuning corpus specified in args.finetune_corpus_path
        #     2. A path args.reading_params_path containing pretrained model
        #         parameters, or None if finetuning without a pretrained model
        #     3. An output path args.writing_params_path for the model parameters
        # - Goals:
        #     1. If args.reading_params_path is specified, load these parameters
        #         into the model
        #     2. Finetune the model on this corpus
        #     3. Save the resulting model in args.writing_params_path

    elif args.function == 'evaluate':

        from torch.cuda import is_available, current_device

        # save the device
        if is_available():
            device = current_device()
        else:
            device = 'cpu'
            # keep the block size 128

        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        model.load_state_dict(torch.load(args.reading_params_path))
        correct = 0
        total = 0
        with open(args.outputs_path, 'w') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path)):
                x = line.split('\t')[0]
                x = x + '⁇'
                x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
                pred = utils.sample(model, x, 32, sample=False)[0]
                completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
            print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
        else:
            print('Predictions written to {}; no targets provided'
                  .format(args.outputs_path))

    else:
        raise NotImplementedError
