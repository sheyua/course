#!/bin/bash

location=$(dirname $0)

if [ "$1" = "train" ]; then
  python $location/run.py train --train-src=$location/zh_en_data/train.zh --train-tgt=$location/zh_en_data/train.en --dev-src=$location/zh_en_data/dev.zh --dev-tgt=$location/zh_en_data/dev.en --vocab=$location/outputs/vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3 --save-to=$location/outputs/model.bin
elif [ "$1" = "train_local" ]; then
  python $location/run.py train --train-src=$location/zh_en_data/train.zh --train-tgt=$location/zh_en_data/train.en --dev-src=$location/zh_en_data/dev.zh --dev-tgt=$location/zh_en_data/dev.en --vocab=$location/outputs/vocab.json --lr=5e-4 --save-to=$location/outputs/model.local.bin
elif [ "$1" = "train_debug" ]; then
	python $location/run.py train --train-src=$location/zh_en_data/train_debug.zh --train-tgt=$location/zh_en_data/train_debug.en --dev-src=$location/zh_en_data/dev.zh --dev-tgt=$location/zh_en_data/dev.en --vocab=$location/outputs/vocab.json --lr=5e-4 --save-to=$location/outputs/model.debug.bin
elif [ "$1" = "test" ]; then
  python3 $location/run.py decode $location/outputs/model.bin $location/zh_en_data/test.zh $location/zh_en_data/test.en $location/outputs/test_outputs.txt --cuda
elif [ "$1" = "test_local" ]; then
  python $location/run.py decode $location/outputs/model.bin $location/zh_en_data/test.zh $location/zh_en_data/test.en $location/outputs/test_outputs.txt
elif [ "$1" = "dev" ]; then
  python3 $location/run.py decode $location/outputs/model.bin $location/zh_en_data/dev.zh $location/zh_en_data/dev.en $location/outputs/dev_outputs.txt --cuda
elif [ "$1" = "vocab" ]; then
	python $location/vocab.py --train-src=$location/zh_en_data/train.zh --train-tgt=$location/zh_en_data/train.en $location/outputs/vocab.json
else
	echo "Invalid Option Selected"
fi
