#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/pride_and_prejudice

mkdir -p $data/pride_and_prejudice/raw
curl -L https://www.gutenberg.org/files/1342/1342-0.txt -o $data/pride_and_prejudice/raw/tales.txt


# preprocess slightly

 cp $data/pride_and_prejudice/raw/tales.txt $data/pride_and_prejudice/raw/tales.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/pride_and_prejudice/raw/tales.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/pride_and_prejudice/raw/tales.preprocessed.txt

# split into train, valid and test

head -n 440 $data/pride_and_prejudice/raw/tales.preprocessed.txt | tail -n 400 > $data/pride_and_prejudice/valid.txt
head -n 840 $data/pride_and_prejudice/raw/tales.preprocessed.txt | tail -n 400 > $data/pride_and_prejudice/test.txt
tail -n 3075 $data/pride_and_prejudice/raw/tales.preprocessed.txt | head -n 2955 > $data/pride_and_prejudice/train.txt

