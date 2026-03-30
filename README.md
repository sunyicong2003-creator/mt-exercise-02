# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Task 2: Parameter Tuning -Dropout

### Description

In this task, we investigate the effect of different dropout values on language model performances.
we train multiple models with varying dropout settings and compare their training, validation, and test perplexities.

---


### Changes Made

- Modified `tools/pytorch-examples/word_language_model/main.py`:
  - Added logging for:
    - Training perplexity (`*.train.log`)
    - Validation perplexity (`*.valid.log`)
    - Test perplexity (`*.test.log`)
  - Computed average training loss per epoch

- Created multiple training scripts with different dropout values:
  - `scripts/train_dropout_0.sh`
  - `scripts/train_dropout_0.1.sh`
  - `scripts/train_dropout_0.3.sh`
  - `scripts/train_dropout_0.6.sh`
  - `scripts/train_dropout_0.9.sh`

- Created `scripts/make_tables_and_plots.py`:
  - Reads log files from `models/`
  - Generates:
    - CSV tables for training, validation, and test perplexity
    - Line plots for training and validation perplexity

- Generate samples:
 - Modified `./scripts/generate.sh`
 	- replacing `$models/model.pt \` to `$models/model_(DROPOUT VALUE YOU NEED).pt \`

	
# Steps


### How to Run

Run the following commands in order:

- train models
```bash
./scripts/train_dropout_0.sh'
./scripts/train_dropout_0.1.sh
./scripts/train_dropout_0.3.sh
./scripts/train_dropout_0.6.sh
./scripts/train_dropout_0.9.sh
```
- get the plots and tables
`python scripts/make_tables_and_plots.py`

- generate samples
`./scripts/generate`


