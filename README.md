# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Task 1: Custom Dataset

For Task 1, replacing the original Grimm dataset with a new dataset:"Pride and Prejudice" from Project Gutenberg.

### Changes Made

- Modified 'scripts/download_data.sh':
	- Replaced the dataset with "Pride and Prejudice" (Gutenberg ID 1342)
	- Removed the use of 'preprocess_raw.py', as it removed too much content for this dataset
	- Used 'cp' to directly copy the raw text before tokenisation

- Update dataset paths:
 - Changed from'data/grimm' to 'data/pride_and_prejudice'
 
- Update scripts:
	- 'scripts/train.sh' now uses the new dataset
	- 'scripts/generate.sh' now uses the new dataset
	
# Steps

Clone this repository in the desired place:

    git clone https://github.com/marcamsler1/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh


