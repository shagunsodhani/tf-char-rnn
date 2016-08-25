# tf-char-rnn

Tensorflow implementation of char-rnn using Multi-layer Recurrent Neural Networks.

## Usage

* Install [Tensorflow](http://www.tensorflow.org)
* Put the data (to train on) in a file called 'data.txt'
* Run `python train.py --input_dir=path_to_folder_containing_data.txt`
* For example, if data is in `input` folder, run `python train.py --input_dir=input`
* To generate text using a checkpointed model, run `python generate.py`
* All supported arguments can be seen in [here](app/utils/argumentparser.py)

## Standing on the shoulder of giants

* Tensorflow [Recurrent Neural Networks](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks)
* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* sherjilozair's [char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)
