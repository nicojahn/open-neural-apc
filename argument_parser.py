# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import argparse

# parsing default parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='open-neural-apc argument parser')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train')
    parser.add_argument('-a', '--aux_scale', type=int , help='fraction of aux loss used (denominator)')
    parser.add_argument('-c', '--concatenation_length', type=int , help='the number of sequences to be concatenated')
    parser.add_argument('-l', '--learning_rate', type=float , help='constant learning rate')
    parser.add_argument('-b', '--batch_size', type=int , help='training batch size')
    parser.add_argument('-d', '--lstm_depth', type=int , help='lstm core depth')
    parser.add_argument('-w', '--lstm_width', type=int , help='lstm core width')
    return parser.parse_known_args()

# overwriting existing config options
def overwrite_config(parsed_arguments, model_parameter, training_parameter):
    arguments = vars(parsed_arguments)
    for argument in arguments:
        if not arguments[argument] is None:
            if argument in model_parameter:
                model_parameter[argument] = arguments[argument]
            elif argument in training_parameter:
                training_parameter[argument] = arguments[argument]

if __name__ == "__main__":
    # example usage
    from utils import loadConfig
    _, model_parameter, training_parameter = loadConfig(verbose=0)
    parsed_arguments, _ = parse_arguments()
    overwrite_config(parsed_arguments, model_parameter, training_parameter)