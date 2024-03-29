# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021, Nico Jahn
# All rights reserved.
# pylint: disable=missing-module-docstring
import argparse

from utils import load_config


# parsing default parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="open-neural-apc argument parser")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs to train")
    parser.add_argument(
        "-a", "--aux_scale", type=int, help="fraction of aux loss used (denominator)"
    )
    parser.add_argument(
        "-c",
        "--concatenation_length",
        type=int,
        help="the number of sequences to be concatenated",
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, help="constant learning rate"
    )
    parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-d", "--lstm_depth", type=int, help="lstm core depth")
    parser.add_argument("-w", "--lstm_width", type=int, help="lstm core width")
    return parser.parse_known_args()


# overwriting existing config options
def overwrite_config(
    parsed_arguments, data_parameter, model_parameter, training_parameter
):
    arguments = vars(parsed_arguments)
    for argument in arguments:
        if not arguments[argument] is None:
            if argument in data_parameter:
                data_parameter[argument] = arguments[argument]
            elif argument in model_parameter:
                model_parameter[argument] = arguments[argument]
            elif argument in training_parameter:
                training_parameter[argument] = arguments[argument]


# example usage
def main():

    (data_parameter, model_parameter, training_parameter) = load_config(verbose=0)
    (parsed_arguments, _) = parse_arguments()
    overwrite_config(
        parsed_arguments, data_parameter, model_parameter, training_parameter
    )


if __name__ == "__main__":
    main()
