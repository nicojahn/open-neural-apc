# Copyright (c) 2020, Nico Jahn
# All rights reserved.

# load model config
def loadConfig(config_path='config.json',verbose=1):
    import json
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    # extract parameter classes
    data_parameter = config_data['data_parameter']
    model_parameter = config_data['model_parameter']
    training_parameter = config_data['training_parameter']

    # show content of config
    if verbose:
        print(json.dumps(config_data, indent=2, sort_keys=True))
        
    return data_parameter, model_parameter,training_parameter

def allow_growth():
    import tensorflow as tf
    # Copied from: https://tensorflow.google.cn/guide/gpu?hl=en#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)# set this TensorFlow session as the default session for Keras

def prepare_env():
    # disable annoying tf warnings (retracing etc.)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # enable GPU memory growth
    allow_growth()

    # extention to run neural-apc in google colab and retrieving all neccessary files
    import sys
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        os.system("git clone https://github.com/nicojahn/open-neural-apc.git")
        os.chdir('open-neural-apc')