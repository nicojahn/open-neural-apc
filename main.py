import tensorflow as tf
import numpy as np
import datetime

import time as t
import pathlib

from network import network
from data_loader import loader as load

from data_preparation import readData
from data_preparation import prepareData
from data_preparation import createMasks

from optional_features import createVideo
from optional_features import metrics

# my metainformation files include
train_file = 'train.csv'
valid_file = 'valid.csv'

# important paths
label_directory = './data/labels/'
data_directory = './data/csv/'
models_directory = './models/'

# switch between training and testing
# testing restores a defined model, disables dropout, sets batch_size to 1 and does not perform gradient steps
# testing does not reduce the concatenation to 1 or disables the augmentations (flipping along axis)
training = True
# if True would use the CUDNN LSTM (GPU only)
# else uses CudnnCompatibleLSTMCell (CPU only)
use_gpu = False

#max epochs or cost minimization
max_epochs = 50
safe_steps = 5
#1: (1-batch) --> stochastic gradient
batchsize = 16
# model specific hyperparameter
input_dims = 500 # flattened input image size
output_dims = 2 # number of labels
hidden_dims = 50 # lstm units in each hidden layer
hidden_layer = 5 # num lstm layers
dropout_rate = 0.2 # only used for LSTM
lr = 0.001 # learning rate (constant during whole training)
accuracy_error_niveau = 0.5 # A slack of 0.5 for the accuracy output (error is still calculated exactly)
jump_frames = 4 # not every frame of those sequences are used

# verbose batch metrics
verbose = True

# some optional features
print_metrics = False
create_video = False

def createModelFolder():
    subdir = datetime.datetime.now().strftime("%y%m%d_%H%M%S.%f")
    subdir = subdir+'_%dx%d_%.5f_%d'%(hidden_layer,hidden_dims,lr,batchsize)
    pathlib.Path(models_directory).mkdir(exist_ok=True)
    model_path = '%s%s/'%(models_directory,subdir)
    pathlib.Path(model_path).mkdir(exist_ok=False)
    pathlib.Path(model_path+'training/').mkdir(exist_ok=False)
    return model_path

def createCheckpointSaver():
    with tf.name_scope("Saver"):
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=None)
    return saver

def performBatch(model,batch,num_batches,sess,x,y,bound_mask,label_mask):

    batch_time = t.time()

    current_x = np.asarray(x[batch],dtype=float)
    current_y = np.array(y[batch],dtype=float)
    bounding_mask_ = np.asarray(bound_mask[batch],dtype=float)
    label_mask_ = np.asarray(label_mask[batch],dtype=float)
    feed = {model['input']:current_x, model['label']:current_y, model['bound_mask']:bounding_mask_, model['label_mask']:label_mask_}

    fetch = [model['loss'],model['acc'],model['epoch_loss_update'],model['epoch_acc_update'], model['opt']]
    loss, acc, _, _, _ = sess.run(fetch, feed_dict=feed)

    if verbose:
        print('took: %f sec. for batch %d of %d\tCost: %f\tAccuracy: %f\n'%(round(t.time()-batch_time,2),batch+1,num_batches,round(loss,5),round(acc,5)))

    if print_metrics:
        fetch = [model['error_with_slack'],model['last_pred'],model['last_label']]
        rounded,last_pred,last_label = sess.run(fetch, feed_dict=feed)
        num_subsequences = np.asarray(np.mean(np.sum(label_mask_,axis=1),axis=1),dtype=int)
        metrics(rounded,last_pred,last_label,num_subsequences)
        

    if create_video:
        fetch = [model['prediction'],model['upper_bound'],model['lower_bound'],model['label_mask']]
        prediction,upper_bound,lower_bound,label_mask = sess.run(fetch, feed_dict=feed)
        # during the computation of the loss, for the last frame of each sequence, the upper bound is crucial
        # the lower bound derived from the bound_mask actually does not change at the last frame
        # therefore, to show how for the model the bounds look like (are calculated), we do this small trick:
        lower_bound = (1.-label_mask)*lower_bound+label_mask*upper_bound
        # plot videos (currently only for first sample in batch)
        createVideo(0,current_x,prediction,upper_bound,lower_bound)
    
def performEpoch(epoch,model,data,label,length,sess,writer):
    epoch_time = t.time()

    # here partition the data, get your masks and then run the m batches
    x, _, sampling, y = prepareData(data,label,batchsize)
    label_mask, bound_mask, y = createMasks(sampling,length, y, batchsize)
    
    # perform all min-batches
    num_batches = len(label_mask)
    for i in range(num_batches):
        performBatch(model,i,num_batches,sess,x,y,bound_mask,label_mask)
    
    # Epoch end metric
    print('#### EPOCH %s took %.2f sec.\tavg. cost.: %f\tavg. acc: %f\n'%(epoch, t.time()-epoch_time, sess.run(model['epoch_loss']), sess.run(model['epoch_acc'])))
    
    if writer is not None:
        # Tensorboard
        # TrainingAccuracy
        writer.add_summary(sess.run(model['epoch_acc_scalar']),epoch)
        # TrainingCost
        writer.add_summary(sess.run(model['epoch_loss_scalar']),epoch)
    
    sess.run([model['epoch_acc_reset'],model['epoch_loss_reset']])

def cleanExit(writer,sess):
    if writer is not None:
        writer.flush()
        writer.close()
    sess.close()
    
def createModel():
    tf.reset_default_graph()
    model = network(input_dims, hidden_dims, output_dims, hidden_layer, lr, accuracy_error_niveau, dropout_rate if training else 0., use_gpu)
    model.create_model()
    saver = createCheckpointSaver()
    return model, saver
    

def getCheckpoints(model_path):
    import re
    import os
    
    # Function from https://gist.github.com/hhanh/1947923 for getting the epoch number of all models in the model_path
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    #iterates through the directory
    checkpoint_list = []
    model_epoch = []
    for file in sorted(os.listdir('%straining/'%model_path),key=natural_keys)[::3]:
        if file.startswith('model'):
            checkpoint_list += ['%straining/%sckpt'%(model_path,file.split('ckpt.')[0])]
            model_epoch += [int(file.split('.ckpt.')[0].split('model')[1])]
    return checkpoint_list, model_epoch

def loadData(label_file):
    # keep noise (differently to bachelor thesis, but reliable) and use only every fourth picture
    data, label = readData(label_directory, label_file, data_directory, keep_noise=True, frame_hops=jump_frames)
    length = np.array(list(map(lambda var: len(var),data)))
    return data,label,length
    
if __name__ == "__main__":

    print("\nNeuralNet with %d hidden layers, where each has %d nodes.\nWith an Error Niveau of %d percent, a dropout rate of %.2f and a learning rate of %.5f\n"%(hidden_layer, hidden_dims, int(accuracy_error_niveau*100), dropout_rate, lr))
    model,saver = createModel()
    
    ############################# HERE STARTS THE SESSION ###############################
    with tf.Session() as sess:
        # init all variables and matrices
        sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
        
        data,label,length = loadData(train_file)
        
        if training:
            # training process
            model_path = createModelFolder()
            writer = tf.summary.FileWriter(logdir='%straining/'%(model_path), graph=sess.graph)
            for epoch in range(max_epochs):
                print('----------------------EPOCH: %d ----------------------\n'%(epoch))
                performEpoch(epoch,model.model,data,label,length,sess,writer)

                ##########################################
                #Safe it after safe_steps epochs
                if epoch % safe_steps == 0:
                    # safe to checkpoint file
                    saver.save(sess, '%straining/model%05d.ckpt'%(model_path,epoch))
        else:
            model_path = models_directory+'200208_201748.255945_5x50_0.00100_16/'
            writer = None
            # test process
            checkpoint_list, model_epoch = getCheckpoints(model_path)
            for idx,epoch in enumerate(model_epoch):
                #restore
                saver.restore(sess,checkpoint_list[idx])
                model.model['opt'] = model.model['acc']
                batchsize = 1
                verbose = False
                
                print('----------------------EPOCH: %d ----------------------\n'%(epoch))
                performEpoch(epoch,model.model,data,label,length,sess,None)
        
        cleanExit(writer,sess)
