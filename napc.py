from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Reshape, InputLayer, LSTM, Dense, LeakyReLU, Dropout
import numpy as np
import datetime
import os

class NeuralAPC():
    def __init__(self,*args,verbose=0,restored=0,**kwargs):
        self.model_parameter,self.training_parameter = args
        self.model_path = "./models/%s/"%str(datetime.datetime.now()).replace(' ','_')
        self.verbose = verbose
        
        self.model = self.createNewModel()
        # assemble network
        self.AddInput()
        self.AddCore()
        self.AddOutput()
        
        if self.verbose:
            # print the model properties
            self.model.summary()
        
        if restored:
            # load weights, optimizers, losses etc.
            pass
        else:
            # set an epoch
            self.epoch = 0
            self.model.optimizer = keras.optimizers.Adam(self.training_parameter['learning rate'],*self.training_parameter["optimizer parameter"])
            
            # helper for the loss
            self.zero = K.cast(0.,dtype=K.floatx())
            self.one = K.cast(1.,dtype=K.floatx())
            self.aux_scale = K.cast(self.training_parameter['aux scale'],dtype=K.floatx())
            self.slack = K.cast(self.training_parameter['accuracy error niveau'],dtype=K.floatx())
    
    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.model.optimizer, metrics=[self.accuracy])
        
    def save(self):
        #create model directory first
        os.makedirs(self.model_path, exist_ok=True)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%smodel.json"%(self.model_path), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%smodel_%d.h5"%(self.model_path,self.epoch))
        if self.verbose:
            print("Saved model to disk")
    
    def loadModel(self,epoch=-1,model_path=None):
        if epoch < 0:
            epoch = self.epoch
        else:
            self.epoch = epoch
        if model_path is None:
            model_path = self.model_path
        # load json and create model
        with open("%smodel.json"%(model_path), 'r') as json_file:
            model = model_from_json(json_file.read())
        # load weights into new model
        model.load_weights("%smodel_%d.h5"%(model_path,epoch))
        
        # due to an problem in the tf.keras framework a cudnn lstm is restored as cuda lstm (much slower)
        # a newly initialized model uses the cudnn implementation, but needs the right weights
        for idx,layer in enumerate(model.layers):
            self.model.layers[idx].set_weights(layer.get_weights())
        
        if self.verbose:
            print("Loaded model from disk")
        self.compile()
        
    def createNewModel(self):
        # initial definition of the sequential model
        model = keras.Sequential(name='open-neural-apc')
        model.add(InputLayer(input_shape=[None,*self.model_parameter['input dimensions']],dtype=self.training_parameter["calculation dtype"]))
        return model
    
    # input layer which is currently just a dense layer, therefore we have to flatten the input frames        
    def AddInput(self):
        self.model.add(Reshape(target_shape=(-1, np.multiply(*self.model_parameter['input dimensions'])),name='InputReshape'))
        self.model.add(Dense(self.model_parameter['lstm width'],name='InputLayer'))
        self.model.add(Dropout(self.training_parameter['dropout rate'],name='InputDropoutLayer'))
        self.model.add(LeakyReLU(name='InputActivation'))

    # the core network based on lstm
    def AddCore(self):
        for idx in range(self.model_parameter['lstm depth']):
            self.model.add(LSTM(units=self.model_parameter['lstm width'],return_sequences=True,\
                            dropout=self.training_parameter['dropout rate'],name='CoreLayer%d'%idx))

    # the output layer just reducing the dimensionality to the regression output
    def AddOutput(self):
        self.model.add(Dense(self.model_parameter['output dimensions'],name='OutputLayer'))
        self.model.add(LeakyReLU(-1,name='OutputActivation'))
        
    def aux_losses(self,mask,prediction):
        # try to predict close to integer values (error = distance to closest integer)
        integer_error = mask * (prediction-K.round(prediction))

        # try to not change prediction too much through time (error = distance/change to previous prediction)
        small_zero = K.zeros_like(prediction[:,:1,:])
        stabelize_change_forward = K.maximum(self.zero,\
                                             prediction-K.concatenate([prediction[:,1:,:],small_zero],axis=1))
        stabelize_change_backward = mask * K.concatenate([small_zero,stabelize_change_forward[:,:-1,:]],axis=1)

        # freeze your prediction, when shown constant -1 images(padding) (error = distance/change to last valid prediction)
        inverted_mask = K.cast(K.less(mask,self.one),dtype=K.floatx())
        very_last_valid_frame = mask*K.concatenate([inverted_mask[:,1:,:],small_zero],axis=1)
        last_prediction = K.sum(prediction*very_last_valid_frame,axis=1,keepdims=True)
        freezed_last_prediction = inverted_mask * (prediction-last_prediction)

        # combine all auxilary losses
        return K.concatenate([K.abs(integer_error),\
                              K.abs(stabelize_change_backward),
                              K.abs(freezed_last_prediction)],axis=0)
    
    def loss_function(self,upper_bound,lower_bound,prediction):
        mask = K.cast(K.greater_equal(upper_bound,self.zero),dtype=K.floatx())
        # main error to the label (the predictions outside the bounding boxes)
        error = mask * (K.maximum(self.zero,prediction-upper_bound) +\
                            K.minimum(self.zero,prediction-lower_bound))
        # additional losses (independed from label)
        aux_loss = self.aux_losses(mask,prediction)
        return K.abs(error)+K.mean(aux_loss,axis=0,keepdims=True)/self.aux_scale

    def loss(self,y_true, y_pred):
        output_dimensions = self.model_parameter['output dimensions']
        upper_bound = K.cast(y_true[:,:,:output_dimensions],dtype=K.floatx())
        lower_bound = K.cast(y_true[:,:,output_dimensions:2*output_dimensions],dtype=K.floatx())
        return K.mean(self.loss_function(upper_bound,lower_bound,y_pred),axis=0,keepdims=True)

    def accuracy(self,y_true, y_pred):
        output_dimensions = self.model_parameter['output dimensions']
        upper_bound = K.cast(y_true[:,:,:output_dimensions],dtype=K.floatx())

        mask = K.cast(K.greater_equal(upper_bound,self.zero),dtype=K.floatx())
        accuracy_mask = K.cast(y_true[:,:,2*output_dimensions:],dtype=K.floatx())
        # because the accuracy_mask is originally also padded with -1, we mask it
        accuracy_mask = mask * accuracy_mask
        error_with_slack = K.abs(y_pred-upper_bound)-self.slack    
        error_with_slack = K.cast(K.less_equal(error_with_slack,self.zero),dtype=K.floatx())
        # number of right predicted sequences divided by count of sequences
        return K.sum(accuracy_mask*error_with_slack) / K.sum(accuracy_mask)