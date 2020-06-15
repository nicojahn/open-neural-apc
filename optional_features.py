# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf

def createVideo(epoch,batch_idx,sequence,prediction,upper_bound,lower_bound):

    from PIL import Image
    import cv2
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    
    sequence = np.squeeze(sequence)
    prediction = np.squeeze(prediction)
    upper_bound = np.squeeze(upper_bound)
    lower_bound = np.squeeze(lower_bound)

    multiplier = 255/np.max(sequence)
    input_video = np.array(np.reshape(sequence,[-1,20,25])*multiplier,dtype=np.uint8)
    
    # calculate the maximum height (y-values) of the plots
    prediction_max_in = max(prediction[:,0])
    prediction_max_out = max(prediction[:,1])
    label_max_in = max(upper_bound[:,0])
    label_max_out = max(upper_bound[:,1])
    
    # maximal values
    max_in = max(prediction_max_in,label_max_in)
    max_out = max(prediction_max_out,label_max_out)
    
    writer = None
    # for each frame create the following plot
    for k,frame in tqdm(enumerate(input_video),desc='Video progress'):
        
        fig, axes  = plt.subplots(2,1)
        plt.minorticks_on()
        # since saving all plots as images is memory intensive, someone can scale the plots here (smaller values loose details)
        fig.set_size_inches(10, 5)
        
        # upper plot
        axes[0].plot(prediction[:,0],'-',color='navy',label='prediction')
        axes[0].plot([0,k],[prediction[k,0],prediction[k,0]],'--',color='darkgreen')
        axes[0].plot([k,k],[0,max_in+1],color='r')
        axes[0].set_xlabel('# of frames')
        axes[0].set_ylabel('# ingoing people')
        
        # lower plot
        axes[1].plot(prediction[:,1],'-',color='navy',label='prediction')
        axes[1].plot([0,k],[prediction[k,1],prediction[k,1]],'--',color='darkgreen')
        axes[1].plot([k,k],[0,max_out+1],color='r')
        axes[1].set_xlabel('# of frames')
        axes[1].set_ylabel('# outgoing people')
        
        # plot upper and lower bound
        axes[0].plot(upper_bound[:,0],'k-',label='upper bound')
        axes[0].plot(lower_bound[:,0],'k_',label='lower bound')
        axes[1].plot(upper_bound[:,1],'k-',label='upper bound')
        axes[1].plot(lower_bound[:,1],'k_',label='lower bound')
        
        # set good dimensions
        axes[0].set_ylim([0, max_in+1])
        axes[1].set_ylim([0, max_out+1])

        # plot the horizontal lines for easier counting
        spacing = 1
        axes[0].yaxis.set_minor_locator(MultipleLocator(spacing))
        axes[1].yaxis.set_minor_locator(MultipleLocator(spacing))
        axes[0].yaxis.set_major_locator(MultipleLocator(spacing))
        axes[1].yaxis.set_major_locator(MultipleLocator(spacing))
        
        # Set grid to use minor tick locations.
        axes[0].grid(which = 'minor',axis='y',alpha=0.5)
        axes[1].grid(which = 'minor',axis='y',alpha=0.5)
        axes[0].grid(which = 'major',axis='y',alpha=0.5)
        axes[1].grid(which = 'major',axis='y',alpha=0.5)
        
        # clear unused space and draw it without showing a plot
        fig.tight_layout(pad=1)
        fig.canvas.draw()
        
        # Now we can save it to a numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        h,w,c = data.shape
        
        # upscaling of the video image and creating 3 channels
        im = Image.fromarray(frame)
        other_image = np.asarray(im.resize((int(h*(25/20)),h),Image.NEAREST))
        other_image = np.stack((other_image,)*c, axis=-1)
        # combining plots and video
        image = np.concatenate([data,other_image],axis=1)
        
        if k==0:
            writer = cv2.VideoWriter('./results/videos/video%d_%d.avi'%(epoch,batch_idx),\
                             cv2.VideoWriter_fourcc(*'MJPG'), 10, (image.shape[1],image.shape[0]))
        writer.write(image)
        plt.close(fig)
    writer.release()

class customPlot(tf.keras.callbacks.Callback):
    def __init__(self,preprocessor,napc,plot_freq=1000):
        self.plot_freq = plot_freq
        self.preprocessor = preprocessor
        self.napc = napc
        if self.napc.is_distributed:
            raise ValueError("Callback 'customPlot' is currently unsupported when using a distribution strategy.")
    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.plot_freq==0:
            # draw a random sample from all sequences in the preprocessor
            index = np.random.randint(len(self.preprocessor.sequence_list))
            # simulate a epoch with only 1 random sample
            simulated_indices = [[[index]]]
            x = self.preprocessor.epochWrapper(simulated_indices,self.preprocessor.sequenceEpoch)
            label_mask =  self.preprocessor.epochWrapper(simulated_indices,self.preprocessor.labelEpoch)
            accuracy_mask = self.preprocessor.epochWrapper(simulated_indices,self.preprocessor.accuracyEpoch)
            y = self.preprocessor.combineMasks(label_mask,accuracy_mask)

            # draw the only batch
            x = np.asarray(x)[0,:,:,:]
            y = np.asarray(y)[0,:,:,:]

            # process the simulated batch
            prediction = self.napc.model.predict_on_batch(x)
            error = self.napc.loss_function(y[:,:,:2],y[:,:,2:4],prediction)

            # prepare plot
            num_plots = 2
            fig, ax = plt.subplots(1,num_plots,figsize=(14,3), dpi=80)
            fig.suptitle('Predictions in/out')
            for idx in range(num_plots):
                ax[idx].plot(y[0,:,idx],label='max')
                ax[idx].plot(y[0,:,idx+2],label='min')
                ax[idx].plot(y[0,:,4],label='end')
                ax[idx].plot(prediction[0,:,idx],label='prediction')
                ax[idx].plot(error[0,:,idx],label='error')
                ax[idx].legend()
            plt.show(fig)
