# Copyright (c) 2020, Nico Jahn
# All rights reserved.

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf

def createVideo(epoch,batch_idx,sequence,prediction,upper_bound,lower_bound,dpi=300):

    from PIL import Image
    import cv2
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    
    sequence = np.squeeze(sequence)
    prediction = np.squeeze(prediction)
    upper_bound = np.squeeze(upper_bound)
    lower_bound = np.squeeze(lower_bound)

    multiplier = 255/np.max(sequence)
    input_video = np.array(np.reshape(sequence,[-1,20,25])*multiplier,dtype=np.uint8)
    
    # calculate the maximum height (y-values) of the plots
    max_values = []
    for idx in range(2):
        max_values += [max(max(prediction[:,idx]),max(upper_bound[:,idx]))]
    
    writer = None
    # for each frame create the following plot
    for k,frame in tqdm(enumerate(input_video),desc='Video progress'):
        
        fig, axes  = plt.subplots(2,1, dpi=dpi)
        plt.minorticks_on()
        # since saving all plots as images is memory intensive, someone can scale the plots here (smaller values loose details)
        fig.set_size_inches(8, 4)
        
        for idx, direction in enumerate(['boarding','alighting']):
            # main plot
            axes[idx].plot(prediction[:,idx],'-',color='limegreen',label='prediction')
            # helper lines (horizontal and vertical)
            axes[idx].plot([0,k],[prediction[k,idx],prediction[k,idx]],'--',color='gray')
            axes[idx].plot([k,k],[0,max_values[idx]+1],color='gray')
            axes[idx].set_xlabel('# of frames')
            axes[idx].set_ylabel(f'# {direction} passengers')
        
            # plot upper and lower bound
            axes[idx].plot(upper_bound[:,idx],'--',color='red',label='upper bound')
            axes[idx].plot(lower_bound[:,idx],'--',color='blue',label='lower bound')

            # set good dimensions
            axes[idx].set_ylim([0, max_values[idx]+1])

            # plot the horizontal lines for easier counting
            spacing = 1
            axes[idx].yaxis.set_minor_locator(MultipleLocator(spacing))
            # setting the tick numbers to just integer and not all values have to be plotted
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
        
            # Set grid to use minor tick locations.
            axes[idx].grid(which = 'minor',axis='y',alpha=0.5)
            axes[idx].grid(which = 'major',axis='y',alpha=0.5)
        
        # Adding legend in second plot
        #axes[0].legend()
        axes[1].legend()
        
        # clear unused space and draw it without showing a plot
        fig.tight_layout(pad=1)
        fig.canvas.draw()
        
        # Now we can save it to a numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:,:,::-1]
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

class StoppingAfterWarmup(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='accuracy', min_delta=0.005, patience=100, verbose=0, mode='max',\
                 baseline=None, restore_best_weights=False, baseline_accuracy = 0.4):
        super(StoppingAfterWarmup, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        self.baseline_accuracy = baseline_accuracy
        self.active = False

    def on_epoch_end(self, epoch, logs={}):
        if not "accuracy" in logs: return
        if logs["accuracy"] > self.baseline_accuracy or self.active:
            super().on_epoch_end(epoch, logs)
            self.active = True