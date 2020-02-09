import numpy as np

def createVideo(epoch,current_x,prediction,upper_bound,lower_bound):

    from PIL import Image
    import imageio
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    
    # create video
    sample = 0
    # get some properties of the predicitions
    vid_length = current_x[sample].shape[0]
    input_video = np.array(np.reshape(current_x[sample],[vid_length,20,25])*255,dtype=np.uint8)
    
    # calculate the maximum height (y-values) of the plots
    prediction_max_in = max(prediction[sample][:vid_length,0])
    prediction_max_out = max(prediction[sample][:vid_length,1])
    label_max_in = max(upper_bound[sample][:vid_length,0])
    label_max_out = max(upper_bound[sample][:vid_length,1])
    # maximal values
    max_in = max(prediction_max_in,label_max_in)
    max_out = max(prediction_max_out,label_max_out)
    
    # image buffer for the video
    image_list = []
    # to keep the memory usage low for longer sequence skip some frames and use at most 500 frames
    frames = np.linspace(0,vid_length-1,num=min(vid_length,400),endpoint=True,dtype=int)
    # for each frame create the following plot
    for k in frames:
        
        fig, axes  = plt.subplots(2,1)
        # since saving all plots as images is memory intensive, someone can scale the plots here (smaller values loose details)
        # used in bachelors thesis (10,5)
        fig.set_size_inches(10, 5)
        
        # upper plot
        axes[0].plot(prediction[sample][:vid_length,0],'-',color='navy',label='prediction')
        axes[0].plot([0,k],[prediction[sample][k,0],prediction[sample][k,0]],'--',color='darkgreen')
        axes[0].plot([k,k],[0,max_in+1],color='r')
        axes[0].set_xlabel('# of frames')
        axes[0].set_ylabel('# ingoing people')
        
        # lower plot
        axes[1].plot(prediction[sample][:vid_length,1],'-',color='navy',label='prediction')
        axes[1].plot([0,k],[prediction[sample][k,1],prediction[sample][k,1]],'--',color='darkgreen')
        axes[1].plot([k,k],[0,max_out+1],color='r')
        axes[1].set_xlabel('# of frames')
        axes[1].set_ylabel('# outgoing people')
        
        # plot upper and lower bound
        axes[0].plot(upper_bound[sample][:vid_length,0],'k-',label='upper bound')
        axes[0].plot(lower_bound[sample][:vid_length,0],'k_',label='lower bound')
        axes[1].plot(upper_bound[sample][:vid_length,1],'k-',label='upper bound')
        axes[1].plot(lower_bound[sample][:vid_length,1],'k_',label='lower bound')
        
        # set good dimensions
        axes[0].set_xlim([0, vid_length+10])
        axes[1].set_xlim([0, vid_length+10])
        axes[0].set_ylim([0, max_in+1])
        axes[1].set_ylim([0, max_out+1])

        # plot the horizontal lines for easier counting
        spacing = 1
        axes[0].yaxis.set_minor_locator(MultipleLocator(spacing))
        axes[1].yaxis.set_minor_locator(MultipleLocator(spacing))
        # Set grid to use minor tick locations.
        axes[0].grid(which = 'minor',axis='y',alpha=0.5)
        axes[1].grid(which = 'minor',axis='y',alpha=0.5)
        
        # clear unused space and draw it without showing a plot
        fig.tight_layout(pad=1)
        fig.canvas.draw()
        
        # Now we can save it to a numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        h,w,c = data.shape
        
        # upscaling of the video image and creating 3 channels
        im = Image.fromarray(input_video[k])
        other_image = np.asarray(im.resize((int(h*(25/20)),h),Image.NEAREST))
        other_image = np.stack((other_image,)*c, axis=-1)
        # combining plots and video
        image = np.concatenate([data,other_image],axis=1)
        # save image to the videobuffer
        image_list += [image]
        plt.close(fig)

    # write the whole video at once (memory intensive)
    vid_writer = imageio.get_writer('videos/video%d.mp4'%(epoch), fps=10)
    for elem in image_list:
        vid_writer.append_data(elem)
    vid_writer.close()
    
def metrics(rounded,last_pred,last_label,num_subsequences):
    from tabulate import tabulate
    # per sequence level
    num_subsequences = list(num_subsequences)+[0]
    for idx,_ in enumerate(num_subsequences[:-1]):
        # for choosing which labels/predcitions have to be taken from everything
        start = sum(num_subsequences[:idx])
        ende = sum(num_subsequences[:idx+1])
        # stack accumulated prediction and label + indicator if it was right/wrong (with respect to slack)
        def getMetricStack(class_idx):
            return np.around(np.stack([last_pred[start:ende,class_idx],\
                                            last_label[start:ende,class_idx],\
                                            np.equal(rounded[start:ende,class_idx],\
                                                    0*rounded[start:ende,class_idx])\
                                            ],1),5)
        # nice tables on command line
        print('Ingoing:')
        print(tabulate(getMetricStack(0), headers=['prediction', 'accumulated label', 'False(0)/True(1)'], tablefmt='orgtbl'))
        print('Outgoing:')
        print(tabulate(getMetricStack(1), headers=['prediction', 'accumulated label', 'False(0)/True(1)'], tablefmt='orgtbl'))
        print()

