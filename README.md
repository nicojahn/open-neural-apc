# open-neural-apc
## Introduction
This repository contains my original implementation of an RNN-based APC (Automated Passenger Counting) system. This was the subject of my bachelor thesis "Counting People on Image Sequences Using Recurrent Neural Networks". A newer, more basic implementation in TF 2.0 is provided via the jupyter notebook. A pre-trained (maybe not optimal) model is delivered with a handful of test sequences. The bachelor thesis will be provided upon request.

## Which license is applied to the code?
BSD 3-Clause License

## What technologies are used?
* A <a href="https://github.com/tensorflow/tensorflow" target="_blank" rel="noopener noreferrer">Tensorflow</a> 2 implementation of the ML model
* Memory-mapped files which contain the sequences
 * Provides faster access and more efficient memory handling compared to CSV/image files
* Configuration parser and close to complete parameterized code
* A visualization tool, for investigation

## What to expect from such a model?
* The main problem which was solved with this model was an end-to-end ML algorithm
* It only needs single labels per sequences (e.g. count at the end of the sequences)
* The model works efficiently and can process sequences of a variety of lengths
 * dataset has typically 100s to multiple 1000s of images/sequence
 * Uses 10 frames per second
* With the loss function, a property of the data was exploited
 * The problem is the exponential decreasing distribution of the labels (many easy cases, few hard cases)
 * We can combine sequences (concatenate them)
 * A sequences usually start and ends with the train door opening/closing respectively
 * The count can continue for further videos (sequences are independent, but we make them dependent)

## Some validation examples:
* The right side contains the sequence which the model receives as input
 * Top-Down view of a depth-sensing sensor (20x25 pixel resolution)
 * Noisy sequence
 * The door is located on the bottom
* On the left half of the video, the neural-apc is shown in action
 * The top plot is the alighting passengers, bottom the boarding ones
 * The x-axis resembles the number of frames
 * The y-axis the count
 * The black line on the top the label
  * Only label for the end of the sequence as upper bound for every intermediate frame
 * The red line is the current progress of the video
 * The green dashed line is just a helping projection of the current model count to the y-axis
 * The blue line is the **raw output** of the model
 
![neural-apc](./results/gifs/10000_9.gif)

You can find more GIFs <a href="./results/gifs/" target="_blank" rel="noopener noreferrer">here</a>.

## What do the labels look like?
In short:
 * What we want is the green line (should be the ground truth and therefore, also our prediction)
 * We just have a label for the last frame of the video, but since the optimal counting function of each category is non-decreasing, we have a lower and upper bound (via the label)
 * We just learn from this bounding box the green line

<object data="https://github.com/nicojahn/open-neural-apc/blob/master/label_problem.pdf" type="application/pdf" width="750px" height="750px">
    <embed src="https://github.com/nicojahn/open-neural-apc/blob/master/label_problem.pdf" type="application/pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/nicojahn/open-neural-apc/blob/master/label_problem.pdf">Download PDF</a>.</p>
    </embed>
</object>

## You told the loss function was special?
In my honest opinion, the model is rather boring, it was planned just as a basic model. You can see it at the bottom of this page (Visualized with <a href="https://github.com/lutzroeder/netron" target="_blank" rel="noopener noreferrer">Netron</a> by <a href="https://github.com/lutzroeder" target="_blank" rel="noopener noreferrer">Lutz Roeder</a>). The main selling point, the loss function, was chosen to avoid a weighting of classes. As previously mentioned the labels are (as usual) not well distributed. More than 75% of the >5000 labels (of ~2500 sequences) contained only 0,1 or 2 passengers. Luckily we can produce larger labels when concatenating sequences and therefore get larger counts. The drawback is the computational complexity when preparing batches and computing through the sequences.<br>
Visual representation of the concatenation and a synthetic prediction is shown below. Since the bounding boxes are the only knowledge we have, we can't apply a loss as long as the prediction is inside the boundaries. Only if the model predicts values outside the valid range or for the last frame of each sequence we can compute an error. For the last frame, we calculate an absolute loss. These are the only losses that are applied and rely on the labels. Auxiallry losses were introduced by myself in 2020 in this repository.

<object data="https://github.com/nicojahn/open-neural-apc/blob/master/concatenate.pdf" type="application/pdf" width="750px" height="750px">
    <embed src="https://github.com/nicojahn/open-neural-apc/blob/master/concatenate.pdf" type="application/pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/nicojahn/open-neural-apc/blob/master/concatenate.pdf">Download PDF</a>.</p>
    </embed>
</object>

### Basic model
![NN model](./model.json.svg)