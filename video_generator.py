# Copyright (c) 2020, Nico Jahn
# All rights reserved.


def create_video(
    epoch,
    video_index,
    sequence,
    prediction,
    upper_bound,
    lower_bound,
    input_dimensions,
    class_names,
    dpi=300,
    fps=10,
):

    from tqdm.notebook import trange
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    import matplotlib.animation as animation

    plt.minorticks_on()

    # remove batch dimension if present
    sequence = np.squeeze(sequence)
    prediction = np.squeeze(prediction)
    upper_bound = np.squeeze(upper_bound)
    lower_bound = np.squeeze(lower_bound)

    # normalize the video
    multiplier = 255 / np.max(sequence)
    input_video = np.array(
        np.reshape(sequence, [-1, *input_dimensions]) * multiplier, dtype=np.uint8
    )
    sequence_length = input_video.shape[0]
    num_classes = len(class_names)

    # calculate the maximum height (y-values) of the plots
    max_values = []
    for idx in range(num_classes):
        max_values += [max(max(prediction[:, idx]), max(upper_bound[:, idx]))]

    # Initialize the grid
    nrows = 4
    ncols = 14
    grid = GridSpec(
        nrows,
        ncols,
        left=0.05,
        bottom=0.15,
        right=1.0,
        top=0.95,
        wspace=0.1,
        hspace=0.25,
    )

    fig = plt.figure(0, dpi=dpi)
    fig.clf()
    fig.set_size_inches(12, 4)

    # Add axes which can span multiple grid boxes (plots on the left)
    axes = [fig.add_subplot(grid[:2, 0:9])]
    axes += [fig.add_subplot(grid[2:, 0:9])]

    # The image sequence on the right
    image = fig.add_subplot(grid[:, 9:])
    image.set_axis_off()
    im = image.imshow(
        input_video[0],
        interpolation="none",
        aspect="equal",
        vmin=0,
        vmax=255,
        cmap="gray",
    )

    # set all the axis, labels and the prediction curve once
    for idx, direction in enumerate(class_names):
        # main plot
        axes[idx].plot(prediction[:, idx], "-", color="limegreen", label="prediction")
        axes[idx].set_xlabel("# of frames")
        axes[idx].set_ylabel(f"# {direction} passengers")

        # plot upper and lower bound
        axes[idx].plot(upper_bound[:, idx], "--", color="red", label="upper bound")
        axes[idx].plot(lower_bound[:, idx], "--", color="blue", label="lower bound")

        # set good dimensions
        axes[idx].set_ylim([0, max_values[idx] + 1])

        # plot the horizontal lines for easier counting
        spacing = 1
        axes[idx].yaxis.set_minor_locator(MultipleLocator(spacing))
        # setting the tick numbers to just integer and not all values have to be plotted
        axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Set grid to use minor tick locations.
        axes[idx].grid(which="minor", axis="y", alpha=0.5)
        axes[idx].grid(which="major", axis="y", alpha=0.5)

    # legend goes into the last plot
    axes[-1].legend()

    # disable x-axis for all plot but the last
    for ax in axes[:-1]:
        ax.get_xaxis().set_visible(False)

    lines = []
    for i in range(num_classes):
        (line_0,) = axes[i].plot([], [], "--", color="gray")
        (line_1,) = axes[i].plot([], [], color="gray")
        lines += [line_0, line_1]

    def init():
        return lines

    # animation function which updates the image and the visual helper for the predictions
    def animate(k):
        for i in range(num_classes):
            lines[num_classes * i].set_data(
                [0, k], [prediction[k, i], prediction[k, i]]
            )
            lines[num_classes * i + 1].set_data([k, k], [0, max_values[i] + 1])
        im.set_array(input_video[k])
        return lines

    # calling the animation function
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=trange(
            sequence_length,
            total=sequence_length - 1,
            desc="Frames done",
            leave=False,
            unit="frames",
        ),
        interval=1,
        blit=True,
    )

    # saves the animation
    anim.save(
        "./results/videos/video%05d_%03d.mp4" % (epoch, video_index),
        writer="ffmpeg",
        fps=fps,
    )


def create_videos(x, y, predictions, epoch, video_idx_offset, class_names):
    import numpy as np
    import tensorflow as tf
    from tqdm.notebook import trange, tqdm

    # has to create the videos for every element
    for sample_idx, prediction in enumerate(
        tqdm(predictions, desc="Videos created", leave=False, unit="videos")
    ):
        input_dimensions = np.shape(x)[2:]
        output_dimensions = np.shape(predictions)[2]

        # mask/remove the padding if batched
        binary_mask = (
            np.minimum(0, y[sample_idx, :, 2 * output_dimensions]) + 1
        ).astype(bool)
        input_sequence = x[sample_idx][binary_mask]
        pred = prediction[binary_mask]
        lower_bound = y[sample_idx][binary_mask][:, 0:output_dimensions]
        upper_bound = y[sample_idx][binary_mask][
            :, output_dimensions : 2 * output_dimensions
        ]

        # if creating the video takes to long, you can adjust the default dpi=300 parameter
        # epoch and video_index are just used for the video name
        # the other arguments are actually plotted in the video
        video_index = video_idx_offset + sample_idx
        create_video(
            epoch,
            video_index,
            input_sequence,
            pred,
            lower_bound,
            upper_bound,
            input_dimensions,
            class_names=class_names,
        )
