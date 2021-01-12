import numpy as np
import moviepy.editor as mpy


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, f_name, duration=2, true_image=False, salimgs=None):

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salimgs[int(len(salimgs) / duration * t)]
        except:
            x = salimgs[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)  # TODO: Remove 2s
    clip.write_gif(f_name, fps=len(images) / duration, verbose=False)
