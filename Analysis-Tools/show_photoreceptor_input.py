import json
import numpy as np
import moviepy.editor as mp

from Tools.make_gif import make_gif

# TODO: Not finished or used yet.

with open("../Assay-Output/base-1/Assay-1.json", "r") as file:
    data = json.load(file)

observation = [i["observation"] for i in data]

observation = np.array(observation)

make_gif(observation, "observation.gif", duration=len(observation) * 0.03, true_image=True)

text_list = range(len(observation))


# Add a timestamp
def time_stamp(clip):
    """ Adds today's date at the bottom right of the clip"""
    clip_list = []
    for text in text_list:
        print(str(text))
        txt = (mp.TextClip(txt=str(text),
                           fontsize=70, color="white", font='Ubuntu-Bold',
                           stroke_width=5, stroke_color="black")
               # .resize(width=clip.w / 3)  # txt width will be one third of clip width
               .set_position(('right', 'bottom'))
               .set_duration(1))
        clip_list.append(txt)

    return mp.CompositeVideoClip([clip, clip_list])


clip = mp.VideoFileClip("observation.gif")
timestamped_clip = time_stamp(clip)
timestamped_clip.write_videofile('observation_timestamped.gif', bitrate='8000k')

