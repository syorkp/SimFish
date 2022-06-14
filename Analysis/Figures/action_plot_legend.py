import numpy as np
import matplotlib.pyplot as plt

from Analysis.Behavioural.Tools.get_action_name import get_action_name

used_actions = [0, 1, 2, 3, 4, 5, 6, 9]
colors = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]
# colors = [c for i, c in enumerate(colors) if i in used_actions]

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none", markersize=20)[0]
handles = [f("s", colors[i]) for i in used_actions]
labels = [get_action_name(i) for i in used_actions]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True, fontsize=20)


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    fig.set_size_inches(10, 10)
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.show()