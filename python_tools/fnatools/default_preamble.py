import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import datetime

try:
    print(pyplot_for_latex)
except NameError:
    pyplot_for_latex = False

if pyplot_for_latex:
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.unicode": True,
        "text.latex.preamble": [
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath}",
            r"\usepackage{upgreek}",
            r"\DeclareMathAlphabet{\mathup}{OT1}{\familydefault}{m}{n}",
            r"\newcommand{\sub}[1]{\ensuremath{_{\mathup{#1}}}}",
            r"\newcommand{\unit}[1]{\ensuremath{\,\mathup{#1}}}",
        ],
        "font.family": "serif",
        "font.serif": "Computer Modern Roma",
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "xtick.direction": 'in',
        "ytick.labelsize": 8,
        "ytick.direction": 'in',
        "legend.frameon": False,
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "savefig.dpi": 500,
    })

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roma",
    "text.latex.preamble": [
        r"\usepackage{upgreek}",
    ],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 500,
})

class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def figsize(scalewidth=0.9, ratio=((np.sqrt(5.0) - 1.0) / 2.0)):
    # Get this from LaTeX using \the\textwidth
    fig_width_pt = 418.25368            # obtain from latex by \the\textwidth
    in_per_pt = 1.0 / 72.27                             # Convert pt to inch
    fig_width = fig_width_pt * in_per_pt * scalewidth   # width in inches
    fig_height = fig_width * ratio                      # height in inches
    return [fig_width, fig_height]