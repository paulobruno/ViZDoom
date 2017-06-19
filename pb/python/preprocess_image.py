#!/usr/bin/env python
#####################################################################
# This script presents a way to save 3 different types of view of 
# screen buffer, which is the state given to the agent.
#
# Paulo Bruno de Sousa Serafim
# Federal University of Ceara, Fortaleza, Brazil
#####################################################################

from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

import numpy as np
import skimage.color, skimage.transform
import matplotlib.pyplot as plt

config_file_path = '../../scenarios/basic.cfg'
game = DoomGame()
game.load_config(config_file_path)
game.set_window_visible(False)
game.set_mode(Mode.PLAYER)
game.set_screen_format(ScreenFormat.BGR24)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

grayscale = True

def preprocess(img, def_h, def_w, gray):
    img = skimage.transform.resize(img, (def_w, def_h))
    if gray:
        img = skimage.color.rgb2gray(img) # convert to gray
    img = img.astype(np.float32)
    return img

# colored preprocessed image
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(7.2, 4.8)
#fig.set_size_inches(0.45, 0.3)
img1 = preprocess(game.get_state().screen_buffer, 90, 60, False)
ax.imshow(img1)
plt.savefig('../../../basic_preproc.png')
#fig.add_subplot(1,3,2)
#plt.imshow(img1)
#plt.show()
plt.close(fig)

# colored full definition image
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(6.4, 4.8)
ax.imshow(game.get_state().screen_buffer)
plt.savefig('../../../basic.png')
#fig.add_subplot(1,3,1)
#plt.imshow(game.get_state().screen_buffer)
#plt.show()
plt.close(fig)

# grayscale preprocessed image
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(7.2, 4.8)
#fig.set_size_inches(0.45, 0.3)
img2 = preprocess(game.get_state().screen_buffer, 90, 60, grayscale)
# ploting if converted to gray
if grayscale: 
    img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)
ax.imshow(img2)
plt.savefig('../../../basic_gray.png')
#fig.add_subplot(1,3,3)
#plt.imshow(img2)
#plt.show()
plt.close(fig)
