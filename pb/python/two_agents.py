#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice
from vizdoom import *
from time import time, sleep
from tqdm import trange
from ReplayMemory import *
from DeepNetwork import *
from DqnPlayer import *

import itertools as it
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import os
import errno

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

if __name__ == '__main__':
    player1 = DqnPlayer(name='Player1', colorset='0', is_host=True)
    player2 = DqnPlayer(name='Player2', colorset='3', is_host=False)
    
    p1 = Process(target=player1.run)
    p1.start()
    player2.run()

    print("Done")
