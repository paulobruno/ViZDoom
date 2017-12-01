#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from DqnPlayer import *

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread

if __name__ == '__main__':

    config_file = 'settings.cfg'

    player1 = DqnPlayer(name='Player1', colorset='0', is_host=True, settings_file=config_file)
    player2 = DqnPlayer(name='Player2', colorset='3', is_host=False, settings_file=config_file)
    
    p1 = Process(target=player1.run)
    p1.start()
    player2.run()

    print("Done")
