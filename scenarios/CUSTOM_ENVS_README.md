# Note:

These custom scenarios were made by [me](https://github.com/paulobruno)
with the intention to run some specifics tests and may not be suitable
for your use. But if you want to use them, feel free to do so.

## HEALTH POISON
The purpose of this scenario is to teach the agent how to survive
without knowing what makes him survive and what makes him suffer.
Agent know only that life is precious and death is bad so he must
learn what prolongs and what shortens his existence, and that his
health is connected with it.

Map is a rectangle with green, acidic floor which hurts the player
periodically. Initially there are some medkits and poisons spread
uniformly over the map. A new medkit and a new poison fall from the
skies every now and then. Medkits heal some portions of player's
health - to survive agent needs to pick them up - and poisons make
damage to his health - to survive agent cannot pick 3 in sequence.
Episode finishes after player's death or on timeout.

Further configuration:
* living_reward = 1
* 3 available buttons: turn left, turn right, move forward
* 1  available game variable: HEALTH
* death penalty = 100

## HEALTH POISON REWARDS
The purpose of this scenario is to teach the agent how to survive
knowing what makes him survive and what makes him suffer. Agent
know that picking up medkits is good and picking up poisons is bad.

Map is a rectangle with green, acidic floor which hurts the player
periodically. Initially there are some medkits and poisons spread
uniformly over the map. A new medkit and a new poison fall from the
skies every now and then. Medkits heal some portions of player's
health - to survive agent needs to pick them up - and poisons make
damage to his health - to survive agent cannot pick 3 in sequence.
Episode finishes after player's death or on timeout.

__REWARDS:__
+5 for picking up a medkit
-5 for picking up a poison

Further configuration:
* living_reward = 1
* 3 available buttons: turn left, turn right, move forward
* 1 available game variable: HEALTH
* death penalty = 100

## HEALTH POISON REWARDS FLOOR
The purpose of this scenario is to teach the agent how to survive
knowing what makes him survive and what makes him suffer. Agent
know that picking up medkits is good and picking up poisons is bad.

Map is a rectangle with a fixes red, acidic floor which hurts the 
player periodically. Initially there are some medkits and poisons
spread uniformly over the map. A new medkit and a new poison fall 
from the skies every now and then. Medkits heal some portions of
player's health - to survive agent needs to pick them up - and poisons
make damage to his health - to survive agent cannot pick 3 in sequence.
Episode finishes after player's death or on timeout.

__REWARDS:__
+5 for picking up a medkit
-5 for picking up a poison

Further configuration:
* living_reward = 1
* 3 available buttons: turn left, turn right, move forward
* 1 available game variable: HEALTH
* death penalty = 100
