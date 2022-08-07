"""
A bunch of utility functions for dealing with human3.6m data.
"""

from __future__ import division
import numpy as np

import pickle

import torch
import numpy as np 
import random



def define_actions( action ):
  

  actions = ["open_milk", "open_soda_can", "scratch_sponge", "give_coin", "wash_sponge", "open_letter",
   "unfold_glasses", "open_peanut_butter", "read_letter", "pour_liquid_soap", "pour_wine", "tear_paper", "open_wallet", "put_tea_bag", "high_five", 
  "charge_cell_phone", "handshake", "light_candle", "toast_wine", "scoop_spoon", "flip_sponge", 
  "receive_coin", "close_peanut_butter", "use_flash", "flip_pages", "close_liquid_soap", "close_milk", 
  'squeeze_sponge', 'pour_juice_bottle', 'pour_milk', 'take_letter_from_enveloppe', 'use_calculator', 
  "write", "put_salt", "clean_glasses", "prick", "open_liquid_soap", "open_juice_bottle", 
  "close_juice_bottle", "sprinkle", "give_card", "drink_mug", "stir", "put_sugar", "squeeze_paper"]


  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]

define_actions("*")
