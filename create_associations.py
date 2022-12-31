import os
import cv2
import numpy as np

def associate(directory):
  associations = {}
  j = 0
  for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".png"):
      idx = file[:-4]
      associations[idx] = j
      j += 1
  return associations
  