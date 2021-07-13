# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_depthmap(depthmap_full_file_path, depthmap_rows, depthmap_cols, 
                  fileformat=np.float32, is_megapov_depthmap = 0):
  depth_array = []
  if depthmap_full_file_path.endswith('.bin'):
    try:
      depth_array = np.fromfile(depthmap_full_file_path, fileformat, -1, '') # the separator character '' specifies a binary file
    except IOError:
      print 'Could not open file ' + depthmap_full_file_path + ' for reading binary data.'
      raise
  else: 
    if depthmap_full_file_path.endswith('.depth'):
      try:
        if is_megapov_depthmap:
          depth_array = np.fromfile(depthmap_full_file_path, dtype='>d')
        else:
          depth_array = np.fromfile(depthmap_full_file_path, np.float32, -1, ' ') # the separator character ' ' specifies a text file
      except IOError:
        print 'Could not open file ' + depthmap_full_file_path + ' for reading text data.'
        raise
    else:
      raise MapIOError('Depthmap filename suffix is not correct.')
  if (len(depth_array) != (depthmap_rows * depthmap_cols)):
      raise MapIOError('Read data do not match the provided size.')
  return depth_array
  
def load_povray_depthmap(depthmap_full_file_path, rows, cols,
                         scale_factor = 1.0, is_megapov_depthmap = 0):
  depth_array = []  
  try:
    depth_array = load_depthmap(depthmap_full_file_path, rows, cols, is_megapov_depthmap)
  except IOError:
    raise
  except MapIOError:
    raise
  return depth_array * scale_factor
  
def show_depthmap(ax, depth_array, rows, cols, min_value = None, max_value = None):
  image = np.reshape(depth_array, [rows, cols], 'C')
  im = ax.imshow(image, vmin = min_value, vmax = max_value)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)
  