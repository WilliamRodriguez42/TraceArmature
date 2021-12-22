import os
import numpy as np
import bpy

video = bpy.data.movieclips['video']
green_screen = bpy.data.movieclips['green_screen']

# Target video reconstruction settings
VIDEO_PATH = bpy.path.abspath(video.filepath)
VIDEO_GREEN_SCREEN_PATH = bpy.path.abspath(green_screen.filepath)

#                       Transpose, flip x, flip y
VIDEO_CROP_TRANSPOSE = (    False,  False,  False)
