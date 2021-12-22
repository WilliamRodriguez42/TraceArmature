import os
import subprocess
import bpy
import sys
import pathlib
import numpy as np

config = bpy.data.texts["Config"].as_module()

if __name__ == '__main__':
	python = os.path.join(sys.prefix, 'bin', 'python.exe')

	parent_dir = pathlib.Path(__file__).parent.parent.absolute()
	detect_faces_path = bpy.path.abspath('//external_scripts/detect_faces_process.py')

	subprocess.Popen([python, detect_faces_path, config.VIDEO_PATH]).wait() # Run in seperate process so that tensorflow releases resources when complete
