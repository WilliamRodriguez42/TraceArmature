import bpy
import subprocess
import sys
import os
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
read_pickled, write_pickled = hf.read_pickled, hf.write_pickled

cmvp = bpy.data.texts["CameraMVP"].as_module()

if __name__ == '__main__':
	python = os.path.join(sys.prefix, 'bin', 'python.exe')
	process_path = bpy.path.abspath('//external_scripts/hand_capture_process.py')
	cameras = [bpy.data.objects[f'Camera.00{i}'] for i in range(5)]
	MVPs = np.array([cmvp.projection_matrix(camera) for camera in cameras], dtype=np.float64)

	with subprocess.Popen([python, process_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p: # Run in seperate process since mediapipe cannot open correctly within blender
		points = write_pickled(MVPs, p.stdin)
		points = read_pickled(p.stdout)

		p.wait()

	for i in range(21):
		hand_landmark_object = bpy.data.objects[f'HandLandmark.{i+1:>03}']

		hand_landmark_object.location = points[i]

	print("SUCCESS")