import cv2
import bpy
import cupy as cp
import cupyx.scipy.ndimage
import scipy.ndimage
import time

# Goal is to give a rough crop of where the aruco codes could be based on MeTRAbs head point

cmvp = bpy.data.texts["CameraMVP"].as_module()
config = bpy.data.texts["Config"].as_module()
# The following nasty import is done because blender will not reload a library if it is edited
# after blender has already loaded it, even if it was loaded in a previous run of this script.
# Using just plain old from helper_functions import read_pickled would require a restart of blender
# every time that function is edited :(
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
start_video_at_frame, get_next_frame = hf.start_video_at_frame, hf.get_next_frame

if __name__ == '__main__':
	crop_height = 700 # TODO: Config this
	crop_width = 700 # TODO: Config this

	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1

	# Load video
	cap = start_video_at_frame(config.VIDEO_PATH, start_frame)

	# Find MeTRAbs neck upper
	neck_upper = bpy.data.objects['neck.upper']

	# Get camera MVP
	camera = bpy.data.objects['Camera']
	MVP = cmvp.projection_matrix(camera)
	render = bpy.context.scene.render # Get render for resolution

	# Create aruco dictionary and parameters
	aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
	aruco_parameters = cv2.aruco.DetectorParameters_create()

	# Get MeTRAbs head at frame
	for frame in range(start_frame, end_frame):
		print(frame)

		# Load current frame
		success, image = get_next_frame(cap, config.VIDEO_CROP_TRANSPOSE)
		if not success:
			break

		# Get MeTRAbs head location at frame
		bpy.context.scene.frame_set(frame + 1)

		location = neck_upper.location
		screen_location = cmvp.world_to_screen(location, MVP, render.resolution_x, render.resolution_y)

		# Crop image at screen_location
		top =    int(round(screen_location[0, 0] - crop_height / 2))
		bottom = int(round(screen_location[0, 0] + crop_height / 2))
		left =   int(round(screen_location[0, 1] - crop_width / 2))
		right =  int(round(screen_location[0, 1] + crop_width / 2))

		top =    max(top, 0)
		bottom = min(bottom, render.resolution_y)
		left =   max(left, 0)
		right =  min(right, render.resolution_x)

		crop = image[top:bottom, left:right, :]

		# Unsharp mask (on gpu)
		crop_gpu = cp.array(crop) / 255

		for i in range(3):
			blurred = cupyx.scipy.ndimage.gaussian_filter(crop_gpu, (2, 2, 0))
			crop_gpu = crop_gpu * 2 - blurred

		crop_gpu[crop_gpu > 1] = 1
		crop_gpu[crop_gpu < 0] = 0
		crop_gpu = (crop_gpu * 255).astype(cp.uint8)

		crop_sharp = cp.asnumpy(crop_gpu)

		# Find aruco codes
		corners, ids, rejected = cv2.aruco.detectMarkers(crop_sharp, aruco_dictionary, parameters=aruco_parameters)
		if ids is None:
			continue

		marked = cv2.aruco.drawDetectedMarkers(crop_sharp, corners, ids)

		cv2.imshow('marked', marked)
		cv2.waitKey(1)
