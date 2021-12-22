import cv2
import mediapipe as mp
import pdb
import multiprocessing
import bpy
import numpy as np

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





mp_face_mesh = mp.solutions.face_mesh

aspect_ratio = 1.61803398875 # Doesn't matter so why not use the golden ratio
face_mesh_resolution = (256, int(256*aspect_ratio)) # This resolution will be enforced before handing off to mediapipe face mesh

def face_mesh_detection(frame, face_mesh, canonical_face, y_offset, MVP_i, bbox=None):

	if bbox is not None:
		# Expand the extend of the bounding box by a constant factor
		top, right, bottom, left = bbox
		cx = (left + right) / 2 # Find center coordinates
		cy = (top + bottom) / 2
		height = (bottom - top) * 2 # Width and height are multiplied by some value TODO: Config this
		width = (right - left) * 2

		height = max(aspect_ratio * width, height) # Enforce an aspect ratio so that all parsed frames have the same shape
		width = max(height / aspect_ratio, width)

		height = int(round(height)) # Ensure these values are integers
		width = int(round(width))

		top = int(cy - height / 2) # Modify the top, bottom, left, and right values
		right = int(cx + width / 2)
		bottom = int(cy + height / 2)
		left = int(cx - width / 2)

		top_clamped = max(top, 0) # Make sure that by expanding the width and height, we didn't go beyond the bounds of the screen
		right_clamped = min(right, frame.shape[1])
		bottom_clamped = min(bottom, frame.shape[0])
		left_clamped = max(left, 0)

		pad_i = top_clamped - top # This will most likely be zero, but stores the amount we should pad the image by if expanding the bbox resulted in an out of frame region
		pad_j = left_clamped - left
		clamped_width = right_clamped - left_clamped # Get the width and height of the clamped region
		clamped_height = bottom_clamped - top_clamped

		face_image = np.zeros((height, width, 3), dtype=np.uint8) # Create a new image array
		face_image[pad_i:pad_i+clamped_height, pad_j:pad_j+clamped_width] = frame[top_clamped:bottom_clamped, left_clamped:right_clamped] # Crop the section and pad it if necessary
		face_image = cv2.resize(face_image, face_mesh_resolution) # Resize the result to a standard
	else: # If no bounding box was defined, use the entire frame
		face_image = frame

	face_image.flags.writeable = False # This helps things MediaPipe work faster
	results = face_mesh.process(face_image)

	# Draw the face mesh annotations on the image.
	face_image.flags.writeable = True
	if results.multi_face_landmarks:
		face_landmarks = results.multi_face_landmarks[0]

		vertices_screen = np.zeros((468, 2), dtype=np.float32)
		y_world = np.zeros(468, dtype=np.float32)

		for i, landmark in enumerate(face_landmarks.landmark):
			x = landmark.x * width + left
			z = landmark.y * height + top

			vertices_screen[i, 0] = x
			vertices_screen[i, 1] = z

			y = landmark.z * width / frame.shape[1] * 50 + y_offset
			y_world[i] = y

		vertices_camera = cmvp.screen_to_camera(vertices_screen, frame.shape[1], frame.shape[0])
		vertices_world = cmvp.camera_to_world(vertices_camera, y_world, MVP_i)

		for i in range(468):
			canonical_face.data.vertices[i].co = vertices_world[i]

		return True
	else:
		return False

def mediapipe_face_for_current_frame():
	camera = bpy.data.objects['Camera']
	MVP = cmvp.projection_matrix(camera)
	MVP_i = np.linalg.inv(MVP)

	canonical_face = bpy.data.objects['canonical_face']

	# Load face bounding boxes found in DetectFaces.py
	bboxes_path = bpy.path.abspath('//external_scripts/tmp/bboxes.npz')
	with open(bboxes_path, 'rb') as f:
		bboxes = np.load(f)

	neck_upper = bpy.data.objects['neck.upper']
	y_offset = 50

	# Jump to current frame
	scene = bpy.data.scenes['Scene']
	current_frame = scene.frame_current - 1 # Subtract one because blender starts at frame 1 and cv2 starts at frame 0

	cap = start_video_at_frame(config.VIDEO_PATH, current_frame)
	success, frame = get_next_frame(cap, config.VIDEO_CROP_TRANSPOSE)
	if not success: # Early return if frame failed to load
		raise f'Start frame {current_frame} is out of range for video {config.VIDEO_PATH}'

	with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.9) as face_mesh:
		bbox = bboxes[current_frame]
		if bbox[0] != -1: # bbox[0] is negative 1 if no face was detected
			result = face_mesh_detection(frame, face_mesh, canonical_face, y_offset, MVP_i, bbox=bbox)
		else:
			print(f'WARNING: detect_faces_process.py never found a face at frame {current_frame + 1}, attempting to analyze the full frame')
			result = face_mesh_detection(frame, face_mesh, canonical_face, y_offset, MVP_i)

		if not result:
			print(f'MediaPipe failed to find a face at frame {current_frame + 1}')
			error_message("HI")


	cap.release()

	return 0

def ShowMessageBox(message, title, icon='ERROR'):

	def draw(self, context):
		self.layout.label(text=message)

	bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

if __name__ == '__main__':
	mediapipe_face_for_current_frame()