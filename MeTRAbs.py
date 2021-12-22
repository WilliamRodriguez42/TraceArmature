import pdb
import cv2
import bpy
import sys
import subprocess
import os

# The following nasty import is done because blender will not reload a library if it is edited
# after blender has already loaded it, even if it was loaded in a previous run of this script.
# Using just plain old from helper_functions import read_pickled would require a restart of blender
# every time that function is edited :(
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
read_pickled, write_pickled, write_end = hf.read_pickled, hf.write_pickled, hf.write_end



config = bpy.data.texts["Config"].as_module()
cmvp = bpy.data.texts["CameraMVP"].as_module()

icosphere_order = [
	'spine.3',
	'hip.left',
	'hip.right',
	'spine.2',
	'knee.left',
	'knee.right',
	'spine.1',
	'ankle.left',
	'ankle.right',
	'spine.0',
	'foot.left',
	'foot.right',
	'neck.lower',
	'collar.left',
	'collar.right',
	'neck.upper',
	'shoulder.left',
	'shoulder.right',
	'elbow.left',
	'elbow.right',
	'wrist.left',
	'wrist.right',
	'hand.left',
	'hand.right',
]

def gather_subprocess_args():
	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1

	batch_size = 2 # TODO: config this

	camera = bpy.data.objects['Camera']
	intrinsics = cmvp.get_calibration_matrix_K_from_blender(camera.data, use_render_percent=True) # Use full camera resolution
	quality_scale = bpy.context.scene.render.resolution_percentage / 100
	intrinsics = intrinsics[None, :, :] # Prepend a new axis to represent there is only 1 camera being used throughout the entire video

	model_path = bpy.path.abspath('//models/metrabs_multiperson_smpl_combined') # TODO: config this

	args = (
		start_frame,
		end_frame,
		quality_scale,
		batch_size,
		config.VIDEO_PATH,
		config.VIDEO_CROP_TRANSPOSE,
		intrinsics,
		model_path,
	)

	return args, start_frame, end_frame, batch_size # Return start_frame, end_frame, and batch_size so that this process can loop alongside the subprocess

if __name__ == '__main__':

	python = os.path.join(sys.prefix, 'bin', 'python.exe')
	process_path = bpy.path.abspath('//external_scripts/MeTRAbs_process.py')

	with subprocess.Popen([python, process_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p: # Run in seperate process so that tensorflow releases resources when complete

		# Write args to subprocess
		subprocess_args, start_frame, end_frame, batch_size = gather_subprocess_args()
		write_pickled(subprocess_args, p.stdin)


		for i in range(start_frame, end_frame, batch_size):

			# Read MeTRAbs results
			detections, poses3d, poses2d = read_pickled(p.stdout)

			for j in range(batch_size):
				if detections[j] is not None: # Skip over elements where no person was detected
					for k in range(24):
						ico = bpy.data.objects[icosphere_order[k]]
						ico.location = poses3d[j][k, :] * 0.01 # Multiplying by 0.01 brings the armature closer to the camera, since the camera is also at (0, 0, 0), this does not affect the 2d joint positions
						ico.keyframe_insert(data_path='location', frame=i+j+1)
