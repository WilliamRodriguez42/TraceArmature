import bpy
import numpy as np
import cv2
import os
import sys
import subprocess

config = bpy.data.texts["Config"].as_module()
oadr = bpy.data.texts["OptimizeArmatureDiffRender"].as_module()
cmvp = bpy.data.texts["CameraMVP"].as_module()

# The following nasty import is done because blender will not reload a library if it is edited
# after blender has already loaded it, even if it was loaded in a previous run of this script.
# Using just plain old from helper_functions import read_pickled would require a restart of blender
# every time that function is edited :(
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
def quaternion_to_mat4(quaternion, translation): # Create a wrapper function so I don't have to hand in np all the time
	return hf.quaternion_to_mat4(quaternion, translation, np)
read_pickled, write_pickled, write_end = hf.read_pickled, hf.write_pickled, hf.write_end



character = bpy.data.objects['me.low_poly.metrabs'] # Use the optimized high poly mesh, since this is the mesh we want to use for UV transfer

oadr.character = character

if __name__ == '__main__':


	# Find rest vertices
	vertices = np.array([vertex.co for vertex in character.data.vertices], dtype=np.float64)
	ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
	vertices = np.concatenate([vertices, ones], axis=1)

	tri_vertex_indices, tri_uv_indices, tri_uvs, tri_normals = oadr.get_triangle_data(orient_faces_to_normals=True, vertices=vertices[:, :3])
	tri_uvs = tri_uvs[tri_uv_indices] # Unwrap uvs
	vertices = vertices[tri_vertex_indices] # Unwrap vertices

	camera = bpy.data.objects['Camera']
	model_view_matrix, projection_matrix = cmvp.projection_matrix(camera, as_world_and_perspective=True)

	texture_path = bpy.path.abspath('//resources/texture.png') # TODO: Config this

	# Create an output path for the process result
	output_path = bpy.path.abspath('//external_scripts/tmp/')

	python = os.path.join(sys.prefix, 'bin', 'python.exe')
	process_path = bpy.path.abspath('//external_scripts/find_akaze_features_process.py')

	with subprocess.Popen([python, process_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p: # Run in seperate process so that blender doesn't segfault when the opengl context is released

		# Gather subprocess arguments
		args = (
			config.VIDEO_PATH,
			texture_path,
			vertices,
			tri_uvs,
			projection_matrix,
			output_path,
		)

		write_pickled(args, p.stdin)