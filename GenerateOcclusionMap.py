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



character = bpy.data.objects['me.low_poly.optimized'] # Use the optimized high poly mesh, since this is the mesh we want to use for UV transfer
armature_object = bpy.data.objects['Armature.optimized'] # Use the optimized armature instead of the metrabs armature
armature = bpy.data.armatures['Armature.optimized']
armature_object_optimized = bpy.data.objects['Armature.optimized']
armature_optimized = bpy.data.armatures['Armature.optimized']

oadr.character = character
oadr.armature_object = armature_object
oadr.armature = armature
oadr.armature_object_optimized = armature_object_optimized
oadr.armature_optimized = armature_optimized

def gather_pose_vertices(
	frame,
	vertices_bone_inverted,
):
	# This script requires pretty much the same information as OptimizeArmatureDiffRender subprocess
	# So I'm just going to copy and paste a lot of code from there
	bpy.context.scene.frame_set(frame)

	pose_bone_matrices_world = [pose_bone.matrix.transposed() for pose_bone in armature_object.pose.bones]
	pose_bone_matrices_world = np.array(pose_bone_matrices_world, dtype=np.float64)

	# Vertices output contains the vertices after being posed
	vertices_output = np.matmul(vertices_bone_inverted, pose_bone_matrices_world)
	vertices_output = np.sum(vertices_output, axis=0)

	return vertices_output

if __name__ == '__main__':

	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1



	# Find rest vertices
	vertices = np.array([vertex.co for vertex in character.data.vertices], dtype=np.float64)
	ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
	vertices = np.concatenate([vertices, ones], axis=1)

	# Get rest bone position
	bone_matrices_world = [bone.matrix_local.transposed() for bone in armature.bones]
	bone_matrices_world = np.array(bone_matrices_world, dtype=np.float64)
	bone_matrices_world_inv = np.linalg.inv(bone_matrices_world)

	# Find name to bone_index mapping
	bone_name_to_index = {} # This will be necessary for associating bone weights to matrices
	bone_name_to_index.update([(bone.name, i) for i, bone in enumerate(armature.bones)])

	# Gather vertex group data
	bone_group_indices, bone_group_weights, vertex_group_indices, vertex_group_weights = oadr.get_vertex_groups(bone_name_to_index)

	# Find which vertices represent the head and which ones don't
	head_selection_group_indices = vertex_group_indices['head_selection']
	is_non_head = np.ones(vertices.shape[0], dtype=bool)
	is_non_head[head_selection_group_indices] = False

	# Gather vertices_bone_inverted

	# The vertices array is the location of the vertex after the bone matrix world was applied.
	# So if we apply bone_matrices_world_inv to a vertex, then apply the pose_bone_quaternion / translation,
	# we will be at the expected final vertex location.
	vertices_bone_inverted = np.zeros((bone_group_indices.shape[0], vertices.shape[0], 4), dtype=np.float64)
	for i in range(bone_group_indices.shape[0]):

		ones = np.ones((bone_group_indices[i].shape[0], 1), dtype=np.float64)
		bone_vertices = vertices[bone_group_indices[i], :]
		bone_vertices_weighed = bone_vertices * bone_group_weights[i][:, np.newaxis] # We can apply the scalar weight now since any following matrix multiplications are commutative with scalar multiplication

		inverted = bone_vertices_weighed.dot(bone_matrices_world_inv[i])

		vertices_bone_inverted[i, bone_group_indices[i], :] = inverted

	# Normalize bone weights so they guarantee to sum to 1
	weights = vertices_bone_inverted[:, :, 3]
	vertex_total_weight = weights.sum(axis=0)
	vertices_bone_inverted /= vertex_total_weight[np.newaxis, :, np.newaxis]



	tri_vertex_indices, tri_uv_indices, tri_uvs, tri_normals = oadr.get_triangle_data(orient_faces_to_normals=True, vertices=vertices[:, :3])
	tri_uvs = tri_uvs[tri_uv_indices] # Unwrap uvs

	camera = bpy.data.objects['Camera']

	# Create an output path for the process result
	output_path = bpy.path.abspath('//external_scripts/tmp/')

	python = os.path.join(sys.prefix, 'bin', 'python.exe')
	process_path = bpy.path.abspath('//external_scripts/GenerateOcclusionMap_process.py')

	with subprocess.Popen([python, process_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p: # Run in seperate process so that blender doesn't segfault when the opengl context is released
		head_aspect_ratio = 1.61803398875 # TODO: Config this
		head_image_resolution = (512, int(512*head_aspect_ratio)) # This resolution will be enforced before handing off to mediapipe face mesh

		# Gather subprocess arguments
		args = (
			tri_vertex_indices,
			tri_uvs.astype(np.float32), # moderngl likes floats and we were working with doubles, cast to float32
			is_non_head,
			config.VIDEO_PATH,
			config.VIDEO_GREEN_SCREEN_PATH,
			config.VIDEO_CROP_TRANSPOSE,
			start_frame,
			end_frame,
			output_path,
			head_image_resolution,
		)

		write_pickled(args, p.stdin)


		for frame in range(start_frame, end_frame):
			# Find the vertex locations using pose bones
			vertices_posed = gather_pose_vertices(
				frame+1,
				vertices_bone_inverted,
			)

			intrinsics = cmvp.get_calibration_matrix_K_from_blender(camera.data, use_render_percent=False)
			extrinsics = cmvp.get_3x4_RT_matrix_from_blender(camera)

			projection_matrix = intrinsics.dot(extrinsics)
			screen_points = projection_matrix.dot(vertices_posed.T).T
			screen_points[:, :2] /= screen_points[:, 2, np.newaxis] # Normalize pixel positions, but keep 3rd element as depth

			scene = bpy.context.scene
			video_width = scene.render.resolution_x
			video_height = scene.render.resolution_y

			screen_uvs = screen_points.copy()
			screen_uvs[:, :2] = screen_points[:, :2] / (video_width, video_height)

			# Round screen points so we can use them as image bounds
			screen_xy = screen_points[:, :2].round().astype(np.int32)

			# Find the head_selection_group bounds in pixel space
			head_selection_screen_xy = screen_xy[head_selection_group_indices, :2]

			"""
			screen_xy coordinates:

			* top left of image                   *top right (video width, 0)
			0-----------------------------------> +x
			|
			|
			|
			|
			|
			|
			|
			|
			|
			|
			+y
			* bottom left (0, video height)        * bottom right (video width, video height)
			"""

			# Crop out the head region
			top = head_selection_screen_xy[:, 1].min()
			right = head_selection_screen_xy[:, 0].max()
			bottom = head_selection_screen_xy[:, 1].max()
			left = head_selection_screen_xy[:, 0].min()
			cx = (left + right) / 2 # Find center coordinates
			cy = (top + bottom) / 2
			height = (bottom - top) * 1.2 # Width and height are multiplied by some value TODO: Config this
			width = (right - left) * 1.2

			top = int(cy - height / 2) # Modify the top, bottom, left, and right values
			right = int(cx + width / 2)
			bottom = int(cy + height / 2)
			left = int(cx - width / 2)

			height = max(head_aspect_ratio * width, height) # Enforce an aspect ratio so that all parsed frames have the same shape
			width = max(height / head_aspect_ratio, width)

			height = int(round(height))
			width = int(round(width))

			top_clamped = max(top, 0) # Make sure that by expanding the width and height, we didn't go beyond the bounds of the screen
			right_clamped = min(right, video_width)
			bottom_clamped = min(bottom, video_height)
			left_clamped = max(left, 0)

			bbox = top_clamped, right_clamped, bottom_clamped, left_clamped # This is the region that we need moderngl to render face occlusions for

			# pad_i = top_clamped - top # This will most likely be zero, but stores the amount we should pad the image by if expanding the bbox resulted in an out of frame region
			# pad_j = left_clamped - left
			# clamped_width = right_clamped - left_clamped # Get the width and height of the clamped region
			# clamped_height = bottom_clamped - top_clamped

			# face_image = np.zeros((height, width, 3), dtype=np.uint8) # Create a new image array
			# face_image[pad_i:pad_i+clamped_height, pad_j:pad_j+clamped_width] = frame[top_clamped:bottom_clamped, left_clamped:right_clamped] # Crop the section and pad it if necessary
			# face_image = cv2.resize(face_image, head_image_resolution) # Resize the result to a standard

			frame_args = (
				screen_uvs.astype(np.float32), # moderngl likes floats and we were working with doubles, cast to float32
				bbox,
			)

			write_pickled(frame_args, p.stdin)