import cv2
import bpy
import sys
import pdb
import numpy as np
import mathutils
import math
import subprocess
import os
from multiprocessing import Lock
import threading
import time

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


class InformationGatherer:
	def __init__(self, character_name, armature_name=None, reference_character_name=None):

		# Before we begin, we need to grab the character and the armature
		self.character = bpy.data.objects[character_name]
		self.armature = bpy.data.objects[armature_name] if armature_name is not None else None
		self.reference_character = bpy.data.objects[reference_character_name] if reference_character_name is not None else None # Assuming the reference character has the same topology as the character

	def get_bone_matrices(self):
		if self.armature is None:
			return None

		# Let's start with gathering the armature matrices

		# First we need matrices in rest position
		bone_name_to_index = {} # This will be necessary for associating bone weights to bone matrices
		bone_name_to_index.update([(pose_bone.name, i) for i, pose_bone in enumerate(self.armature.pose.bones)])

		# Bone transform in world coordinates
		bone_matrices_world = [pose_bone.bone.matrix_local for pose_bone in self.armature.pose.bones]
		bone_quaternions  = [matrix.to_quaternion() for matrix in bone_matrices_world]
		bone_translations = [matrix.to_translation() for matrix in bone_matrices_world]

		# Now we need indices that represent the bone parents
		children = [bone_name_to_index[pose_bone.name] for pose_bone in self.armature.pose.bones if pose_bone.parent]
		parents = [bone_name_to_index[pose_bone.parent.name] for pose_bone in self.armature.pose.bones if pose_bone.parent]
		bone_parents = np.zeros(len(self.armature.pose.bones), dtype=np.int64) - 1
		bone_parents[children] = parents

		# Convert bone quaternions and translations into numpy arrays
		bone_quaternions = np.array(bone_quaternions, dtype=np.float32)
		bone_translations = np.array(bone_translations, dtype=np.float32)

		self.bone_name_to_index = bone_name_to_index
		self.bone_parents = bone_parents
		self.bone_quaternions = bone_quaternions
		self.bone_translations = bone_translations

		return bone_name_to_index, bone_parents, bone_quaternions, bone_translations

	def get_pose_bone_matrices(self):
		if self.armature is None:
			return None

		# Pose bone matrix relative to armature
		pose_bone_matrices_world = [pose_bone.matrix for pose_bone in self.armature.pose.bones]
		pose_bone_quaternions  = [matrix.to_quaternion() for matrix in pose_bone_matrices_world]
		pose_bone_translations = [matrix.to_translation() for matrix in pose_bone_matrices_world]
		pose_bone_initial_scales = [matrix.to_scale() for matrix in pose_bone_matrices_world]

		# Convert quaternions, translations, and scales to numpy arrays
		pose_bone_quaternions  = np.array(pose_bone_quaternions, dtype=np.float32)
		pose_bone_translations = np.array(pose_bone_translations, dtype=np.float32)
		pose_bone_initial_scales = np.array(pose_bone_initial_scales, dtype=np.float32)

		self.pose_bone_quaternions = pose_bone_quaternions
		self.pose_bone_translations = pose_bone_translations
		self.pose_bone_initial_scales = pose_bone_initial_scales

		return pose_bone_quaternions, pose_bone_translations, pose_bone_initial_scales

	def get_vertex_colors(self):
		# color_layer = character.data.vertex_colors["optimization_colors"]

		# vertex_colors = np.zeros((len(character.data.vertices), 4), dtype=np.float32)
		# for poly in character.data.polygons:
		# 	for loop_index in poly.loop_indices:
		# 		loop_vert_index = character.data.loops[loop_index].vertex_index

		# 		vertex_colors[loop_vert_index] = color_layer.data[loop_index].color

		# return vertex_colors

		# Grabbing vertex colors from vertex group name
		vertex_colors = np.zeros((len(self.character.data.vertices), 4), dtype=np.float32) + 1 # Default to white

		for color_string, indices in self.vertex_group_indices.items():
			color_strings = color_string.split()
			vertex_colors[indices, :3] = [float(string) for string in color_strings]

		return vertex_colors

	def get_triangle_data(self, orient_faces_to_normals=False, vertices=None):
		uv_layer = self.character.data.uv_layers["UVMap"]

		loop_index_to_vertex_index = np.array([loop.vertex_index for loop in self.character.data.loops], dtype=np.uint32)
		loop_index_to_uv_coord = np.array([data.uv for data in uv_layer.data], dtype=np.float64)
		face_normals = np.array([poly.normal for poly in self.character.data.polygons], dtype=np.float64)

		face_num_vertices = np.array([len(poly.vertices) for poly in self.character.data.polygons], dtype=np.uint8)
		quad_indices = np.where(face_num_vertices == 4)[0]
		invalid_face_indices = np.where(face_num_vertices > 4)[0]

		if invalid_face_indices.shape[0] > 0:
			raise Exception(f"Found faces with more than 4 vertices. I can only automatically triangulate quads. Invalid face indices {invalid_face_indices}")

		# Figure out how many triangles we need to store this mesh
		num_quads = quad_indices.shape[0]
		num_tris_and_quads = face_num_vertices.shape[0]
		num_needed_tris = num_tris_and_quads + num_quads # Double counting quads (since they will become two tris)

		# Handle the first 3 loop indices in each poly as a tri, even if it is a quad
		first_tri_loop_indices = np.array([
			poly.loop_indices[:3]
			for poly in self.character.data.polygons
		], dtype=np.uint32)

		# Handle the second set of loop indices (0, 2, 3) as a tri, only for quads

		# Create a numpy array of blender polygons so we can use fancy indexing to filter for only quads
		poly_object_array = np.array([poly for poly in self.character.data.polygons], dtype=object)

		second_tri_loop_indices = np.array([
			(poly.loop_indices[0], poly.loop_indices[2], poly.loop_indices[3])
			for poly in poly_object_array[quad_indices]
		], dtype=np.uint32)

		# Concatenate first and second tri loop indices so we can handle all of them at once (this implies that the second triangles produced by quads will be towards the end of these arrays)
		if first_tri_loop_indices.size == 0 and second_tri_loop_indices.size == 0:
			raise Exception("The mesh doesn't have any faces???")
		elif first_tri_loop_indices.size == 0:
			all_tri_loop_indices = second_tri_loop_indices
		elif second_tri_loop_indices.size == 0:
			all_tri_loop_indices = first_tri_loop_indices
		else:
			all_tri_loop_indices = np.concatenate([first_tri_loop_indices, second_tri_loop_indices])

		tri_vertex_indices = loop_index_to_vertex_index[all_tri_loop_indices]
		tri_uv_indices = all_tri_loop_indices
		tri_uvs = loop_index_to_uv_coord

		# Create face normals for newly created tris
		tri_normals = np.zeros((num_needed_tris, 3), dtype=np.float64)
		tri_normals[:num_tris_and_quads] = face_normals # The normals for the first_tri_loop_indices will be the exact same as the face_normals
		tri_normals[num_tris_and_quads:] = face_normals[quad_indices] # The normals for second_tri_loop_indices will be a copy of their corresponding first tri normals

		if orient_faces_to_normals:
			# Check if vertices were handed in
			if vertices is None:
				raise Exception("Cannot use orient_faces_to_normals flag without also handing in the vertices array")

			# Get vertices from polygon triangles
			tri_vertices = vertices[tri_vertex_indices] # Unwrap vertices for each tri
			vector10 = tri_vertices[:, 1, :] - tri_vertices[:, 0, :]
			vector20 = tri_vertices[:, 2, :] - tri_vertices[:, 0, :]

			cross_product = np.cross(vector10, vector20)
			dot_product = np.matmul(cross_product[:, np.newaxis, :], tri_normals[:, :, np.newaxis]) # Using matmul so we can batch many dot products
			mask = (dot_product < 0).reshape(tri_vertex_indices.shape[0]) # True if normals oppose vertex order

			# If normals are inverted, flip triangle clockwise / counterclockwise
			tri_vertex_indices[mask, :] = np.fliplr(tri_vertex_indices[mask, :])
			tri_uv_indices[mask, :] = np.fliplr(tri_uv_indices[mask, :])

		return tri_vertex_indices, tri_uv_indices, tri_uvs, tri_normals

	def get_vertices_and_face_indices(self):
		vertices = np.array([vertex.co for vertex in self.character.data.vertices], dtype=np.float64)

		tri_vertex_indices, tri_uv_indices, tri_uvs, tri_normals = self.get_triangle_data(orient_faces_to_normals=True, vertices=vertices)

		vertices = vertices.astype(np.float32)
		tri_vertex_indices = tri_vertex_indices.astype(np.int32)
		tri_uv_indices = tri_uv_indices.astype(np.int32)
		tri_uvs = tri_uvs.astype(np.float32)

		return vertices, tri_vertex_indices, tri_uv_indices, tri_uvs

	def get_vertex_groups(self, return_non_bone_vertex_groups=True):
		group_name_to_index = {}
		group_name_to_index.update( [ (key, group.index) for key, group in self.character.vertex_groups.items() ] )

		vertex_group_indices_dict = {}
		vertex_group_indices_dict.update( [ (group.index, []) for group in self.character.vertex_groups ] ) # Create empty list of vertex indices per bone

		vertex_group_weights_dict = {}
		vertex_group_weights_dict.update( [ (group.index, []) for group in self.character.vertex_groups ] ) # Create empty list of vertex weights per bone

		for vertex in self.character.data.vertices: # Iterate over all vertices in mesh
			for group in vertex.groups: # Iterate over all groups this vertex is a part of
				vertex_group_indices_dict[group.group].append(vertex.index) # Append the vertex index to the list of vertex indices for a group
				vertex_group_weights_dict[group.group].append(group.weight) # Append the vertex weight to the list of vertex weights for a group

		num_bones = len(self.armature.pose.bones)
		bone_vertex_indices = np.empty(num_bones, dtype=object) # Create an element (will be assigned with array of vertex indices) for each bone (even if it doesn't deform the mesh)
		bone_vertex_weights = np.empty(num_bones, dtype=object) # Create an element (will be assigned with array of vertex weights) for each bone (even if it doesn't deform the mesh)
		for bone_name, bone_index in self.bone_name_to_index.items(): # Iterate over bones

			group_index = group_name_to_index.get(bone_name)
			if group_index is not None: # Is a deform bone

				bone_vertex_indices[bone_index] = np.array(vertex_group_indices_dict[group_index], dtype=np.int64)
				bone_vertex_weights[bone_index] = np.array(vertex_group_weights_dict[group_index], dtype=np.float64)

			else: # Non deform bone

				bone_vertex_indices[bone_index] = np.array([], dtype=np.int64)
				bone_vertex_weights[bone_index] = np.array([], dtype=np.float64)

		# Create a dense matrix of the bone weights
		dense_bone_weights = np.zeros((bone_vertex_weights.shape[0], len(self.character.data.vertices)), dtype=np.float64)
		for i in range(bone_vertex_weights.shape[0]):
			dense_bone_weights[i, bone_vertex_indices[i]] = bone_vertex_weights[i]
		self.dense_bone_weights = dense_bone_weights

		if return_non_bone_vertex_groups:
			vertex_group_indices = {}
			vertex_group_weights = {}
			for group_name, group_index in group_name_to_index.items(): # Iterate over all groups
				if group_name not in self.bone_name_to_index: # If it is not a bone group
					vertex_group_indices[group_name] = np.array(vertex_group_indices_dict[group_index], dtype=np.int64)
					vertex_group_weights[group_name] = np.array(vertex_group_weights_dict[group_index], dtype=np.float64)

			self.bone_vertex_indices = bone_vertex_indices
			self.bone_vertex_weights = bone_vertex_weights
			self.vertex_group_indices = vertex_group_indices
			self.vertex_group_weights = vertex_group_weights

			return bone_vertex_indices, bone_vertex_weights, dense_bone_weights, vertex_group_indices, vertex_group_weights
		else:
			self.bone_vertex_indices = bone_vertex_indices
			self.bone_vertex_weights = bone_vertex_weights

			return bone_vertex_indices, bone_vertex_weights, dense_bone_weights

	# ptracker stands for projected tracker. These are trackers that have a 2d screen position defined
	# as well as a 3d world position. The optimizer will pose the 3d points then project them and compare
	# them against the 2d points
	def get_ptrackers(self, frame):
		tracking = config.video.tracking

		ptrackers_2d = []
		ptrackers_3d = []
		ptrackers_bone_weights = []

		# Iterate over the tracking marks and find the 2d screen position, the 3d world position, and the interpolated bone weight associated with the 3d point
		for track in tracking.tracks:
			marker = track.markers.find_frame(frame)
			if marker is not None and not marker.mute:
				# A marker exists for this frame, find the corresponding 3d empty
				empty = bpy.data.objects[track.name]

				# Assert that the shrinkwrap target is our reference character
				target = empty.constraints['Shrinkwrap'].target
				assert self.reference_character == target, "Shrinkwrap modifier target is not as expected"

				# Find the character polygon that is closest to the shrinkwrapped point
				found, location, normal, face_index = target.closest_point_on_mesh(empty.matrix_world.translation)
				assert found == True

				polygon = target.data.polygons[face_index]

				# Find the 2d and 3d coordinates
				ptrackers_2d.append(marker.co) # Coordinates in screen space
				ptrackers_3d.append(location) # Coordinates in world space (on the target surface)

				# Interpolate the bone weights of the nearest polygon for the 3d point (using inverse distance interpolation)
				total_bone_weights   = np.zeros(self.bone_vertex_weights.shape[0], dtype=np.float64)
				inverse_distance_sum = 0

				for vertex_index in polygon.vertices:
					vertex = target.data.vertices[vertex_index]
					distance = (location - vertex.co).length

					inverse_distance = 1 / distance
					total_bone_weights += inverse_distance * self.dense_bone_weights[:, vertex_index]
					inverse_distance_sum += inverse_distance

				interpolated_bone_weights = total_bone_weights / inverse_distance_sum

				ptrackers_bone_weights.append(interpolated_bone_weights)

		num_trackers = len(ptrackers_2d)
		ptrackers_2d = np.array(ptrackers_2d, dtype=np.float64).reshape((-1, 2)) # Using reshape to enforce dimensionality even when ptrackers_2d is absent
		ptrackers_3d = np.array(ptrackers_3d, dtype=np.float64).reshape((-1, 3))
		ptrackers_bone_weights = np.array(ptrackers_bone_weights, dtype=np.float64).reshape((-1, self.bone_vertex_weights.shape[0]))

		return ptrackers_2d, ptrackers_3d, ptrackers_bone_weights

	# wtracker stands for world tracker. These are trackers with a 3d world coordinate
	# that can be defined by the armature pose as well as a ground truth 3d world coordinate.
	# The posed point can be directly compared with the ground truth using sum of squared errors
	def get_metrabs_wtrackers(self, bone_to_metrabs_map):
		if self.armature is None:
			return None

		num_wtrackers = len(bone_to_metrabs_map.items())

		wtrackers_source = np.zeros((num_wtrackers, 3), dtype=np.float64)
		wtrackers_destination = np.zeros((num_wtrackers, 3), dtype=np.float64)
		wtrackers_bone_indices = np.zeros(num_wtrackers, dtype=np.int64)

		for i, (bone_name, metrabs_name) in enumerate(bone_to_metrabs_map.items()):
			# First we need to find the location of the metrabs object relative to it's associated bone
			bone = self.armature.pose.bones[bone_name].bone
			bone_index = self.bone_name_to_index[bone_name]
			metrabs_object = bpy.data.objects[metrabs_name]

			wtrackers_source[i, :] = (0, bone.length, 0) # The metrabs point is supposed to be at the tail of the bone. The tail of the bone lies on the bone's y-axis by definition. The distance from the bone origin to the tail is the bone's length.
			wtrackers_destination[i, :] = metrabs_object.matrix_world.translation
			wtrackers_bone_indices[i] = bone_index

		return wtrackers_source, wtrackers_destination, wtrackers_bone_indices

def output_transforms(armature_output, information_gatherer, optimized_matrices_world, pose_bone_scale, bone_name_to_scale_index, loss, frame):
	# This code assumes armature matrix world does not have any rotation

	# First animate the loss custom variable
	armature_output['optimization_loss'] = loss
	armature_output.keyframe_insert(data_path='["optimization_loss"]', frame=frame+1)

	for i, pose_bone in enumerate(armature_output.pose.bones):
		optimized_matrix_world_with_optimized_parent = optimized_matrices_world[i, :, :].T

		# I only care to get the relative rotation, since translation is never changed by the armature optimizer

		pi = information_gatherer.bone_parents[i]

		# Using numpy instead of mathutils to help reduce rounding errors
		if pi != -1:
			optimized_parent_matrix_world = optimized_matrices_world[pi, :, :].T # World transform of parent bone

			rest_parent_matrix_world = np.array(pose_bone.parent.bone.matrix_local, dtype=np.float64)
			rest_matrix_world_with_rest_parent = np.array(pose_bone.bone.matrix_local, dtype=np.float64)

			rest_matrix_relative_to_rest_parent = np.matmul(np.linalg.inv(rest_parent_matrix_world), rest_matrix_world_with_rest_parent)
			rest_matrix_world_with_optimized_parent = np.matmul(optimized_parent_matrix_world, rest_matrix_relative_to_rest_parent)

			optimized_matrix_relative_to_rest_and_parent = np.matmul(np.linalg.inv(rest_matrix_world_with_optimized_parent), optimized_matrix_world_with_optimized_parent)
		else:
			rest_matrix_world_with_rest_parent = np.array(pose_bone.bone.matrix_local, dtype=np.float64)
			rest_transform_inv = np.linalg.inv(rest_matrix_world_with_rest_parent)
			optimized_matrix_relative_to_rest_and_parent = np.matmul(rest_transform_inv, optimized_matrix_world_with_optimized_parent)

		optimized_matrix_world_with_optimized_parent = mathutils.Matrix(optimized_matrix_world_with_optimized_parent)
		optimized_matrix_relative_to_rest_and_parent = mathutils.Matrix(optimized_matrix_relative_to_rest_and_parent)

		pose_bone.rotation_quaternion = optimized_matrix_relative_to_rest_and_parent.to_quaternion()
		pose_bone.location = optimized_matrix_relative_to_rest_and_parent.to_translation()
		pose_bone.scale = optimized_matrix_relative_to_rest_and_parent.to_scale()

		pose_bone.keyframe_insert(data_path='location', frame=frame+1)
		pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=frame+1)
		pose_bone.keyframe_insert(data_path='scale', frame=frame+1)

		pose_bone_transform = bpy.data.objects[pose_bone.name + '.transform']
		pose_bone_transform.matrix_world = mathutils.Matrix(optimized_matrix_world_with_optimized_parent)

		pose_bone_transform.keyframe_insert(data_path='location', frame=frame+1)
		pose_bone_transform.keyframe_insert(data_path='rotation_quaternion', frame=frame+1)
		pose_bone_transform.keyframe_insert(data_path='scale', frame=frame+1)

		# bpy.context.scene.frame_set(frame+1) # Updates pose_bone.matrix
		# print(pose_bone.name)
		# print(pose_bone.matrix.to_quaternion())
		# print(pose_bone.matrix.to_translation())
		# print(pose_bone.matrix.to_scale())
		# print(optimized_matrix_world_with_optimized_parent.to_quaternion())
		# print(optimized_matrix_world_with_optimized_parent.to_translation())
		# print(optimized_matrix_world_with_optimized_parent.to_scale())

		# import pdb
		# pdb.set_trace()


	# bpy.ops.pose.visual_transform_apply()



def gather_subprocess_args(information_gatherer):
	# Gather information needed to start the process (all gathered args are available within information_gatherer as well, this is important because many of these variables will be needed outside of this function)
	bone_name_to_index, bone_parents, bone_quaternions, bone_translations = information_gatherer.get_bone_matrices()
	bone_vertex_indices, bone_vertex_weights, dense_bone_weights, vertex_group_indices, vertex_group_weights = information_gatherer.get_vertex_groups(return_non_bone_vertex_groups=True)
	vertices, polygons, faces_uvs, uvs = information_gatherer.get_vertices_and_face_indices()
	vertex_colors = information_gatherer.get_vertex_colors()

	texture = cv2.imread(bpy.path.abspath('//resources/texture.png')) / 255

	# Create a tuple which will be pickled and sent to the spawned job
	args = (
		bone_name_to_index,
		bone_name_to_scale_index,
		bone_parents,
		bone_quaternions,
		bone_translations,
		bone_vertex_indices,
		bone_vertex_weights,
		dense_bone_weights,
		vertices,
		vertex_colors,
		polygons,
		faces_uvs,
		uvs,
		texture,
		config.VIDEO_PATH,
		config.VIDEO_GREEN_SCREEN_PATH,
		config.VIDEO_CROP_TRANSPOSE,
	)

	return args


def gather_subprocess_frame_args(
	frame,
	information_gatherer,
	num_iterations=150,
	use_differential_rendering=True,
	bone_to_metrabs_map=None,
	use_wtrackers=False,
	use_ptrackers=True,
	reference_last_frame=False,
	generate_features=False,
	match_features=False,
):
	bpy.context.scene.frame_set(frame+1)

	# Gather information needed to analyze this frame
	camera = bpy.data.objects['Camera']
	intrinsics = cmvp.get_calibration_matrix_K_from_blender(camera.data)
	extrinsics = cmvp.get_3x4_RT_matrix_from_blender(camera)
	pose_bone_quaternions, pose_bone_translations, pose_bone_initial_scales = information_gatherer.get_pose_bone_matrices()

	wtrackers_source, wtrackers_destination, wtrackers_bone_indices = information_gatherer.get_metrabs_wtrackers(bone_to_metrabs_map)

	if use_ptrackers:
		ptrackers_2d, ptrackers_3d, ptrackers_bone_weights = information_gatherer.get_ptrackers(frame)
	else:
		ptrackers_2d = np.array([], dtype=np.float64).reshape((-1, 2)) # Using reshape to enforce dimensionality even when ptrackers_2d is absent
		ptrackers_3d = np.array([], dtype=np.float64).reshape((-1, 3))
		ptrackers_bone_weights = np.array([], dtype=np.float64).reshape((-1, pose_bone_quaternions.shape[0]))

	scene = bpy.context.scene
	quality_scale = scene.render.resolution_percentage / 100

	# Create a tuple which will be pickled and sent to the spawned job
	args = (
		intrinsics,
		extrinsics,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		ptrackers_2d,
		ptrackers_3d,
		ptrackers_bone_weights,
		wtrackers_source,
		wtrackers_destination,
		wtrackers_bone_indices,
		use_wtrackers,
		quality_scale,
		frame,
		use_differential_rendering,
		num_iterations,
		reference_last_frame,
		generate_features,
		match_features,
	)

	return args

frame_lock = Lock() # Only one thread can read / write keyframe data at a time
def spawn_optimizer(frame_range):
	global subprocess_args, information_gatherer, bone_name_to_scale_index, bone_to_metrabs_map, start_time, num_frames_optimized
	python = os.path.join(sys.prefix, 'bin', 'python.exe')
	# process_path = bpy.path.abspath('//external_scripts/OptimizeArmatureDiffRender_process.py')
	# process_path = bpy.path.abspath('//external_scripts/optimize_armature_pytorch.py')
	process_path = bpy.path.abspath('//external_scripts/optimize_armature_pytorch_deodr.py')

	with subprocess.Popen([python, process_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p: # Run in seperate process so that tensorflow releases resources when complete

		# Send args that will remain constant throughout the entire run
		write_pickled(subprocess_args, p.stdin)

		for frame in frame_range:

			is_first_frame = num_frames_optimized == 0

			# Write args to subprocess
			with frame_lock:
				subprocess_frame_args = gather_subprocess_frame_args(
					frame,
					information_gatherer,
					num_iterations=300, # Use 1500 iterations for the first frame
					use_differential_rendering=True,
					bone_to_metrabs_map=bone_to_metrabs_map,
					use_wtrackers=True,
					use_ptrackers=True,
					generate_features=True, # Always generate and use features
					match_features=True,
				)

			write_pickled(subprocess_frame_args, p.stdin)

			# Read response from subprocess
			(
				pose_bone_transforms,
				pose_bone_scale,
				loss,
			) = read_pickled(p.stdout)

			# Insert keyframe

			armature_optimized = bpy.data.objects['Armature.optimized']

			with frame_lock:
				output_transforms(
					armature_optimized,
					information_gatherer,
					pose_bone_transforms,
					pose_bone_scale,
					bone_name_to_scale_index,
					loss,
					frame,
				)

			num_frames_optimized += 1
			curr_time = time.time()

			delta_time = curr_time - start_time
			average_delta_time = delta_time / num_frames_optimized

			print(f"Completed optimizing frame {frame:<8}    Average time per frame {average_delta_time:0.4f}    Frame loss: {loss:0.4f}    Progress {num_frames_optimized / (end_frame - start_frame):0.4f}")

		# Send subprocess kill command
		write_end(p.stdin)

		p.wait()


if __name__ == '__main__':
	bpy.ops.object.mode_set(mode='OBJECT') # Needs to be in object mode to get vertex colors

	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1

	num_workers = 1 # TODO: Config this

	full_frame_range = np.arange(start_frame, end_frame, 5)
	num_frames = full_frame_range.size
	num_frames_padded = math.ceil(num_frames / num_workers) * num_workers # Round up to the nearest multiple of num_workers for padding
	full_frame_range_padded = np.zeros(num_frames_padded, dtype=np.int64) - 1 # Default to -1
	full_frame_range_padded[:num_frames] = full_frame_range # Set valid frames to an arange
	full_frame_range_split = full_frame_range_padded.reshape((-1, num_workers)) # Alternate frame indices amongst workers

	armature_optimized = bpy.data.objects['Armature.optimized']
	bone_name_to_scale_index = {pose_bone.name: i for i, pose_bone in enumerate(armature_optimized.pose.bones)}

	bone_to_metrabs_map = {
		'upper.arm.L': 'elbow.left',
		'lower.arm.L': 'wrist.left',
		'upper.arm.R': 'elbow.right',
		'lower.arm.R': 'wrist.right',
		'thigh.L': 'knee.left',
		'shin.L': 'ankle.left',
		# 'foot.L': 'foot.left',
		'thigh.R': 'knee.right',
		'shin.R': 'ankle.right',
		# 'foot.R': 'foot.right',
	}


	information_gatherer = InformationGatherer('me.low_poly.optimized', 'Armature.optimized', reference_character_name='me.low_poly.reference')
	subprocess_args = gather_subprocess_args(information_gatherer)

	workers = [threading.Thread(target=spawn_optimizer, args=(full_frame_range_split[:, i], ), daemon=True) for i in range(num_workers)]

	start_time = time.time()
	num_frames_optimized = 0

	for worker in workers:
		worker.start()

	for worker in workers:
		worker.join()

	curr_time = time.time()
	delta_time = curr_time - start_time
	print(f"Completed optimizing all frames in {delta_time:0.4f} seconds")

