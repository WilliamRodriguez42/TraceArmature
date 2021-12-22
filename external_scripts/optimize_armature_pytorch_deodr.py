import pickle
import struct
import os
import cv2
import pdb
import math
import numpy as np
import moderngl
import moderngl_window
import cupy as cp
import cupyx.scipy.ndimage
import cupyx.scipy.signal

import torch
import torch.nn as nn

from differentiable_renderer_pytorch import CameraPytorch, Scene3DPytorch
from deodr.pytorch.triangulated_mesh_pytorch import ColoredTriMeshPytorch as ColoredTriMesh
from deodr.pytorch.triangulated_mesh_pytorch import TriMeshPytorch as TriMesh

from helper_functions import print, start_video_at_frame, read_pickled, write_pickled, quaternion_to_mat4, get_keypoints_and_descriptors, match_descriptors, quaternion_mul
def _quaternion_to_mat4(quaternion, translation, scale=None): # Create a wrapper function so I don't have to hand in torch all the time
	return quaternion_to_mat4(quaternion, translation, scale=scale, backend=torch)

from custom_transform_modifiers import CustomTransformModifiers
from custom_transform_operations import CustomTransformOperations

def repackage(var, requires_grad):
	var.detach_()
	var.requires_grad = requires_grad

class ArmatureLayer(nn.Module):
	"""Class to fit a deformable mesh to a color image."""

	def __init__(
		self,
		vertices_input,
		vertex_colors,
		faces,
		faces_uvs,
		uvs,
		texture,

		bone_name_to_index,
		bone_scale_share_map,
		bone_parents,
		bone_quaternions,
		bone_translations,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		bone_vertex_indices,
		bone_vertex_weights,

		use_differential_rendering
	):
		super(ArmatureLayer, self).__init__()

		# Calculate bone and pose_bone world transforms
		bone_matrices_world = quaternion_to_mat4(bone_quaternions, bone_translations) # Using original quaternion_to_mat4 to use numpy backend
		bone_matrices_world_inv = np.linalg.inv(bone_matrices_world)

		_, pose_bone_matrices_world = quaternion_to_mat4(pose_bone_quaternions, pose_bone_translations, pose_bone_initial_scales)
		pose_bone_matrices_world_inv = np.linalg.inv(pose_bone_matrices_world)

		# Copy bone name to index
		self.bone_name_to_index = bone_name_to_index
		self.bone_index_to_name = {value: key for key, value in bone_name_to_index.items()} # Also store the inverse

		# Initialize the custom transform modifiers and operations
		self.custom_transform_modifiers = CustomTransformModifiers(self.bone_name_to_index, self.bone_index_to_name)
		self.custom_transform_operations = CustomTransformOperations(self.bone_name_to_index, self.bone_index_to_name)

		# Apply modifiers to the handed-in pose bone data
		pose_bone_transforms_scaled = np.zeros((pose_bone_quaternions.shape[0], 4, 4), dtype=np.float64)
		pose_bone_transforms_scaled[:] = np.eye(4)[np.newaxis, :, :]
		self.custom_transform_modifiers.start_iteration(
			pose_bone_quaternions.copy(), # Hand in copies so that the input quaternions don't get modified with non-world space quaternions as the iterations occur
			pose_bone_translations.copy(),
			pose_bone_initial_scales.copy(),
			pose_bone_transforms_scaled, # This doesn't have to be a copy since we are expected to update this list of matrices with world space pose bone matrices as the iterations occur
		)
		for bone_index in range(pose_bone_quaternions.shape[0]):
			(
				pose_bone_quaternions[bone_index],
				pose_bone_translations[bone_index],
				pose_bone_initial_scales[bone_index],
				pose_bone_transforms_scaled[bone_index],
			) = self.custom_transform_modifiers[self.bone_index_to_name[bone_index]]( # To keep a similar syntax as custom_transform_operations, hand in current iteration values and set current iteration values to result
				bone_index,
				bone_parents[bone_index],
				pose_bone_quaternions[bone_index],
				pose_bone_translations[bone_index],
				pose_bone_initial_scales[bone_index],
			)

		# Convert inputs to torch tensors
		vertices_input = torch.tensor(vertices_input, dtype=torch.float64, requires_grad=False)
		vertex_colors = torch.tensor(vertex_colors, dtype=torch.float64, requires_grad=False)
		faces = torch.tensor(faces, dtype=torch.int64, requires_grad=False)
		bone_quaternions = torch.tensor(bone_quaternions, dtype=torch.float64, requires_grad=False)
		bone_translations = torch.tensor(bone_translations, dtype=torch.float64, requires_grad=False)
		bone_parents = torch.tensor(bone_parents, dtype=torch.int64, requires_grad=False)
		pose_bone_quaternions = torch.tensor(pose_bone_quaternions, dtype=torch.float64, requires_grad=True)
		pose_bone_translations = torch.tensor(pose_bone_translations, dtype=torch.float64, requires_grad=True) # Might need to be requires_grad=False
		pose_bone_initial_scales = torch.tensor(pose_bone_initial_scales, dtype=torch.float64, requires_grad=False)
		bone_matrices_world = torch.tensor(bone_matrices_world, dtype=torch.float64, requires_grad=False)
		bone_matrices_world_inv = torch.tensor(bone_matrices_world_inv, dtype=torch.float64, requires_grad=False)
		pose_bone_matrices_world = torch.tensor(pose_bone_matrices_world, dtype=torch.float64, requires_grad=False)
		pose_bone_matrices_world_inv = torch.tensor(pose_bone_matrices_world_inv, dtype=torch.float64, requires_grad=False)

		bone_vertex_indices = [torch.tensor(value, dtype=torch.int64, requires_grad=False) for value in bone_vertex_indices]
		bone_vertex_weights = [torch.tensor(value, dtype=torch.float64, requires_grad=False) for value in bone_vertex_weights]

		# Register buffers and parameters
		self.register_buffer('vertex_colors', vertex_colors)
		self.register_buffer('faces', faces)
		self.register_buffer('pose_bone_initial_scales', pose_bone_initial_scales)
		self.register_buffer('bone_parents', bone_parents)
		self.register_buffer('bone_matrices_world', bone_matrices_world)
		self.register_buffer('bone_matrices_world_inv', bone_matrices_world_inv)

		self.pose_bone_quaternions = torch.nn.Parameter(pose_bone_quaternions)

		# Copy uv data as numpy
		self.faces_uvs = faces_uvs.copy()
		self.uvs = uvs.copy()
		self.texture = texture[::4, ::4, :].copy() # Downscale the object texture for faster rendering

		# Convert uv range from [0, 1] to [0, width or height]
		self.uvs[:, 0] *= self.texture.shape[0]
		self.uvs[:, 1] = 1 - self.uvs[:, 1]
		self.uvs[:, 1] *= self.texture.shape[1]

		# Convert texture from bgr to rgb
		self.texture = self.texture[:, :, ::-1]
		self.texture *= 1.5 # Make the texture a tad bit brighter
		self.texture[self.texture > 1] = 1

		# Find the bone head position relative to the bone's parent
		num_bones = pose_bone_translations.shape[0]
		relative_pose_bone_translations = torch.ones((num_bones, 4), dtype=torch.float64)
		relative_pose_bone_translations[:, :3] = pose_bone_translations # Default to absolute world translation, this is used in the case a bone has no parent

		has_parent = self.bone_parents != -1 # Mask of which bones have parents
		valid_parent_indices = self.bone_parents[has_parent] # Indices of parents for bones that have parents

		parent_pose_bone_matrices_inv = pose_bone_matrices_world_inv[valid_parent_indices, :, :] # Bones that have parents need their translations transformed by the parent inverse to get their translations relative to their parent

		temp = torch.matmul(relative_pose_bone_translations[has_parent, np.newaxis, :], parent_pose_bone_matrices_inv) # Multiply each bone translation by the inverse of the parent matrix
		relative_pose_bone_translations[has_parent, :] = temp[:, 0, :] # Remove the broadcasting axis

		self.register_buffer('relative_pose_bone_translations', relative_pose_bone_translations)

		# The vertices_input array is the location of the vertex after the bone matrix world was applied (a.k.a. the rest pose).
		# We will need to apply bone_matrices_world_inv to a vertex now, so that when we later apply the pose bone transformation in world space,
		# we will be at the expected final vertex location for that bone. We can also apply the bone weights now, that way,
		# when we later apply each pose bone transformation, we can just take the sum to get the final position.
		vertices_bone_inverted = torch.zeros((len(bone_vertex_indices), vertices_input.shape[0], 4), dtype=torch.float64)
		for bone_index in range(bone_matrices_world.shape[0]):

			ones = torch.ones((bone_vertex_indices[bone_index].shape[0], 1), dtype=torch.float64)
			bone_vertices = vertices_input[bone_vertex_indices[bone_index], :]
			bone_vertices_4 = torch.cat([bone_vertices, ones], dim=1)
			bone_vertices_weighted = bone_vertices_4 * bone_vertex_weights[bone_index][:, np.newaxis] # We can apply the scalar weight now since any following matrix multiplications are commutative with scalar multiplication

			inverted = torch.matmul(bone_vertices_weighted, bone_matrices_world_inv[bone_index])

			vertices_bone_inverted[bone_index, bone_vertex_indices[bone_index], :] = inverted

		# Normalize bone weights so they guarantee to sum to 1
		weights = vertices_bone_inverted[:, :, 3]
		vertex_total_weight = weights.sum(dim=0)
		vertices_bone_inverted /= vertex_total_weight[np.newaxis, :, np.newaxis]

		self.register_buffer('vertices_bone_inverted', vertices_bone_inverted)

		# Need to make up a variable to represent an offset for the entire armature
		offset = torch.zeros(3, dtype=torch.float64)
		self.offset = torch.nn.Parameter(offset)

		# Need to create bone scale variables, but only enough to cover the bone_scale_share_map

		# The bone scale share map maps bone names to shared scale indices, we need it to map bone indices to shared scale indices
		self.bone_index_to_scale_index = {bone_name_to_index[name]: bone_scale_share_map[name] for name in bone_name_to_index.keys()}
		num_shared_bones = np.max(list(self.bone_index_to_scale_index.values())) + 1
		self.pose_bone_scales = torch.nn.Parameter(torch.ones((num_shared_bones, 3), dtype=torch.float64))

		# I need a scalar one for concatenation
		one = torch.ones(1, dtype=torch.float64, requires_grad=False)
		self.register_buffer('one', one)

		self.mesh = ColoredTriMesh(
			self.faces,
			faces_uv=self.faces_uvs,
			uv=self.uvs,
			texture=self.texture,
		)
		# self.mesh.set_vertices_colors(self.vertex_colors[:, :3])

		self.scene = Scene3DPytorch()
		self.scene.set_mesh(self.mesh)

		self.scene.light_directional = None
		self.scene.light_ambient = np.array(1, dtype=np.float64)

		self.using_ptrackers = False
		self.use_differential_rendering = use_differential_rendering

	def set_trackers(self, ptrackers_3d, ptrackers_bone_weights, wtrackers_source, wtrackers_bone_indices, use_wtrackers):
		self.using_ptrackers = ptrackers_3d.size > 0
		self.using_wtrackers = use_wtrackers

		ones = np.ones(wtrackers_source.shape[0], dtype=np.float64)
		wtrackers_source = np.concatenate([wtrackers_source, ones[:, np.newaxis]], axis=1)

		self.wtrackers_source = torch.tensor(wtrackers_source[:, np.newaxis, :], dtype=torch.float64, requires_grad=False)
		self.wtrackers_bone_indices = torch.tensor(wtrackers_bone_indices, dtype=torch.int64, requires_grad=False)
		self.wtrackers_scale = torch.tensor([1], dtype=torch.float64, requires_grad=True)

		if self.using_ptrackers:
			ptrackers_3d = torch.tensor(ptrackers_3d, dtype=torch.float64, requires_grad=False)
			ptrackers_bone_weights = torch.tensor(ptrackers_bone_weights, dtype=torch.float64, requires_grad=False)

			# The vertices_input array is the location of the vertex after the bone matrix world was applied (a.k.a. the rest pose).
			# We will need to apply bone_matrices_world_inv to a vertex now, so that when we later apply the pose bone transformation in world space,
			# we will be at the expected final vertex location for that bone. We can also apply the bone weights now, that way,
			# when we later apply each pose bone transformation, we can just take the sum to get the final position.
			ptrackers_3d_bone_inverted = torch.zeros((self.bone_matrices_world.shape[0], ptrackers_3d.shape[0], 4), dtype=torch.float64)
			for bone_index in range(self.bone_matrices_world.shape[0]):
				ones = torch.ones((ptrackers_3d.shape[0], 1), dtype=torch.float64)
				ptrackers_3d_4 = torch.cat([ptrackers_3d, ones], dim=1)
				ptrackers_3d_weighted = ptrackers_3d_4 * ptrackers_bone_weights[:, bone_index, np.newaxis]

				inverted = torch.matmul(ptrackers_3d_weighted, self.bone_matrices_world_inv[bone_index])

				ptrackers_3d_bone_inverted[bone_index, :, :] = inverted

			weights = ptrackers_3d_bone_inverted[:, :, 3]
			trackers_3d_total_weight = weights.sum(dim=0)
			ptrackers_3d_bone_inverted /= trackers_3d_total_weight[np.newaxis, :, np.newaxis]

			self.register_buffer('ptrackers_3d_bone_inverted', ptrackers_3d_bone_inverted)

	def set_background_color(self, background_color):
		self.scene.set_background_color(background_color)

	def set_projection_matrix(self, intrinsics, extrinsics, shape):
		self.camera = CameraPytorch(
			extrinsic=extrinsics,
			intrinsic=intrinsics,
			width=shape[1],
			height=shape[0],
			distortion=None,
		)

	def forward(
		self,
		combined_image,
		video_image,
		green_screen_image,
	):
		# Repackage all parameters and buffers
		repackage(self.pose_bone_quaternions, True)
		repackage(self.pose_bone_scales, True)
		repackage(self.offset, True)
		repackage(self.wtrackers_scale, True)

		repackage(self.vertices_bone_inverted, False)
		repackage(self.one, False)
		repackage(self.relative_pose_bone_translations, False)
		repackage(self.pose_bone_initial_scales, False)
		repackage(self.vertex_colors, False)

		if self.using_ptrackers:
			repackage(self.ptrackers_3d_bone_inverted, False)

		# Divide quaternions by their magnitude
		norm = torch.norm(self.pose_bone_quaternions, dim=1)
		pose_bone_quaternions_input = self.pose_bone_quaternions / norm[:, np.newaxis]

		# Create variables for storing this iteration's bone transformations
		pose_bone_transforms = [] # Blender requires scale to be applied differently, so keep a separate list of the transforms without the scaling applied
		pose_bone_transforms_scaled = []
		pose_bone_translations = [] # Assign this variable now for debugging, it will not be used in this calculation
		pose_bone_quaternions = []
		pose_bone_scales = []

		# Hand a reference of the bone transformation variables to the custom_transform_operations
		self.custom_transform_operations.start_iteration(
			pose_bone_quaternions,
			pose_bone_translations,
			pose_bone_scales,
			pose_bone_transforms,
			pose_bone_transforms_scaled,
		)

		for bone_index in range(pose_bone_quaternions_input.shape[0]):

			pose_bone_quaternion = pose_bone_quaternions_input[bone_index]
			relative_pose_bone_translation = self.relative_pose_bone_translations[bone_index]
			pose_bone_scale = self.pose_bone_scales[self.bone_index_to_scale_index[bone_index]] * self.pose_bone_initial_scales[bone_index]

			# Convert translation from parent space to world space
			if self.bone_parents[bone_index] == -1: # If there is no parent
				pose_bone_translation = relative_pose_bone_translation[:3] + self.offset # The relative translation is the same as the absolute world translation if there is no parent (but also include the offset)
				pose_bone_translations.append(torch.cat([pose_bone_translation, self.one], dim=0))
			else: # Otherwise convert translation relative to parent to absolute world translation
				parent_pose_bone_transform_scaled = pose_bone_transforms_scaled[self.bone_parents[bone_index]] # Because pose bones are ordered by DFS, pose_bone_transforms_scaled[self.bone_parents[bone_index]] is guaranteed to exist (since bone_index != -1)
				pose_bone_translation = torch.matmul(relative_pose_bone_translation[np.newaxis, :], parent_pose_bone_transform_scaled)[0] # Index 0 to remove broadcasting axis
				pose_bone_translations.append(pose_bone_translation)

			# Convert quaternion, translation, and scale to a matrix transform
			(
				pose_bone_quaternion,
				pose_bone_transform,
				pose_bone_transform_scaled,
			) = self.custom_transform_operations[self.bone_index_to_name[bone_index]](
				bone_index,
				self.bone_parents[bone_index],
				pose_bone_quaternion,
				pose_bone_translation,
				pose_bone_scale,
			)

			# Store intermediate results
			pose_bone_quaternions.append(pose_bone_quaternion) # Append quaternion now that it is in world space
			pose_bone_transforms.append(pose_bone_transform) # Index 0 to remove broadcasting axis
			pose_bone_transforms_scaled.append(pose_bone_transform_scaled)
			pose_bone_scales.append(pose_bone_scale)

		# Stack the lists into arrays
		pose_bone_transforms = torch.stack(pose_bone_transforms, dim=0)
		pose_bone_transforms_scaled = torch.stack(pose_bone_transforms_scaled, dim=0)
		pose_bone_translations = torch.stack(pose_bone_translations, dim=0)

		# Calculate the output vertices
		vertices_output = torch.matmul(self.vertices_bone_inverted, pose_bone_transforms_scaled)
		vertices_output = vertices_output.sum(dim=0)
		vertices_output = vertices_output[:, :3]

		# Calculate the output wtrackers
		wtrackers_output = torch.matmul(self.wtrackers_source * self.wtrackers_scale, pose_bone_transforms_scaled[self.wtrackers_bone_indices, :, :])
		wtrackers_output = wtrackers_output[:, 0, :3] # Remove broadcasting axis and remove ones

		# Calculate the output trackers
		ptrackers_3d_output = None
		if self.using_ptrackers:
			ptrackers_3d_output = torch.matmul(self.ptrackers_3d_bone_inverted, pose_bone_transforms_scaled)
			ptrackers_3d_output = ptrackers_3d_output.sum(dim=0)
			ptrackers_3d_output = ptrackers_3d_output[:, :3]

		self.mesh.set_vertices(vertices_output)

		return (
			self.mesh,
			ptrackers_3d_output,
			wtrackers_output,
			vertices_output,
			pose_bone_transforms_scaled,
			self.pose_bone_scales,
		)

class FeatureInfoRenderer():
	"""
	This class is used to gather information necessary for auto generating
	new tracker points. Specifically, this will take a set of features (possibly
	sift, AKAZE, or maybe generated by Blender) and ray cast them out to a
	fitted model. If the model is fitted well, we should be able to accurately
	find which polygon the feature should land on, and find the resting 3D
	position, as well as the bone weights for this new point using barycentric
	interpolation of the polygon's vertex data. Then we can match these 2D points
	to features found in the next frame, then we will have a large set of 2d
	projected coordinates and 3d resting coordinates that we can regress on.
	"""

	def __init__(
		self,
		ctx,
		wnd,
		vertices,
		tri_vertex_indices,
		dense_bone_weights,
		video_width,
		video_height,
	):
		# Copy arguments
		self.ctx = ctx
		self.wnd = wnd
		self.vertices = vertices
		self.dense_bone_weights = dense_bone_weights.T
		self.tri_vertex_indices = tri_vertex_indices
		self.video_width = video_width
		self.video_height = video_height

		# This shader will return 4 float32 values, the first 3 will be the barycentric weights of the tri vertices
		# mapped over range [0, 1]. The last will be the polygon index as a uint32 reinterpreted as a float32.
		self.barycentric_shader = self.ctx.program(
			vertex_shader='''
				#version 440
				// This is the vertex in screen space, I'm doing it this way because we already have the projected vertices from DEODR
				in vec3 vertex_uvz;

				// Since barycentric weights are calculated behind the scenes, we will need to hand each vertex in the tri a unique one-hot vec3.
				// Then we will sum this interpolated result of this vector in the fragment shader to get a vec3 of barycentric coordinates.
				in vec3 barycentric_weights_in;

				// Each vertex will recieve the polygon index it is a part of. When interpolated, it should remain the same in the fragment shader.
				in float polygon_index_in;

				// Passthrough to fragment
				out vec3 barycentric_weights;
				flat out float polygon_index;

				void main() {
					gl_Position = vec4(vertex_uvz, 1.0);

					barycentric_weights = barycentric_weights_in;
					polygon_index = polygon_index_in;
				}
			''',
			fragment_shader='''
				#version 440
				in vec3 barycentric_weights;
				flat in float polygon_index;

				out vec4 barycentric_weights_out;

				void main() {
					barycentric_weights_out = vec4(barycentric_weights, polygon_index);
				}
			''',
		)

		# Allocate a texture and framebuffer to communicate render results back to the cpu
		self.barycentric_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
		self.barycentric_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
		self.barycentric_shader_framebuffer = self.ctx.framebuffer(self.barycentric_shader_color_texture, self.barycentric_shader_depth_texture)

		# Create vertex attribute and buffers. Leave None for now since we won't have the data to fill them out until the render call
		self.barycentric_shader_vbo = None
		self.barycentric_shader_vao = None

		# Store barycentric weights for tris (unique one-hot for each vertex in a tri)
		self.vertex_barycentric_weights = np.zeros((self.tri_vertex_indices.shape[0], 3, 3), dtype=np.float32)
		self.vertex_barycentric_weights[:, 0, 0] = 1
		self.vertex_barycentric_weights[:, 1, 1] = 1
		self.vertex_barycentric_weights[:, 2, 2] = 1
		self.vertex_barycentric_weights = self.vertex_barycentric_weights.reshape(-1, 3) # Flatten into (number of vertices, 3)

		# Store polygon index for each vertex (arange repeated 3 times)
		self.vertex_polygon_indices_raw = np.arange(self.tri_vertex_indices.shape[0], dtype=np.uint32)
		self.vertex_polygon_indices_raw = np.repeat(self.vertex_polygon_indices_raw, 3)
		self.vertex_polygon_indices = self.vertex_polygon_indices_raw.view(dtype=np.float32) # Re-interpret as a float32 so we can pass it through glsl without losing data

		# Initialize akaze detector
		self.detector = cv2.AKAZE_create()

		# Create variables for keypoint data
		self.prev_keypoints = None # Keypoint data from the previous frame, these will be assigned from curr_ data at the end of the render call
		self.prev_keypoints_2d = None
		self.prev_keypoints_3d = None
		self.prev_descriptors = None
		self.prev_bone_weights = None

		self.curr_keypoints = None # Keypoint data from the current frame
		self.curr_keypoints_2d = None
		self.curr_keypoints_3d = None
		self.curr_descriptors = None
		self.curr_bone_weights = None

		self.matched_keypoints = None # Keypoint data that has a match between the previous frame and the current frame, these will be used for regression
		self.matched_keypoints_2d = None
		self.matched_keypoints_3d = None
		self.matched_descriptors = None
		self.matched_bone_weights = None

	def find_features(self, video_image, green_screen_image):
		# Blur the green screen image on the gpu
		green_screen_image_gpu = cp.array(green_screen_image) / 255
		blurred = cupyx.scipy.ndimage.gaussian_filter(green_screen_image_gpu, (2, 2, 0))
		green_screen_image_gpu = (green_screen_image_gpu * 255).astype(cp.uint8)

		green_screen_image = cp.asnumpy(green_screen_image_gpu)

		# Find akaze features
		keypoints, descriptors = get_keypoints_and_descriptors(video_image, self.detector)

		# Gather keypoint locations
		keypoints_xy = np.array([np.round(keypoint.pt) for keypoint in keypoints], dtype=np.uint32)

		# Filter out keypoints that lie chroma keyed pixels
		keypoints_color = green_screen_image[keypoints_xy[:, 1], keypoints_xy[:, 0], :]
		valid_mask = np.logical_not((keypoints_color > 128).any(axis=1))

		keypoints = keypoints[valid_mask]
		descriptors = descriptors[valid_mask]
		keypoints_xy = keypoints_xy[valid_mask]

		# Store keypoint information for matching and generating regression data
		self.curr_keypoints = keypoints
		self.curr_keypoints_2d = keypoints_xy
		self.curr_descriptors = descriptors

	def match_features_to_previous(self):
		if self.prev_keypoints is not None:
			(
				prev_matched_indices,
				curr_matched_indices,
			) = match_descriptors(
				(self.prev_keypoints, self.prev_descriptors),
				(self.curr_keypoints, self.curr_descriptors),
				return_indices=True,
				# images=(prev_image, image),
			)

			# Fill out the matched keypoints data
			self.matched_keypoints = self.curr_keypoints[curr_matched_indices]
			self.matched_keypoints_2d = self.curr_keypoints_2d[curr_matched_indices]
			self.matched_keypoints_3d = self.prev_keypoints_3d[prev_matched_indices]
			self.matched_descriptors = self.curr_descriptors[curr_matched_indices]
			self.matched_bone_weights = self.prev_bone_weights[prev_matched_indices]

	def render(self, vertices_uvz):
		# Unwrap vertex coordinates for each tri. We don't want to use an ibo since vertices_uvz used in different
		# polygons will have different attributes.
		vertices_uvz = vertices_uvz[self.tri_vertex_indices].reshape(-1, 3).astype(np.float32)

		# Interleave vertex coordinates with barycentric weights and polygon indices
		vertex_data = np.concatenate([vertices_uvz, self.vertex_barycentric_weights, self.vertex_polygon_indices[:, np.newaxis]], axis=1)

		# Update uv shader vbo with current frame tri_uvs_and_screen_uvs
		if self.barycentric_shader_vbo is None:
			self.barycentric_shader_vbo = self.ctx.buffer(vertex_data.tobytes())

			self.barycentric_shader_vao = self.ctx.vertex_array(
				self.barycentric_shader,
				[(self.barycentric_shader_vbo, '3f 3f 1f', 'vertex_uvz', 'barycentric_weights_in', 'polygon_index_in')],
			)
		else:
			self.barycentric_shader_vbo.write(vertex_data.tobytes())

		all_ones = np.uint32(0b11111111111111111111111111111111)
		all_ones_float = all_ones.view(dtype=np.float32)

		self.ctx.clear(0.0, 0.0, 0.0, 0.0)
		self.barycentric_shader_framebuffer.clear(all_ones_float, all_ones_float, all_ones_float, all_ones_float) # Defaulting to NaN (specifically all ones) so we can filter polygon indices on uint32 max value

		# Find pixel distance from the blender camera
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.enable(moderngl.DEPTH_TEST)
		self.barycentric_shader_framebuffer.use()
		self.barycentric_shader_vao.render()
		self.wnd.use()
		self.barycentric_shader_vao.render()

		# Read result to cpu
		raw = self.barycentric_shader_color_texture.read()
		cpu_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))

		raw = self.barycentric_shader_depth_texture.read()
		cpu_depth_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width))

		# Filter cpu image data for keypoints
		keypoints_ij = self.curr_keypoints_2d[:, ::-1] # Convert from x, y to i, j
		keypoint_data = cpu_image[keypoints_ij[:, 0], keypoints_ij[:, 1], :]

		# Split keypoint data into barycentric weights and polygon indices
		barycentric_weights = keypoint_data[:, :3]
		polygon_indices = keypoint_data[:, 3]

		polygon_indices = polygon_indices.view(dtype=np.uint32) # Re-interpret the float32 as uint32, this undoes the first re-interpret done in the constructor

		# cpu_image = cpu_image.copy()
		# for i in range(-2, 3):
		# 	for j in range(-2, 3):
		# 		cpu_image[keypoints_ij[:, 0] + i, keypoints_ij[:, 1] + j, :] = [[1, 0, 0, 1]]
		# cv2.imshow('cpu_image', cpu_image[::4, ::4, :3])
		# cv2.waitKey(0)

		# Filter out keypoints that didn't land on a polygon
		valid_mask = (polygon_indices != all_ones)
		polygon_indices = polygon_indices[valid_mask]
		barycentric_weights = barycentric_weights[valid_mask]
		self.curr_keypoints = self.curr_keypoints[valid_mask]
		self.curr_keypoints_2d = self.curr_keypoints_2d[valid_mask]
		self.curr_descriptors = self.curr_descriptors[valid_mask]

		keypoints_ij = keypoints_ij[valid_mask]

		# # We don't want to match features that are on the edge of the model because those aren't guaranteed to move with the limb they raycast to
		# # So let's find the edges by doing a high pass filter on the depth image, blurring it, thresholding it, then using that as a mask
		# gpu_depth_image = cp.array(cpu_depth_image) # Hand depth information back to the gpu as a CUDA array
		# high_pass_kernel = cp.array([
		# 	[-1, -1, -1],
		# 	[-1,  8, -1],
		# 	[-1, -1, -1],
		# ], dtype=cp.float32)
		# high_pass_response = cupyx.scipy.signal.convolve2d(gpu_depth_image, high_pass_kernel, mode='same')
		# high_pass_response = (high_pass_response > 0.01).astype(cp.float32)
		# high_pass_response = cupyx.scipy.ndimage.gaussian_filter(high_pass_response, (10, 10)) # TODO: Config this
		# high_pass_response = cp.asnumpy(high_pass_response > 0.01)

		# cv2.imwrite(r'C:\Users\William\Desktop\hpr.png', high_pass_response * 255.0)

		# keypoints_high_pass = high_pass_response[keypoints_ij[:, 0], keypoints_ij[:, 1]]
		# valid_mask = keypoints_high_pass == False

		# polygon_indices = polygon_indices[valid_mask]
		# barycentric_weights = barycentric_weights[valid_mask]
		# self.curr_keypoints = self.curr_keypoints[valid_mask]
		# self.curr_keypoints_2d = self.curr_keypoints_2d[valid_mask]
		# self.curr_descriptors = self.curr_descriptors[valid_mask]

		# Find the rest position of each keypoint
		vertex_indices = self.tri_vertex_indices[polygon_indices]
		polygon_vertices = self.vertices[vertex_indices, :] # Resulting should have shape (num polygons, num vertices (3 because tris), 3)
		self.curr_keypoints_3d = (polygon_vertices * barycentric_weights[:, :, np.newaxis]).sum(axis=1)

		# Find the bone weights of each keypoint
		vertex_bone_weights = self.dense_bone_weights[vertex_indices]
		self.curr_bone_weights = (vertex_bone_weights * barycentric_weights[:, :, np.newaxis]).sum(axis=1)

		# Shift curr_ data to prev_
		self.prev_keypoints = self.curr_keypoints
		self.prev_keypoints_2d = self.curr_keypoints_2d
		self.prev_keypoints_3d = self.curr_keypoints_3d
		self.prev_descriptors = self.curr_descriptors
		self.prev_bone_weights = self.curr_bone_weights

		# Invalidate the curr_ data
		self.curr_keypoints = None
		self.curr_keypoints_2d = None
		self.curr_keypoints_3d = None
		self.curr_descriptors = None
		self.curr_bone_weights = None

class CommunicationHandler(moderngl_window.WindowConfig):
	gl_version = (4, 4)

	title = 'CommunicationHandler'
	resizable = True
	aspect_ratio = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		# Read subprocess args
		self.subprocess_args = read_pickled()

		# Default the armature layer to None
		self.armature_layer = None

		# Create a variable for the feature info renderere, it will be initialized if needed
		self.feature_info_renderer = None

	def render(self, time, frametime):
		if not self.handle_stdin():
			self.wnd.close()

	def get_next_frame(self, video_capture, green_screen_capture, transpose, scale):

		_, video_image = video_capture.read()
		_, green_screen_image = green_screen_capture.read()

		if transpose[0]:
			video_image = cv2.transpose(video_image)
			green_screen_image = cv2.transpose(green_screen_image)
		if transpose[1]:
			video_image = cv2.flip(video_image, -1)
			green_screen_image = cv2.flip(green_screen_image, -1)
		if transpose[2]:
			video_image = cv2.flip(video_image, 0)
			green_screen_image = cv2.flip(green_screen_image, 0)


		# Overlay video image with green screen contents where present
		combined_image = video_image.copy()
		chroma_mask = (green_screen_image > 128).any(axis=2) # Fuzzy compare for if any color is present
		combined_image[chroma_mask, :] = green_screen_image[chroma_mask, :]

		# Downscale images
		image_shape = np.array(video_image.shape, dtype=np.float64)
		image_shape *= scale
		image_shape = image_shape.round().astype(np.int32)
		image_shape = (image_shape[1], image_shape[0])

		video_image_ds = cv2.resize(video_image, image_shape, cv2.INTER_NEAREST)
		green_screen_image_ds = cv2.resize(green_screen_image, image_shape, cv2.INTER_NEAREST)
		combined_image_ds = cv2.resize(combined_image, image_shape, cv2.INTER_NEAREST)

		# Flip rgb, cast to float64
		video_image = video_image[:, :, ::-1].astype(np.float64) / 255
		green_screen_image = green_screen_image[:, :, ::-1].astype(np.float64) / 255
		combined_image = combined_image[:, :, ::-1].astype(np.float64) / 255
		video_image_ds = video_image_ds[:, :, ::-1].astype(np.float64) / 255
		green_screen_image_ds = green_screen_image_ds[:, :, ::-1].astype(np.float64) / 255
		combined_image_ds = combined_image_ds[:, :, ::-1].astype(np.float64) / 255

		return (combined_image_ds, video_image_ds, green_screen_image_ds), (combined_image, video_image, green_screen_image)

	def handle_stdin(self):
		# Unpack subprocess args
		(
			bone_name_to_index,
			bone_scale_share_map,
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
			video_path,
			video_green_screen_path,
			video_crop_transpose,
		) = self.subprocess_args

		# Get additional arguments for this frame
		read_result = read_pickled()
		if read_result is None: return False # Break from calling function loop

		(
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
		) = read_result

		if (frame == -1): # Quit if we have reached the frame padding used when dividing frames upon workers
			return False

		# Store the length of the trackers array because we may eventually modify these arrays by appending
		# additional generated trackers, but we want the original trackers to have a higher weight.
		num_user_ptrackers = ptrackers_3d.shape[0]

		# Grab the current frame (TODO: No need to recapture and release the video every time, frame indices should be sequential)
		video_capture = start_video_at_frame(video_path, frame)
		green_screen_capture = start_video_at_frame(video_green_screen_path, frame)
		(
			(combined_image, video_image, green_screen_image), # Downscaled images
			(combined_image_hd, video_image_hd, green_screen_image_hd), # Original high def versions of the images
		) = self.get_next_frame(video_capture, green_screen_capture, video_crop_transpose, quality_scale)
		video_shape = np.array(video_image.shape[:2], dtype=np.uint32)
		video_shape_hd = np.array(video_image_hd.shape[:2], dtype=np.uint32)

		video_capture.release()
		green_screen_capture.release()

		# Create an armature layer if needed, or reuse the old one
		if not reference_last_frame or self.armature_layer is None:
			armature_layer = ArmatureLayer(
				vertices,
				vertex_colors,
				polygons,
				faces_uvs,
				uvs,
				texture,

				bone_name_to_index,
				bone_scale_share_map,
				bone_parents,
				bone_quaternions,
				bone_translations,
				pose_bone_quaternions,
				pose_bone_translations,
				pose_bone_initial_scales,
				bone_vertex_indices,
				bone_vertex_weights,

				use_differential_rendering,
			)
		else:
			armature_layer = self.armature_layer

		# Create tracker and feature information

		if generate_features and self.feature_info_renderer is None: # Initilialize the feature info renderer if we don't already have it
			self.feature_info_renderer = FeatureInfoRenderer(
				self.ctx,
				self.wnd,
				vertices,
				polygons,
				dense_bone_weights,
				video_shape_hd[1],
				video_shape_hd[0],
			)

		if match_features or generate_features:
			self.feature_info_renderer.find_features(video_image_hd, green_screen_image_hd)

		if match_features:
			self.feature_info_renderer.match_features_to_previous()

			# Concatenate the matched features to the ptrackers_3d data
			if self.feature_info_renderer.matched_keypoints_3d is not None:
				ptrackers_3d = np.concatenate([ptrackers_3d, self.feature_info_renderer.matched_keypoints_3d], axis=0)
				ptrackers_bone_weights = np.concatenate([ptrackers_bone_weights, self.feature_info_renderer.matched_bone_weights], axis=0)

		if ptrackers_2d.size > 0:
			ptrackers_2d[:, 1] = 1 - ptrackers_2d[:, 1] # Convert trackers 2d from x, y on range [0, 1] with origin in top left to x, y on range[0, width or height] with origin in bottom left
			ptrackers_2d *= video_shape[np.newaxis, ::-1]

		if generate_features:
			# Concatenate the matched features to the ptrackers_2d data
			if self.feature_info_renderer.matched_keypoints_2d is not None:
				features_2d = self.feature_info_renderer.matched_keypoints_2d / video_shape_hd[::-1] * video_shape[::-1] # Convert from full scale image to downscaled image coordinates
				ptrackers_2d = np.concatenate([ptrackers_2d, features_2d], axis=0)

		ptrackers_2d = torch.tensor(ptrackers_2d, dtype=torch.float64, requires_grad=False)
		wtrackers_destination = torch.tensor(wtrackers_destination, dtype=torch.float64, requires_grad=False)

		# Update the armature layer with the current frame data

		armature_layer.set_trackers(
			ptrackers_3d,
			ptrackers_bone_weights,
			wtrackers_source,
			wtrackers_bone_indices,
			use_wtrackers,
		)

		self.armature_layer = armature_layer

		armature_layer.set_projection_matrix(
			intrinsics,
			extrinsics,
			video_image.shape,
		)

		armature_layer.set_background_color(
			np.array([0, 1, 0])
		)


		# Initialize green screen variables for our loss function
		green_screen_blurred = cv2.GaussianBlur(green_screen_image, (0, 0), 10, 10)
		green_screen_blurred_torch = torch.tensor(green_screen_blurred, dtype=torch.float64, requires_grad=False)
		green_screen_blurred_torch_gpu = green_screen_blurred_torch
		green_mask = torch.tensor(green_screen_image[:, :, 1] > 0.5, dtype=torch.bool, requires_grad=False) # Fuzzy compare if pixel is green
		green = torch.zeros_like(green_screen_blurred_torch)
		green[:, :, 1] = green_mask
		# green = torch.tensor([[[0, 1, 0]]], dtype=torch.float64, requires_grad=False)
		blue = torch.tensor([[[0, 0, 1]]], dtype=torch.float64, requires_grad=False)

		# Initialize optimizer and scheduler
		optimizer = torch.optim.Adam(armature_layer.parameters(), 3e-3, betas=(0.9, 0.9))
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

		blurred_combined_image = combined_image
		combined_image_torch = torch.tensor(combined_image, dtype=torch.float64, requires_grad=False)
		combined_image_torch_gpu = combined_image_torch
		not_red_or_blue_mask = np.logical_and(green_screen_image[:, :, 2] < 0.1, green_screen_image[:, :, 0] < 0.1)
		not_red_or_blue_mask = torch.tensor(not_red_or_blue_mask, dtype=torch.bool, requires_grad=False) #  Fuzzy compare if pixel is red or blue
		not_red_or_blue_mask_gpu = not_red_or_blue_mask

		niter = 0
		bailout_iterations = 0
		max_bailout = max(500, num_iterations)
		quit_keypoint_optimization = int(num_iterations * 0.8)
		while niter < num_iterations and bailout_iterations < max_bailout:

			(
				mesh,
				ptrackers_3d_output,
				wtrackers_output,
				vertices_output,
				pose_bone_transforms,
				pose_bone_scales,
			) = armature_layer(combined_image, video_image, green_screen_image)

			render = armature_layer.scene.render(armature_layer.camera)
			render_gpu = render

			# Sum of squared error between render and image, this is done so we can display this result to the user
			sse_image = ((render_gpu - combined_image_torch_gpu) ** 2).sum(dim=2)
			sse_image *= not_red_or_blue_mask_gpu # gloves shouldn't count towards loss because we don't include hands in the 3d model

			loss = sse_image.cpu().mean() * use_differential_rendering # Store a default loss
			base_image_loss = float(loss) # Store the loss just from the image difference. That way we have a consistent loss to compare against even as we enable or disable other loss components


			# blue_mask = render_gpu[:, :, 2] * (render_gpu[:, :, 2] > 0.9)
			# blue_loss = (blue_mask * green_screen_blurred_torch_gpu[:, :, 2])
			# loss -= blue_loss.cpu().mean() * 100

			# red_mask = render_gpu[:, :, 0] * (render_gpu[:, :, 0] > 0.9)
			# red_loss = (red_mask * green_screen_blurred_torch_gpu[:, :, 0])
			# loss -= red_loss.cpu().mean() * 100

			# green_sse_image = ((render - green) ** 2).sum(dim=2) * 10 # Add an additional penalty for overlapping the green screen
			# green_sse_image *= not_red_or_blue_mask_gpu # gloves shouldn't count towards loss because we don't include hands in the 3d model
			# loss += green_sse_image.mean()

			# Increase the loss if the pose bone scales aren't close to 1
			for i in range(armature_layer.pose_bone_quaternions.shape[0]):
				final_pose_bone_scale = pose_bone_scales[armature_layer.bone_index_to_scale_index[i]] * armature_layer.pose_bone_initial_scales[i]
				diff = final_pose_bone_scale - 1
				_loss = diff**2
				loss += _loss[0]*0.1 + _loss[1]*0.01 + _loss[2]*0.1 # Scale in the x z is more important than in the y
				loss += (diff[0] - diff[2])**2 # Should try to match x and z scale


			# Calculate ptracker position in screen space
			if armature_layer.using_ptrackers and bailout_iterations < quit_keypoint_optimization:
				ptrackers_3d_camera = armature_layer.camera.world_to_camera(ptrackers_3d_output)
				ptrackers_3d_depths = ptrackers_3d_camera[:, 2]
				ptrackers_2d_normalized = ptrackers_3d_camera[:, :2] / ptrackers_3d_depths[:, np.newaxis]
				ptrackers_2d_output = armature_layer.camera.left_mul_intrinsic(ptrackers_2d_normalized)

				ptrackers_2d_delta = (ptrackers_2d_output - ptrackers_2d) / video_shape[0] # Divide by image height so that both x and y have the same scale and the scale is similar between images of different resolutions
				ptrackers_se = (ptrackers_2d_delta ** 2).mean(dim=1)
				# trackers_se_output = ptrackers_se.detach().cpu().numpy()

				# Weigh the user defined trackers more
				loss += (ptrackers_se[:num_user_ptrackers] * 100).mean()
				loss += (ptrackers_se[num_user_ptrackers:] * 300).mean()


			if armature_layer.using_wtrackers and bailout_iterations < quit_keypoint_optimization:
				wtrackers_3d_se = ((wtrackers_output - wtrackers_destination) ** 2).sum(dim=1)
				wtracker_mse = wtrackers_3d_se.mean()
				loss += wtracker_mse * 0.01


			# Perform optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# scheduler.step()

			# Display progress
			if bailout_iterations % 5 == 0:
				if armature_layer.using_wtrackers and bailout_iterations < quit_keypoint_optimization:
					wtrackers_3d_camera = armature_layer.camera.world_to_camera(wtrackers_output)
					wtrackers_3d_depths = wtrackers_3d_camera[:, 2]
					wtrackers_2d_normalized = wtrackers_3d_camera[:, :2] / wtrackers_3d_depths[:, np.newaxis]
					wtrackers_2d_output = armature_layer.camera.left_mul_intrinsic(wtrackers_2d_normalized)

					wtrackers_3d_camera = armature_layer.camera.world_to_camera(wtrackers_destination)
					wtrackers_3d_depths = wtrackers_3d_camera[:, 2]
					wtrackers_2d_normalized = wtrackers_3d_camera[:, :2] / wtrackers_3d_depths[:, np.newaxis]
					wtrackers_2d = armature_layer.camera.left_mul_intrinsic(wtrackers_2d_normalized)

				render = render.detach().cpu().numpy()
				sse_image = sse_image.detach().cpu().numpy()

				display_image = np.column_stack(
					(blurred_combined_image, render, np.tile(sse_image[:, :, None], (1, 1, 3)))
				)

				if armature_layer.using_ptrackers and bailout_iterations < quit_keypoint_optimization:
					_ptrackers_2d_screen_output = ptrackers_2d_output.detach().cpu().numpy().round().astype(np.int64)
					_ptrackers_2d_screen = ptrackers_2d.detach().cpu().numpy().round().astype(np.int64)

					# min_se = trackers_se_output.min()
					# range_se = trackers_se_output.max() - min_se

					_ptrackers_2d_screen_output[:, 0] += blurred_combined_image.shape[1]
					for i in range(_ptrackers_2d_screen.shape[0]):
						# color = int((trackers_se_output[i] - min_se) / range_se * 255)
						color = 255
						cv2.circle(display_image, tuple(_ptrackers_2d_screen[i, :]), 3, (color, 0, color), -1)
						cv2.circle(display_image, tuple(_ptrackers_2d_screen_output[i, :]), 3, (color, 0, color), -1)

				if armature_layer.using_wtrackers and bailout_iterations < quit_keypoint_optimization:
					_wtrackers_2d_screen_output = wtrackers_2d_output.detach().cpu().numpy().round().astype(np.int64)
					_wtrackers_2d_screen = wtrackers_2d.detach().cpu().numpy().round().astype(np.int64)

					_wtrackers_2d_screen_output[:, 0] += blurred_combined_image.shape[1]
					for i in range(_wtrackers_2d_screen_output.shape[0]):
						# color = int((trackers_se_output[i] - min_se) / range_se * 255)
						color = 255
						cv2.circle(display_image, tuple(_wtrackers_2d_screen[i, :]), 3, (0, color, color), -1)
						cv2.circle(display_image, tuple(_wtrackers_2d_screen_output[i, :]), 3, (0, color, color), -1)


				width = display_image.shape[1]
				height = display_image.shape[0]
				target_width = 1000
				target_height = int(target_width / width * height)

				display_image = cv2.resize(display_image, (target_width, target_height))

				cv2.imshow(
					"animation",
					display_image[:, :, ::-1],
				)

				cv2.waitKey(1)

			# Only incremeent niter if we are starting to do okay
			niter += base_image_loss < 0.33
			bailout_iterations += 1

		pose_bone_transforms = pose_bone_transforms.detach().numpy()
		pose_bone_scales = pose_bone_scales.detach().numpy()

		if generate_features:
			vertices_camera = armature_layer.camera.world_to_camera(vertices_output)
			vertices_depths = vertices_camera[:, 2]
			vertices_normalized = vertices_camera[:, :2] / vertices_depths[:, np.newaxis]
			vertices_screen = armature_layer.camera.left_mul_intrinsic(vertices_normalized).detach().numpy()
			vertices_depths = vertices_depths / vertices_depths.max() # Normalize to range 0 to 1

			vertices_uv = (vertices_screen / video_shape[::-1] - 0.5) * 2

			vertices_uvz = np.concatenate([vertices_uv, vertices_depths.detach().numpy()[:, np.newaxis]], axis=1)

			self.feature_info_renderer.render(vertices_uvz)

		# Pickle generated matrices and send over stdout
		result = (
			pose_bone_transforms,
			pose_bone_scales,
			base_image_loss,
		)
		write_pickled(result)

		return True # Continue looping in calling function

if __name__ == '__main__':
	import os

	CommunicationHandler.run()
