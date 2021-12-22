import pickle
import struct
import os
import cv2
import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import soft_renderer as sr

from deodr import LaplacianRigidEnergy
from deodr.tensorflow import CameraTensorflow, LaplacianRigidEnergyTensorflow, Scene3DTensorflow
from deodr.tensorflow.triangulated_mesh_tensorflow import ColoredTriMeshTensorflow as ColoredTriMesh
from deodr.tensorflow.triangulated_mesh_tensorflow import TriMeshTensorflow as TriMesh
from deodr.tensorflow.mesh_fitter_tensorflow import qrot


from helper_functions import print, start_video_at_frame, read_pickled, write_pickled, quaternion_to_mat4
def _quaternion_to_mat4(quaternion, translation, scale=None): # Create a wrapper function so I don't have to hand in tf all the time
	return quaternion_to_mat4(quaternion, translation, torch, scale=scale)

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
	):
		super(ArmatureLayer, self).__init__()

		# Convert inputs to torch tensors
		vertices_input = torch.tensor(vertices_input, dtype=torch.float64, requires_grad=False)
		vertex_colors = torch.tensor(vertex_colors, dtype=torch.float64, requires_grad=False)
		faces = torch.tensor(faces, dtype=torch.int64, requires_grad=False)
		bone_quaternions = torch.tensor(bone_quaternions, dtype=torch.float64, requires_grad=False)
		bone_translations = torch.tensor(bone_translations, dtype=torch.float64, requires_grad=False)
		bone_parents = torch.tensor(bone_parents, dtype=torch.int64, requires_grad=False)
		pose_bone_quaternions = torch.tensor(pose_bone_quaternions, dtype=torch.float64, requires_grad=True)
		pose_bone_translations = torch.tensor(pose_bone_translations, dtype=torch.float64, requires_grad=True)
		pose_bone_initial_scales = torch.tensor(pose_bone_initial_scales, dtype=torch.float64, requires_grad=False)

		bone_vertex_indices = [torch.tensor(value, dtype=torch.int64, requires_grad=False) for value in bone_vertex_indices]
		bone_vertex_weights = [torch.tensor(value, dtype=torch.float64, requires_grad=False) for value in bone_vertex_weights]

		# Register buffers and parameters
		self.register_buffer('vertex_colors', vertex_colors)
		self.register_buffer('faces', faces)
		self.register_buffer('pose_bone_initial_scales', pose_bone_initial_scales)
		self.register_buffer('bone_parents', bone_parents)

		self.pose_bone_quaternions = torch.nn.Parameter(pose_bone_quaternions)

		# Gather information necessary for calculating / recalculating vertex poses during descent step
		bone_matrices_world = _quaternion_to_mat4(bone_quaternions, bone_translations)
		bone_matrices_world_inv = torch.inverse(bone_matrices_world)

		self.register_buffer('bone_matrices_world', bone_matrices_world)

		_, pose_bone_matrices_world = _quaternion_to_mat4(pose_bone_quaternions, pose_bone_translations, pose_bone_initial_scales)
		pose_bone_matrices_world_inv = torch.inverse(pose_bone_matrices_world)

		# Find the bone head position relative to the bone's parent
		num_bones = pose_bone_translations.shape[0]
		relative_pose_bone_translations = torch.ones((num_bones, 4), dtype=torch.float64)
		relative_pose_bone_translations[:, :3] = pose_bone_translations # Default to absolute world translation, this is used in the case a bone has no parent

		has_parent = self.bone_parents != -1 # Mask of which bones have parents
		valid_parent_indices = self.bone_parents[has_parent] # Indices of parents for bones that have parents

		parent_pose_bone_matrices_inv = pose_bone_matrices_world_inv[valid_parent_indices, :, :] # Bones that have parents need their translations transformed by the parent inverse

		# BIG IFFY
		temp = torch.matmul(relative_pose_bone_translations[has_parent, np.newaxis, :], parent_pose_bone_matrices_inv) # Multiply each bone translation by the inverse of the parent matrix
		relative_pose_bone_translations[has_parent, :] = temp[:, 0, :] # Remove the broadcasting axis

		self.register_buffer('relative_pose_bone_translations', relative_pose_bone_translations)

		# The vertices_input array is the location of the vertex after the bone matrix world was applied.
		# So if we apply bone_matrices_world_inv to a vertex, then apply the pose_bone_quaternion / translation,
		# we will be at the expected final vertex location.
		vertices_bone_inverted = torch.zeros((len(bone_vertex_indices), vertices_input.shape[0], 4), dtype=torch.float64)
		for bone_index in range(bone_matrices_world.shape[0]):

			ones = torch.ones((bone_vertex_indices[bone_index].shape[0], 1), dtype=torch.float64)
			bone_vertices = vertices_input[bone_vertex_indices[bone_index], :]
			bone_vertices_4 = torch.cat([bone_vertices, ones], dim=1)
			bone_vertices_weighed = bone_vertices_4 * bone_vertex_weights[bone_index][:, np.newaxis] # We can apply the scalar weight now since any following matrix multiplications are commutative with scalar multiplication

			inverted = torch.matmul(bone_vertices_weighed, bone_matrices_world_inv[bone_index])

			# ANOTHER IFFY
			vertices_bone_inverted[bone_index, bone_vertex_indices[bone_index], :] = inverted

		# Normalize bone weights so they guarantee to sum to 1
		weights = vertices_bone_inverted[:, :, 3]
		vertex_total_weight = weights.sum(dim=0)
		vertices_bone_inverted /= vertex_total_weight[np.newaxis, :, np.newaxis]

		# self.vertices_bone_inverted = tf.constant(vertices_bone_inverted, tf.float64)
		self.register_buffer('vertices_bone_inverted', vertices_bone_inverted)

		# Need to make up a variable to represent an offset for the entire armature
		offset = torch.zeros(3, dtype=torch.float64)
		self.offset = torch.nn.Parameter(offset)

		# Need to create bone scale variables, but only enough to cover the bone_scale_share_map

		# The bone scale share map maps bone names to shared scale indices, we need it to map bone indices to shared scale indices
		self.bone_index_to_scale_index = {bone_name_to_index[name]: bone_scale_share_map[name] for name in bone_name_to_index.keys()}
		num_shared_bones = np.max(list(self.bone_index_to_scale_index.values())) + 1
		self.pose_bone_scales = torch.nn.Parameter(torch.ones((num_shared_bones, 3), dtype=torch.float64))

		# Identify trainable variables
		# self.trainable_variables = [
		# 	self.offset,
		# 	self.pose_bone_quaternions,
		# 	self.pose_bone_scales,
		# ]

		# boundaries = [10, 20]
		# values = [1e-3, 1e-2, 1e-3]
		# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

		# self.optimizer = tf.keras.optimizers.Adam(
		# 	learning_rate=6e-3,
		# 	beta_1=0.9,
		# 	beta_2=0.9,
		# 	epsilon=1e-07,
		# 	amsgrad=False,
		# )

		# I need a scalar one for concatenation
		one = torch.ones(1, dtype=torch.float64, requires_grad=False)
		self.register_buffer('one', one)

	def forward(
		self,
		combined_image,
		video_image,
		green_screen_image,
		chroma_mask,
	):
		# Repackage all parameters and buffers
		repackage(self.pose_bone_quaternions, True)
		repackage(self.pose_bone_scales, True)
		repackage(self.offset, True)

		repackage(self.vertices_bone_inverted, False)
		repackage(self.one, False)
		repackage(self.relative_pose_bone_translations, False)
		repackage(self.pose_bone_initial_scales, False)
		repackage(self.vertex_colors, False)

		# Divide quaternions by their magnitude
		norm = torch.norm(self.pose_bone_quaternions, dim=1)
		pose_bone_quaternions = self.pose_bone_quaternions / norm[:, np.newaxis]

		# Recursively modify translation, but use world rotation
		pose_bone_transforms = []
		pose_bone_transforms_scaled = []
		pose_bone_translations = [] # Assign this variable now for debugging, it will not be used in this calculation
		for i in range(pose_bone_quaternions.shape[0]):

			pose_bone_quaternion = pose_bone_quaternions[i]
			relative_pose_bone_translation = self.relative_pose_bone_translations[i]
			pose_bone_scale = self.pose_bone_scales[self.bone_index_to_scale_index[i]] * self.pose_bone_initial_scales[i]

			if self.bone_parents[i] == -1: # If there is no parent
				pose_bone_translation = relative_pose_bone_translation[:3] + self.offset # The relative translation is the same as the absolute world translation (but also include the offset)
				pose_bone_translations.append(torch.cat([pose_bone_translation, self.one], dim=0))
			else: # Otherwise convert translation relative to parent to absolute world translation
				parent_pose_bone_transform_scaled = pose_bone_transforms_scaled[self.bone_parents[i]] # Because pose bones are ordered by DFS, pose_bone_transforms[self.bone_parents[i]] is guaranteed to exist (if not -1)
				pose_bone_translation = torch.matmul(relative_pose_bone_translation[np.newaxis, :], parent_pose_bone_transform_scaled)[0] # Index 0 to remove broadcasting axis
				pose_bone_translations.append(pose_bone_translation)

			pose_bone_transform, pose_bone_transform_scaled = _quaternion_to_mat4(pose_bone_quaternion[np.newaxis, :], pose_bone_translation[np.newaxis, :], pose_bone_scale[np.newaxis, :])
			pose_bone_transforms.append(pose_bone_transform[0, :, :]) # Index 0 to remove broadcasting axis
			pose_bone_transforms_scaled.append(pose_bone_transform[0, :, :])

		# Stack the arrays
		pose_bone_transforms = torch.stack(pose_bone_transforms, dim=0)
		pose_bone_transforms_scaled = torch.stack(pose_bone_transforms_scaled, dim=0)
		pose_bone_translations = torch.stack(pose_bone_translations, dim=0)

		# Calculate the output vertices
		vertices_output = torch.matmul(self.vertices_bone_inverted, pose_bone_transforms_scaled)
		vertices_output = vertices_output.sum(dim=0)
		vertices_output = vertices_output[:, :3]

		return (
			sr.Mesh(
				vertices_output[np.newaxis, :, :].type(torch.cuda.FloatTensor),
				self.faces[np.newaxis, :, :].type(torch.cuda.IntTensor),
				textures=self.vertex_colors[np.newaxis, :, :3].type(torch.cuda.FloatTensor),
				texture_type='vertex',
			),
			pose_bone_transforms,
			self.pose_bone_scales,
		)

		# View from 90 degrees about z axis, so we can check if depth is correct
		# vertices_output -= self.relative_pose_bone_translations[0, :3]
		# vertices_output = tf.stack([vertices_output[:, 1], -vertices_output[:, 0], vertices_output[:, 2]])
		# vertices_output = tf.transpose(vertices_output)
		# vertices_output += self.relative_pose_bone_translations[0, :3]


		# self.mesh.set_vertices(vertices_output)
		# render = self.scene.render(self.camera)

		# mse_image = tf.math.reduce_sum((render - tf.constant(combined_image)) ** 2, axis=2) # Mean squared error between render and image, doing reduce sum only on axis 2 so we can display this result to the user

		# green_mask = (green_screen_image[:, :, 1] > 0.5) # Fuzzy compare if pixel is green
		# should_be_green = tf.boolean_mask(render, green_mask)
		# green_screen_loss = tf.math.reduce_sum((should_be_green - [[[0, 1, 0]]]) ** 2) # Add an additional penalty for overlapping the green screen

		# loss = tf.reduce_sum(mse_image) + green_screen_loss * 2


		# 	grads = tape.gradient(loss, self.trainable_variables)
		# 	self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		# return render, mse_image, pose_bone_transforms, self.pose_bone_scales


class InstrinsicExtrinsicTransform(sr.transform.Transform):
	def __init__(self, P, video_shape, dist_coeffs=None):
		super().__init__()
		'''
		Calculate projective transformation of vertices given a projection matrix
		P: 3x4 projection matrix
		dist_coeffs: vector of distortion coefficients
		orig_size: original size of image captured by the camera
		'''

		self.P = P
		self.dist_coeffs = dist_coeffs

		if isinstance(self.P, np.ndarray):
			self.P = torch.from_numpy(self.P).cuda()
		if self.P is None or self.P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
			raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
		if dist_coeffs is None:
			self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.P.shape[0], 1)

		self.video_shape = video_shape

	def transform(self, vertices):
		vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
		vertices = torch.bmm(vertices, self.P.transpose(2, 1))
		x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
		x_ = x / (z + 1e-5)
		y_ = y / (z + 1e-5)

		# Get distortion coefficients from vector
		k1 = self.dist_coeffs[:, None, 0]
		k2 = self.dist_coeffs[:, None, 1]
		p1 = self.dist_coeffs[:, None, 2]
		p2 = self.dist_coeffs[:, None, 3]
		k3 = self.dist_coeffs[:, None, 4]

		# we use x_ for x' and x__ for x'' etc.
		r = torch.sqrt(x_ ** 2 + y_ ** 2)
		x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
		y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 * (r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
		x__ = 2 * (x__ - self.video_shape[1] / 2.) / self.video_shape[1]
		y__ = 2 * (y__ - self.video_shape[0] / 2.) / self.video_shape[0]
		vertices = torch.stack([x__, -y__, z], dim=-1)
		return vertices


def get_next_frame(video_capture, green_screen_capture, transpose, scale):

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

	# Downscale images
	image_shape = np.array(video_image.shape, dtype=np.float64)
	image_shape *= scale
	image_shape = image_shape.round().astype(np.int32)
	image_shape = (image_shape[1], image_shape[0])

	video_image = cv2.resize(video_image, image_shape, cv2.INTER_NEAREST)
	green_screen_image = cv2.resize(green_screen_image, image_shape, cv2.INTER_NEAREST)

	# Overlay video image with green screen contents where present
	combined_image = video_image.copy()
	chroma_mask = (green_screen_image > 128).any(axis=2) # Fuzzy compare for if any color is present
	combined_image[chroma_mask, :] = green_screen_image[chroma_mask, :]

	# Flip rgb, cast to float64
	video_image = video_image[:, :, ::-1].astype(np.float64) / 255
	green_screen_image = green_screen_image[:, :, ::-1].astype(np.float64) / 255
	combined_image = combined_image[:, :, ::-1].astype(np.float64) / 255

	return combined_image, video_image, green_screen_image, chroma_mask

def handle_stdin(subprocess_args):
	# Unpack subprocess args
	(
		bone_name_to_index,
		bone_scale_share_map,
		bone_parents,
		bone_quaternions,
		bone_translations,
		bone_vertex_indices,
		bone_vertex_weights,
		vertices,
		vertex_colors,
		polygons,
		video_path,
		video_green_screen_path,
		video_crop_transpose,
	) = subprocess_args

	# Get additional arguments for this frame
	read_result = read_pickled()
	if read_result is None: return False # Break from calling function loop

	(
		intrinsics,
		extrinsics,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		quality_scale,
		frame,
	) = read_result

	# Grab the current frame (TODO: No need to recapture and release the video every time, frame indices should be sequential)
	video_capture = start_video_at_frame(video_path, frame)
	green_screen_capture = start_video_at_frame(video_green_screen_path, frame)
	combined_image, video_image, green_screen_image, chroma_mask = get_next_frame(video_capture, green_screen_capture, video_crop_transpose, quality_scale)
	video_capture.release()
	green_screen_capture.release()

	video_shape = video_image.shape

	# Images need to be square
	image_size = 512
	combined_image = cv2.resize(combined_image, (image_size, image_size))
	video_image = cv2.resize(video_image, (image_size, image_size))
	green_screen_image = cv2.resize(green_screen_image, (image_size, image_size))

	armature_layer = ArmatureLayer(
		vertices,
		vertex_colors,
		polygons,

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
	)

	transform = InstrinsicExtrinsicTransform(
		intrinsics.dot(extrinsics)[np.newaxis, :, :].astype(np.float32),
		video_shape,
	)

	lighting = sr.Lighting()

	rasterizer = sr.SoftRasterizer(
		image_size=image_size,
		background_color=np.array([0, 1.0, 0.0]),
	)

	# rasterizer.set_gamma(0)
	# rasterizer.set_sigma(0)

	green_mask = torch.tensor(green_screen_image[:, :, 1] > 0.5, dtype=torch.bool, requires_grad=False).cuda() # Fuzzy compare if pixel is green
	combined_image_torch = torch.tensor(combined_image, dtype=torch.float32, requires_grad=False).cuda()
	green = torch.tensor([[[0, 1, 0]]], dtype=torch.float32, requires_grad=False).cuda()

	optimizer = torch.optim.Adam(armature_layer.parameters(), 6e-3, betas=(0.9, 0.9))
	# optimizer = torch.optim.SGD(armature_layer.parameters(), lr=1e-4, momentum=0)

	for niter in range(500): # TODO: Config this

		mesh, pose_bone_transforms, pose_bone_scales = armature_layer(combined_image, video_image, green_screen_image, chroma_mask)

		# Render the image
		# mesh = lighting(mesh)
		mesh = transform(mesh)
		renders = rasterizer(mesh)

		renders = renders.permute(0, 2, 3, 1)
		render = renders[0, :, :, :3]

		mse_image = ((render - combined_image_torch) ** 2).sum(dim=2) # Mean squared error between render and image, doing reduce sum only on axis 2 so we can display this result to the user

		# should_be_green = render[green_mask, :]
		# green_screen_loss = ((should_be_green - green) ** 2).sum() # Add an additional penalty for overlapping the green screen

		loss = mse_image.sum() # - green_screen_loss * 2

		print(loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# 	grads = tape.gradient(loss, self.trainable_variables)
		# 	self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		# return render, mse_image, pose_bone_transforms, self.pose_bone_scales


		# cv2.imshow('render', render)
		# cv2.waitKey(0)


		render = render.detach().cpu().numpy()
		mse_image = mse_image.detach().cpu().numpy()

		display_image = np.column_stack(
			(combined_image, render, np.tile(mse_image[:, :, None], (1, 1, 3)))
		)

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

	# Convert pose bone transforms to pure numpy array instead of list of arrays
	pose_bone_transforms = np.array(pose_bone_transforms, dtype=np.float64)
	pose_bone_scales = pose_bone_scales.numpy()

	# Pickle generated matrices and send over stdout
	result = (
		pose_bone_transforms,
		pose_bone_scales,
	)
	write_pickled(result)

	return True # Continue looping in calling function

if __name__ == '__main__':
	import os
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	# # Use experimental gpu memory growth so we can run more than one process at a time
	# gpus = tf.config.list_physical_devices('GPU')
	# # Currently, memory growth needs to be the same across GPUs
	# for gpu in gpus:
	# 	tf.config.experimental.set_memory_growth(gpu, True)
	# 	logical_gpus = tf.config.list_logical_devices('GPU')

	# Read subprocess args
	subprocess_args = read_pickled()

	# Main loop
	while handle_stdin(subprocess_args):
		pass