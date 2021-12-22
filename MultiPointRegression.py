import os
import tensorflow as tf
import bpy
import pdb
import math
import mathutils
import numpy as np

from helper_functions import print, start_video_at_frame, read_pickled, write_pickled, quaternion_to_mat4
def _quaternion_to_mat4(quaternion, translation, scale=None): # Create a wrapper function so I don't have to hand in tf all the time
	return quaternion_to_mat4(quaternion, translation, torch, scale=scale)

def rotation_matrix_xyz(x, y, z):
	num_frames = x.shape[0]

	sinx = tf.sin(x)
	cosx = tf.cos(x)

	siny = tf.sin(y)
	cosy = tf.cos(y)

	sinz = tf.sin(z)
	cosz = tf.cos(z)

	vert_zero = tf.zeros(num_frames, dtype=tf.float32)
	vert_one = tf.ones(num_frames, dtype=tf.float32)

	# Rotation order ry, rx, rz
	a = cosy*cosz + sinx*siny*sinz
	b = -cosy*sinz + cosz*sinx*siny
	c = -cosx*siny
	d = vert_zero
	e = cosx*sinz
	f = cosx*cosz
	g = sinx
	h = vert_zero
	i = -cosy*sinx*sinz + cosz*siny
	j = -cosy*cosz*sinx - siny*sinz
	k = cosx*cosy
	l = vert_zero
	m = vert_zero
	n = vert_zero
	o = vert_zero
	p = vert_one

	stack = tf.stack([
		a, b, c, d,
		e, f, g, h,
		i, j, k, l,
		m, n, o, p,
	])

	stack = tf.transpose(stack)
	stack = tf.reshape(stack, (num_frames, 4, 4))

	return stack

class RotationTranslationLayer(tf.keras.layers.Layer):

	def __init__(
		self,
		MVP, # Model view projection matrix of camera, mat 4x4, assumed constant for all frames
		num_points, # The number of points we wish to optimize
		num_frames, # The number of frames we wish to optimize
		initial_frame_translations=None, # Initial value for the world translation of the object per frame
		initial_frame_quaternions=None, # Initial value for the world rotation of the object per frame (frame_quaternions)
		use_rotation=True,
		rotation_limits=False,
	):
		super().__init__(self)

		self.MVP = tf.constant(MVP, dtype=tf.float32)
		self.use_rotation = use_rotation
		self.rotation_limits = rotation_limits

		# Initialize frame translations to either 0 or initial_frame_translations
		if initial_frame_translations is None:
			frame_translations = np.zeros((num_frames, 1, 3), dtype=np.float32)
		else:
			frame_translations = initial_frame_translations.reshape((num_frames, 1, 3))
		self.frame_translations = tf.Variable(frame_translations, name="frame_translations", dtype=tf.float32)
		self.frame_translations_w = tf.constant(tf.zeros((num_frames, 1, 1)), dtype=tf.float32) # Store matrix of zeros for converting translation vec3 to vec4 (use zeros and not ones because we will be adding these to the existing point vec4)

		# Initialize frame quaternions to either identity or normalized initial_frame_quaternions
		if initial_frame_quaternions is None:
			frame_quaternions = np.zeros((num_frames, 4), dtype=np.float32)
			frame_quaternions[:, 3] = 1
		else:
			frame_quaternions = initial_frame_quaternions.copy()
			frame_quaternions /= np.linalg.norm(frame_quaternions, axis=1)[:, np.newaxis]
		self.frame_quaternions = tf.Variable(quaternions, dtype=tf.float32)

		# Initialize point translation offsets to 0
		self.point_offsets = tf.Variable(tf.zeros((num_points, 3)), name="offset", dtype=tf.float32)
		self.point_offsets_w = tf.constant(tf.zeros((num_points, 1)), dtype=tf.float32) # Store matrix of ones for converting point offset vec3 to vec4 (use zeros and not ones because we will be adding these to the existing point vec4)

	def apply_offsets_to_points(self, points):
		point_offsets = tf.concat([self.point_offsets, self.point_offsets_w], axis=1)
		points = points + point_offsets

		return points

	def apply_frame_transforms(self, points):
		frame_translations = tf.concat([self.frame_translations, self.frame_translations_w], axis=2)
		frame_rotation_matrices = hf.quaternion_to_mat4(self.frame_quaternions)
		points = tf.matmul(frame_rotation_matrices[np.newaxis, :], points)
		points = points + frame_translations

		return points

	def call(self, points):
		# Points has shape (num_points, 3)

		# Normalize frame quaternions
		self.frame_quaternions = self.frame_quaternions / tf.norm(self.frame_quaternions, axis=2)[:, tf.newaxis]

		# Apply offsets to all points and apply each frame translation / rotation to all points
		points = self.apply_offsets_to_points(points)
		points = self.apply_frame_transforms(points)

		# Apply model view projection matrix to all frames
		points = tf.matmul(points, self.MVP)
		points = points / points[:, :, 3, tf.newaxis] # Normalize by dividing by w

		# Return projected points
		return points[:, :, :2]

	def generate_transforms(self):
		num_frames = tf.shape(self.frame_translations)[0]

		frame_translations = self.frame_translations
		identity = np.identity(4, dtype=np.float32)
		post_translation = np.zeros((num_frames, 4, 4), dtype=np.float32)
		post_translation[:, :, :] = identity[np.newaxis, :, :]
		post_translation[:, 3, :3] = frame_translations.numpy().reshape(-1, 3)

		rotation_matrix = hf.quaternion_to_mat4(self.quaternions)

		transforms = np.matmul(rxyz, post_translation)

		return transforms
