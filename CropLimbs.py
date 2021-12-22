import bpy
import cv2
import numpy as np
import mathutils
import os
import pdb

cmvp = bpy.data.texts["CameraMVP"].as_module()
config = bpy.data.texts["Config"].as_module()


def generate_euclidean_distance_from_center_factors(kernel_width):
	arange = np.arange(kernel_width)
	i, j = np.meshgrid(arange, arange)
	i = i.T
	j = j.T
	ij = np.dstack([i, j]) # (i, j) coordinates of every position in a kernel_width x kernel_width matrix

	center = kernel_width // 2 # Center is the same for both x and y axis so I'll let numpy broadcasting handle 2 element vector to scalar math
	euclidean_distance_from_center_squared = np.sum(np.square(ij - center), axis=2) # Distance of every position from the center coordinates squared
	euclidean_distance_from_center_squared[center, center] = 1 # The center distance is ignored so just fill in with 1 to prevent divide by zero
	euclidean_distance_factors = 1 / euclidean_distance_from_center_squared

	return euclidean_distance_factors.astype(np.float32)
euclidean_distance_factors = generate_euclidean_distance_from_center_factors(3)


parts = [
	bpy.data.objects['shoulder.left.outer.circle'],
	bpy.data.objects['elbow.left.circle'],
	bpy.data.objects['wrist.left.circle'],

	bpy.data.objects['shoulder.right.outer.circle'],
	bpy.data.objects['elbow.right.circle'],
	bpy.data.objects['wrist.right.circle'],

	bpy.data.objects['hip.left.circle'],
	bpy.data.objects['knee.left.circle'],
	bpy.data.objects['ankle.left.circle'],

	bpy.data.objects['hip.right.circle'],
	bpy.data.objects['knee.right.circle'],
	bpy.data.objects['ankle.right.circle'],
]

trap_vertex_parts = [
	'shoulder.left.bt.00',
	'elbow.left.bt.00',

	'shoulder.right.bt.00',
	'elbow.right.bt.00',

	'hip.left.bt.00',
	'knee.left.bt.00',

	'hip.right.bt.00',
	'knee.right.bt.00',
]

def order_points(pts):
	partitioned_x = np.argpartition(pts[:, 0], -2)
	left = pts[partitioned_x[:2], :]
	right = pts[partitioned_x[2:], :]

	partitioned_y = np.argpartition(left[:, 1], -1)
	top_left = left[partitioned_y[0], :]
	bottom_left = left[partitioned_y[1], :]

	partitioned_y = np.argpartition(right[:, 1], -1)
	top_right = right[partitioned_y[0], :]
	bottom_right = right[partitioned_y[1], :]

	return np.stack([bottom_right, bottom_left, top_left, top_right])


def intersect_start_dir(s1, d1, s2, d2):
	u = (s1[:, 1]*d2[:, 0] + d2[:, 1]*s2[:, 0] - s2[:, 1]*d2[:, 0] - d2[:, 1]*s1[:, 0] ) / (d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0])
	return s1 + d1 * u[:, np.newaxis]

def bounding_trap(c1, radius1, c2, radius2):
	# Create output arrays
	p1 = np.zeros_like(c1)
	p2 = np.zeros_like(c1)
	p3 = np.zeros_like(c1)
	p4 = np.zeros_like(c1)

	# Get differences
	dy = c2[:, 1] - c1[:, 1]
	dx = c2[:, 0] - c1[:, 0]
	dx[dx == 0] = 1e-8 # Prevent divide by zero

	sign = (dx >= 0) * 2 - 1

	gamma_arg = dy / dx
	beta_arg = (radius2 - radius1) / np.sqrt(dy*dy + dx*dx)

	# If beta_arg outside arcsin domain, then one circle encapsulates the other
	encapsulated = np.logical_or(beta_arg < -1, beta_arg > 1)
	normal = np.logical_not(encapsulated)
	circle1_outer = np.logical_and(encapsulated, radius1 > radius2)
	circle2_outer = np.logical_and(encapsulated, radius2 >= radius1)

	# _circle1_outer means that we are only viewing the cases where circle1 encapsulates circle2
	c1_circle1_outer = c1[circle1_outer]
	radius1_circle1_outer = radius1[circle1_outer]

	# _circle1_outer means that we are only viewing the cases where circle2 encapsulates circle1
	c2_circle2_outer = c2[circle2_outer]
	radius2_circle2_outer = radius2[circle2_outer]


	p1[circle1_outer] = c1_circle1_outer + np.stack([radius1_circle1_outer, radius1_circle1_outer]).T
	p2[circle1_outer] = c1_circle1_outer + np.stack([radius1_circle1_outer, -radius1_circle1_outer]).T
	p3[circle1_outer] = c1_circle1_outer + np.stack([-radius1_circle1_outer, radius1_circle1_outer]).T
	p4[circle1_outer] = c1_circle1_outer + np.stack([-radius1_circle1_outer, -radius1_circle1_outer]).T

	p1[circle2_outer] = c2_circle2_outer + np.stack([radius2_circle2_outer, radius2_circle2_outer]).T
	p2[circle2_outer] = c2_circle2_outer + np.stack([radius2_circle2_outer, -radius2_circle2_outer]).T
	p3[circle2_outer] = c2_circle2_outer + np.stack([-radius2_circle2_outer, radius2_circle2_outer]).T
	p4[circle2_outer] = c2_circle2_outer + np.stack([-radius2_circle2_outer, -radius2_circle2_outer]).T

	# _normal means that we are excluding the cases where one circle encapsulates another
	gamma_normal = -np.arctan(gamma_arg[normal])
	beta_normal = sign[normal] * np.arcsin(beta_arg[normal])
	radius1_normal = radius1[normal].reshape(-1)
	radius2_normal = radius2[normal].reshape(-1)
	c1_normal = c1[normal]
	c2_normal = c2[normal]
	p1_normal = np.zeros_like(c1_normal)
	p2_normal = np.zeros_like(c1_normal)
	p3_normal = np.zeros_like(c1_normal)
	p4_normal = np.zeros_like(c1_normal)

	alpha_normal = gamma_normal - beta_normal
	sina_normal = np.sin(alpha_normal)
	cosa_normal = np.cos(alpha_normal)

	p1_normal = c1_normal + np.stack([radius1_normal * sina_normal, radius1_normal * cosa_normal]).T
	p2_normal = c2_normal + np.stack([radius2_normal * sina_normal, radius2_normal * cosa_normal]).T

	alpha_normal = gamma_normal + beta_normal
	sina_normal = np.sin(alpha_normal)
	cosa_normal = np.cos(alpha_normal)

	p3_normal = c1_normal - np.stack([radius1_normal * sina_normal, radius1_normal * cosa_normal]).T
	p4_normal = c2_normal - np.stack([radius2_normal * sina_normal, radius2_normal * cosa_normal]).T

	# Right now the points are both tangent and lie on the circumference of the circles, but we want to extend them out to encapsulate the circles, so intersect with the lines perpendicular to the farthest points on each circle
	# dc_normal = np.stack([dx[normal], dy[normal]]).T
	# dc_normal /= np.linalg.norm(dc_normal, axis=1)[:, np.newaxis]
	# pdc_normal = dc_normal[:, ::-1].copy()
	# pdc_normal[:, 0] *= -1

	# d12_normal = p2_normal - p1_normal
	# d34_normal = p4_normal - p3_normal

	# c1_farthest_normal = c1_normal - dc_normal*radius1_normal[:, np.newaxis]
	# c2_farthest_normal = c2_normal + dc_normal*radius2_normal[:, np.newaxis]

	# p1[normal] = intersect_start_dir(c1_farthest_normal, pdc_normal, p1_normal, d12_normal)
	# p2[normal] = intersect_start_dir(c2_farthest_normal, pdc_normal, p2_normal, d12_normal)
	# p3[normal] = intersect_start_dir(c1_farthest_normal, pdc_normal, p3_normal, d34_normal)
	# p4[normal] = intersect_start_dir(c2_farthest_normal, pdc_normal, p4_normal, d34_normal)

	p1[normal] = p1_normal
	p2[normal] = p2_normal
	p3[normal] = p3_normal
	p4[normal] = p4_normal

	return p1, p2, p3, p4

def calculate_bounding_trap_vertices(MVP, MVP_i):
	scene = bpy.data.scenes['Scene']
	current_frame = scene.frame_current
	num_parts = len(parts)

	render = bpy.context.scene.render

	c = np.zeros((num_parts, 3), dtype=np.float32)
	rp = np.zeros_like(c)

	for j, part in enumerate(parts):
		v = part.data.vertices[0].co

		center_point = part.matrix_world.translation
		radius_point = part.matrix_world @ v

		c[j, :] = center_point
		rp[j, :] = radius_point[:3]

	# world_to_screen flattens the first two dimensions so these are (num_frames*8, 2)
	c = cmvp.world_to_screen(c, MVP, render.resolution_x, render.resolution_y)
	rp = cmvp.world_to_screen(rp, MVP, render.resolution_x, render.resolution_y)

	diff = c - rp
	r = np.linalg.norm(diff, axis=1)

	c1 = np.empty((8, 2), dtype=np.float32)
	c2 = np.empty_like(c1)
	r1 = np.empty(8, dtype=np.float32)
	r2 = np.empty_like(r1)

	c1[0] = c[0]
	c2[0] = c[1]
	c1[1] = c[1]
	c2[1] = c[2]
	c1[2] = c[3]
	c2[2] = c[4]
	c1[3] = c[4]
	c2[3] = c[5]
	c1[4] = c[6]
	c2[4] = c[7]
	c1[5] = c[7]
	c2[5] = c[8]
	c1[6] = c[9]
	c2[6] = c[10]
	c1[7] = c[10]
	c2[7] = c[11]

	r1[0] = r[0]
	r2[0] = r[1]
	r1[1] = r[1]
	r2[1] = r[2]
	r1[2] = r[3]
	r2[2] = r[4]
	r1[3] = r[4]
	r2[3] = r[5]
	r1[4] = r[6]
	r2[4] = r[7]
	r1[5] = r[7]
	r2[5] = r[8]
	r1[6] = r[9]
	r2[6] = r[10]
	r1[7] = r[10]
	r2[7] = r[11]

	d = c2 - c1

	l = np.linalg.norm(d, axis=1)
	d /= l[:, np.newaxis]

	p1, p2, p3, p4 = bounding_trap(c1, r1, c2, r2)
	btv = np.stack([p1, p2, p3, p4]) # Shape (4, 8, 2)

	return c1, r1, c2, r2, btv

def get_bounding_trap_vertices():
	render = bpy.context.scene.render
	camera = bpy.data.objects['Camera']
	MVP = cmvp.projection_matrix(camera)
	MVP_i = np.linalg.inv(MVP)

	c1, r1, c2, r2, btv = calculate_bounding_trap_vertices(MVP, MVP_i)

	# P1 through 4 are in screen coordinates, first convert to camera coordinates, then to world coordinates
	btv = cmvp.screen_to_camera(btv, render.resolution_x, render.resolution_y) # Shape (32, 2)
	btv = cmvp.camera_to_world(btv, 5, MVP_i) # Shape (32, 3)
	btv = btv.reshape((4, 8, 3))

	output_dict = {}
	for i in range(4):
		for j in range(8):
			part_name = f'{trap_vertex_parts[j]}{i+1}'
			output_dict[part_name + '_translation'] = btv[i, j, :]

	return output_dict
