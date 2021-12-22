import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

import pdb
import numpy as np

# https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix

def view_plane(camd, winx, winy, xasp, yasp):
	#/* fields rendering */
	ycor = yasp / xasp
	use_fields = False
	if (use_fields):
	  ycor *= 2

	def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
		#/* sensor size used to fit to. for auto, sensor_x is both x and y. */
		if (p_sensor_fit == 'VERTICAL'):
			return sensor_y

		return sensor_x

	if (camd.type == 'ORTHO'):
	  #/* orthographic camera */
	  #/* scale == 1.0 means exact 1 to 1 mapping */
	  pixsize = camd.ortho_scale
	else:
	  #/* perspective camera */
	  sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
	  pixsize = (sensor_size * camd.clip_start) / camd.lens

	#/* determine sensor fit */
	def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
		if (p_sensor_fit == 'AUTO'):
			if (sizex >= sizey):
				return 'HORIZONTAL'
			else:
				return 'VERTICAL'

		return p_sensor_fit

	sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

	if (sensor_fit == 'HORIZONTAL'):
	  viewfac = winx
	else:
	  viewfac = ycor * winy

	pixsize /= viewfac

	#/* extra zoom factor */
	pixsize *= 1 #params->zoom

	#/* compute view plane:
	# * fully centered, zbuffer fills in jittered between -.5 and +.5 */
	xmin = -0.5 * winx
	ymin = -0.5 * ycor * winy
	xmax =  0.5 * winx
	ymax =  0.5 * ycor * winy

	#/* lens shift and offset */
	dx = camd.shift_x * viewfac # + winx * params->offsetx
	dy = camd.shift_y * viewfac # + winy * params->offsety

	xmin += dx
	ymin += dy
	xmax += dx
	ymax += dy

	#/* fields offset */
	#if (params->field_second):
	#    if (params->field_odd):
	#        ymin -= 0.5 * ycor
	#        ymax -= 0.5 * ycor
	#    else:
	#        ymin += 0.5 * ycor
	#        ymax += 0.5 * ycor

	#/* the window matrix is used for clipping, and not changed during OSA steps */
	#/* using an offset of +0.5 here would give clip errors on edges */
	xmin *= pixsize
	xmax *= pixsize
	ymin *= pixsize
	ymax *= pixsize

	return xmin, xmax, ymin, ymax


def projection_matrix(camera, as_world_and_perspective=False):
	camd = camera.data
	r = bpy.context.scene.render
	left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)

	farClip, nearClip = camd.clip_end, camd.clip_start

	Xdelta = right - left
	Ydelta = top - bottom
	Zdelta = farClip - nearClip

	mat = np.zeros((4, 4), dtype=np.float32)

	mat[0][0] = nearClip * 2 / Xdelta
	mat[1][1] = nearClip * 2 / Ydelta
	mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
	mat[2][1] = (top + bottom) / Ydelta
	mat[2][2] = -(farClip + nearClip) / Zdelta
	mat[2][3] = -1
	mat[3][2] = (-2 * nearClip * farClip) / Zdelta

	mv = camera.matrix_world.inverted()
	mv = np.array(mv, dtype=np.float32)

	if as_world_and_perspective:
		return (mv.T, mat)
	else:
		return mv.T.dot(mat)

def ray(points, MVP_i):
	p1 = np.zeros((points.shape[0], 4), dtype=np.float32) # shape (batch size, 4 for xyzw)
	p1[:, :2] = points
	p1[:, 2] = -1
	p1[:, 3] = 1

	p2 = p1.copy()
	p2[:, 2] = 1

	p1 = p1.dot(MVP_i)
	p2 = p2.dot(MVP_i)
	p1 /= p1[:, 3, np.newaxis]
	p2 /= p2[:, 3, np.newaxis]

	return p1, p2

def camera_to_world(points, y, MVP_i):
	points = np.array(points, dtype=np.float32).reshape(-1, 2)
	p1, p2 = ray(points, MVP_i)

	t = (y - p1[:, 1]) / (p2[:, 1] - p1[:, 1])
	p3 = p1 + (p2 - p1) * t[:, np.newaxis]

	return p3[:, :3]

def camera_xy_depth_to_world(camera_pos, points, depth, MVP_i):
	points = points.reshape(-1, 2)
	p1, p2 = ray(points, MVP_i)

	direction = p2 - p1
	direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]

	p3 = direction[:, :3] * depth + camera_pos

	return p3

def world_to_screen(points, MVP, resolution_x, resolution_y):
	points = np.array(points).reshape(-1, 3)
	world_pos = np.ones((points.shape[0], 4), dtype=np.float32)
	world_pos[:, :3] = points
	screen_pos = world_pos.dot(MVP)
	screen_pos /= screen_pos[:, 3, np.newaxis]

	screen_pos = screen_pos * 0.5 + 0.5 # [-1, 1] to [0, 1]
	screen_pos[:, 0] *= resolution_x
	screen_pos[:, 1] *= resolution_y
	screen_pos[:, 1] = resolution_y - screen_pos[:, 1] # Flip y

	return screen_pos[:, [1, 0]] # i, j coordinates, floating point

def screen_to_camera(points, resolution_x, resolution_y):
	points = points.reshape(-1, 2)

	points[:, 1] = resolution_y - points[:, 1]
	points[:, 0] /= resolution_x
	points[:, 1] /= resolution_y

	points = points * 2 - 1

	return points



# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd, use_render_percent=True): # Use render percent means we will use a downscaled resolution based on render settings percent
	f_in_mm = camd.lens
	scene = bpy.context.scene
	resolution_x_in_px = scene.render.resolution_x
	resolution_y_in_px = scene.render.resolution_y
	scale = scene.render.resolution_percentage / 100 if use_render_percent else 1
	sensor_width_in_mm = camd.sensor_width
	sensor_height_in_mm = camd.sensor_height
	pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
	if (camd.sensor_fit == 'VERTICAL'):
		# the sensor height is fixed (sensor fit is horizontal),
		# the sensor width is effectively changed with the pixel aspect ratio
		s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
		s_v = resolution_y_in_px * scale / sensor_height_in_mm
	else: # 'HORIZONTAL' and 'AUTO'
		# the sensor width is fixed (sensor fit is horizontal),
		# the sensor height is effectively changed with the pixel aspect ratio
		pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
		s_u = resolution_x_in_px * scale / sensor_width_in_mm
		s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

	# Parameters of intrinsic calibration matrix K
	alpha_u = f_in_mm * s_u
	# alpha_v = f_in_mm * s_v
	u_0 = resolution_x_in_px * scale / 2
	v_0 = resolution_y_in_px * scale / 2
	skew = 0 # only use rectangular pixels

	K = Matrix((
		( alpha_u,    skew,   u_0 ),
		(       0, alpha_u,   v_0 ),
		(       0,       0,     1 ),
	))
	return np.array(K, dtype=np.float64)

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
	# bcam stands for blender camera
	R_bcam2cv = Matrix(
		((1, 0,  0),
		(0, -1, 0),
		(0, 0, -1)))

	# Transpose since the rotation is object rotation,
	# and we want coordinate rotation
	# R_world2bcam = cam.rotation_euler.to_matrix().transposed()
	# T_world2bcam = -1*R_world2bcam * location
	#
	# Use matrix_world instead to account for all constraints
	location, rotation = cam.matrix_world.decompose()[0:2]
	R_world2bcam = rotation.to_matrix().transposed()

	# Convert camera location to translation vector used in coordinate changes
	# T_world2bcam = -1*R_world2bcam*cam.location
	# Use location from matrix_world to account for constraints:
	T_world2bcam = -1*R_world2bcam @ location

	# Build the coordinate transform matrix from world to computer vision camera
	# NOTE: Use * instead of @ here for older versions of Blender
	# TODO: detect Blender version
	R_world2cv = R_bcam2cv@R_world2bcam
	T_world2cv = R_bcam2cv@T_world2bcam

	# put into 3x4 matrix
	RT = Matrix((
		R_world2cv[0][:] + (T_world2cv[0],),
		R_world2cv[1][:] + (T_world2cv[1],),
		R_world2cv[2][:] + (T_world2cv[2],)
		))
	return np.array(RT, dtype=np.float64)

if __name__ == '__main__':
	camera = bpy.data.objects['Camera']
	print(projection_matrix(camera))