import sys
import pickle
import struct
import cv2
import numpy as np
import math

# There is some dumb shit with blender's python shell that doesn't create a sys.stdin like normal.
# I don't really need the stdin and stdout functionality to work because I won't be running subprocesses
# from that shell. I just need a handful of things to work for debugging purposes, so let's just
# override to None in that case.
if not hasattr(sys.stdin, 'buffer'):
	sys_stdin = None
else:
	sys_stdin = sys.stdin.buffer

if not hasattr(sys.stdout, 'buffer'):
	sys_stdout = None
else:
	sys_stdout = sys.stdout.buffer

# external scripts are intended to be launched as a subprocess and is meant to have stdout
# reserved for subprocess communication, write everything to stderr instead
__print = print
def print(*args, **kwargs):
	__print(*args, file=sys.stderr, **kwargs)

# Starts a view at the specified frame
def start_video_at_frame(video_path, frame):
	green_screen_capture = cv2.VideoCapture(video_path)
	green_screen_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
	return green_screen_capture

# Gets the next frame of a video capture while applying transpose and scale
def get_next_frame(video_capture, transpose, scale=1.0, as_float=False, as_bgr=True):

	ret, video_image = video_capture.read()

	if not ret:
		return ret, None

	if transpose[0]:
		video_image = cv2.transpose(video_image)
	if transpose[1]:
		video_image = cv2.flip(video_image, -1)
	if transpose[2]:
		video_image = cv2.flip(video_image, 0)

	# Downscale image
	if scale != 1.0:
		image_shape = np.array(video_image.shape, dtype=np.float64)
		image_shape *= scale
		image_shape = image_shape.round().astype(np.int32)
		image_shape = (image_shape[1], image_shape[0])

		video_image = cv2.resize(video_image, image_shape, cv2.INTER_CUBIC)

	order = 1 if as_bgr else -1

	if as_float:
		return ret, video_image[:, :, ::order].astype(np.float64) / 255
	else:
		return ret, video_image[:, :, ::order]

# Read arguments coming from stdin as pickled object with size long prepended
def read_pickled(stdin=sys_stdin): # Any external script will read using sys.stdin.buffer, so use that by default
	size_long = stdin.read(8)
	size = struct.unpack('>Q', size_long)[0] # Read size of pickled arguments

	if size == 0: # Received stop command
		return None

	response = stdin.read(size)

	pyobject = pickle.loads(response)

	return pyobject

def write_pickled(pyobject, stdout=sys_stdout): # Any external script will write to sys.stdout.buffer, so use that as default
	pickled = pickle.dumps(pyobject)

	size_long = struct.pack('>Q', len(pickled))

	result = size_long + pickled

	stdout.write(result)
	stdout.flush()

def write_end(stdout=sys_stdout):
	size_long = struct.pack('>Q', 0)
	stdout.write(size_long)
	stdout.flush()

def quaternion_to_mat4(quaternion, translation, scale=None, backend=np):
	using_torch = 'torch' in str(backend) # Syntax is the same between tensorflow and numpy, but not pytorch

	qw = quaternion[:, 0]
	qx = quaternion[:, 1]
	qy = quaternion[:, 2]
	qz = quaternion[:, 3]
	tx = translation[:, 0]
	ty = translation[:, 1]
	tz = translation[:, 2]
	if scale is not None:
		sx = scale[:, 0]
		sy = scale[:, 1]
		sz = scale[:, 2]

	# The following was copied from mathutils source function quat_to_mat3_no_error: https://github.com/dfelinto/blender/blob/c4ef90f5a0b1d05b16187eb6e32323defe6461c0/source/blender/blenlib/intern/math_rotation.c
	M_SQRT2 = math.sqrt(2)

	q0 = M_SQRT2 * qw
	q1 = M_SQRT2 * qx
	q2 = M_SQRT2 * qy
	q3 = M_SQRT2 * qz

	qda = q0 * q1
	qdb = q0 * q2
	qdc = q0 * q3
	qaa = q1 * q1
	qab = q1 * q2
	qac = q1 * q3
	qbb = q2 * q2
	qbc = q2 * q3
	qcc = q3 * q3

	m00 = 1.0 - qbb - qcc
	m01 = qdc + qab
	m02 = -qdb + qac

	m10 = -qdc + qab
	m11 = 1.0 - qaa - qcc
	m12 = qda + qbc

	m20 = qdb + qac
	m21 = -qda + qbc
	m22 = 1.0 - qaa - qbb

	zeros = backend.zeros_like(m00)
	ones = backend.ones_like(m00)

	reconstructed = backend.stack([
		m00,   m01,   m02, zeros,
		m10,   m11,   m12, zeros,
		m20,   m21,   m22, zeros,
		tx,    ty,    tz,  ones,
	])
	reconstructed = backend.reshape(reconstructed, (4, 4, -1))
	if using_torch:
		reconstructed = reconstructed.permute(2, 0, 1)
	else:
		reconstructed = backend.transpose(reconstructed, (2, 0, 1))


	if scale is not None:
		scaled = backend.stack([
			sx*m00,          sx*m01,          sx*m02,          zeros,
			sy*m10,          sy*m11,          sy*m12,          zeros,
			sz*m20,          sz*m21,          sz*m22,          zeros,
			tx,              ty,              tz,              ones,
		])
		scaled = backend.reshape(scaled, (4, 4, -1))

		if using_torch:
			scaled = scaled.permute(2, 0, 1)
		else:
			scaled = backend.transpose(scaled, (2, 0, 1))

		return reconstructed, scaled

	return reconstructed

def quaternion_mul(a, b, backend=np):
	w0, x0, y0, z0 = b
	w1, x1, y1, z1 = a

	c = backend.stack([
		-x1*x0 - y1*y0 - z1*z0 + w1*w0,
		 x1*w0 + y1*z0 - z1*y0 + w1*x0,
		-x1*z0 + y1*w0 + z1*x0 + w1*y0,
		 x1*y0 - y1*x0 + z1*w0 + w1*z0,
	])

	return c

# Adapted from https://github.com/majimboo/py-mathutils/blob/655da9912ece6830653d2b9fc2fd9473c82547cb/src/blenlib/intern/math_rotation.c
def quaternion_invert(q):
	f = q.dot(q)
	conjugate = q.copy()
	conjugate[1:] *= -1
	return conjugate / f

# Adapted from: https://github.com/adamlwgriffiths/Pyrr/blob/master/pyrr/matrix44.py
def look_at_mat4(eye_position, target_position, up_vector):
	# eye_position = np.asarray(eye_position)
	# target_position = np.asarray(target_position)
	# up_vector = np.asarray(up_vector)

	forward = target_position - eye_position
	forward /= np.linalg.norm(forward)

	side = np.cross(forward, up_vector)
	side /= np.linalg.norm(side)

	up_vector = np.cross(side, forward)
	up_vector /= np.linalg.norm(up_vector)

	return np.array((
			(                    side[0],                    up_vector[0],                    -forward[0],  0.),
			(                    side[1],                    up_vector[1],                    -forward[1],  0.),
			(                    side[2],                    up_vector[2],                    -forward[2],  0.),
			(-np.dot(side, eye_position), -np.dot(up_vector, eye_position), np.dot(forward, eye_position), 1.0)
		), dtype=np.float64)

def get_keypoints_and_descriptors(im, detector):
	if im.dtype != np.uint8:
		im = (im * 255).astype(np.uint8)

	# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	gray = im

	# for i in range(2):
	# 	blurred1 = cv2.GaussianBlur(gray, (0, 0), 2, 2)
	# 	gray = gray * 2 - blurred1

	keypoints, descriptors = detector.detectAndCompute(gray, None)

	print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
	# cv2.imshow("AKAZE matching", im)
	# cv2.waitKey(20)

	return np.array(keypoints, dtype=object), descriptors

def match_descriptors(keypoints_descriptors_1, keypoints_descriptors_2, quality=0.6, return_indices=False, images=None):
	(kps1, descs1) = keypoints_descriptors_1
	(kps2, descs2) = keypoints_descriptors_2

	# Match the features
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descs1, descs2, k=2)
	# matches = bf.match(descs1, descs2)

	# matches = sorted(matches, key=lambda x: x.distance)
	# good = matches[:10]

	# Apply ratio test
	good = [m for m, n in matches]
	good = []
	for m,n in matches:
		if m.distance < quality*n.distance:
			good.append(m)

	if type(images) == tuple and len(images) == 2:
		im1, im2 = images

		if im1.dtype != np.uint8:
			im1 = (im1 * 255).astype(np.uint8)

		if im2.dtype != np.uint8:
			im2 = (im2 * 255).astype(np.uint8)

		result = np.concatenate([im1, im2], axis=1)

		for match in good:
			query_point = kps1[match.queryIdx].pt
			train_point = kps2[match.trainIdx].pt

			query_point = (int(round(query_point[0])), int(round(query_point[1])))
			train_point = (int(round(train_point[0])) + images[0].shape[1], int(round(train_point[1])))

			cv2.line(result, query_point, train_point, (255, 0, 0), 4)

		cv2.imshow('Akaze Matches', result[::4, ::4, :])
		cv2.waitKey(1)

	query_indices_matched = np.array([match.queryIdx for match in good], dtype=np.uint32)
	train_indices_matched = np.array([match.trainIdx for match in good], dtype=np.uint32)

	if return_indices:
		return query_indices_matched, train_indices_matched

	else:

		kps1_matched = [kps1[idx] for idx in query_indices_matched]
		descs1_matched = descs1[query_indices_matched]
		keypoints_descriptors_1_matched = (kps1_matched, descs1_matched)

		kps2_matched = [kps2[idx] for idx in train_indices_matched]
		descs2_matched = descs2[train_indices_matched]
		keypoints_descriptors_2_matched = (kps2_matched, descs2_matched)

		return keypoints_descriptors_1_matched, keypoints_descriptors_2_matched

def ray(points, MVP_i):
	p1 = np.zeros((points.shape[0], 4), dtype=np.float64) # shape (batch size, 4 for xyzw)
	p1[:, :2] = points
	p1[:, 2] = 0
	p1[:, 3] = 1

	p2 = p1.copy()
	p2[:, 2] = 0.9

	p1 = p1.dot(MVP_i)
	p2 = p2.dot(MVP_i)
	p1 /= p1[:, 3, np.newaxis]
	p2 /= p2[:, 3, np.newaxis]

	return p1, p2
