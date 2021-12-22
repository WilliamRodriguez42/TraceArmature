# pipenv install Cython tensorflow mediapipe deodr moderngl moderngl-window lxml cupy-cuda102 pygame torch torchvision opencv-contrib-python

import moderngl
import moderngl_window
import cv2
import mediapipe as mp
import pickle
import os
import pdb
import time
import threading
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


class WebCamFaceUVRecorder(moderngl_window.WindowConfig):
	gl_version = (4, 4)

	title = 'WebCamFaceUVRecorder'
	resizable = True
	aspect_ratio = None

	def __init__(self, **kwargs):
		global cap
		global vertices, tri_vertex_indices, tri_uv_indices, tri_uvs

		super().__init__(**kwargs)

		# self.wnd.exit_key = None # Don't allow user to exit using escape

		"""
		I'm keeping two depth buffers for the distance shader. One is a color renderbuffer with 1 channel
		that represents the real world pixel distance from the camera. The second is the depth renderbuffer
		which stores the depth on the range of 0 to 1 by subtracting the minimum pixel depth, then dividing
		the differnece between the max and min pixel depths. I'm doing this because I don't want to be limited
		by near / far clipping planes. Also because I like to think in blender units, and unfortunately those
		won't suffice for the DEPTH_TEST.
		"""

		self.distance_shader = self.ctx.program(
			vertex_shader='''
				#version 440
				in vec3 screen_uv; // vec3 because 3rd element is camera distance
				out float depth; // AI approximated camera distance

				void main() {
					vec2 screen_pos_2d = screen_uv.xy * 2 - 1;

					depth = screen_uv.z;
					gl_Position = vec4(screen_pos_2d, depth, 1.0);
				}
			''',
			fragment_shader='''
				#version 440
				out float color;
				in float depth;

				void main() {
					color = depth;
				}
			''',
		)

		self.uv_shader = self.ctx.program(
			vertex_shader='''
				#version 440
				in vec2 tri_uv;
				in vec3 tri_screen_uv;

				out vec2 uv;
				out vec3 screen_uv;

				void main() {
					screen_uv = tri_screen_uv;

					vec2 screen_coords = tri_uv * 2 - 1;
					gl_Position = vec4(screen_coords, 0, 1);
				}
			''',
			fragment_shader='''
				#version 440
				out vec4 color;
				in vec3 screen_uv;

				uniform sampler2D video_texture;
				uniform sampler2D distance_texture;

				void main() {
					float min_distance_for_pixel = texture(distance_texture, screen_uv.xy).r;
					float distance_for_frag = screen_uv.z;

					// Occluded if there is geometry in the way
					bool is_closest = distance_for_frag < min_distance_for_pixel + 0.01f; // Add a small value because distance_texture doesn't have a one to one mapping to this fragment, so we need a bit of wiggle room in the comparison to prevent aliasing

					if (is_closest) {
						color = texture(video_texture, screen_uv.xy).rgba; // Use video color
					} else {
						color = vec4(1, 0, 1, 0); // Any RGB value with no alpha will indicate an occluded area
					}
				}
			''',
		)

		# Gather initial values for screen_uvs and video_image
		video_image, screen_uvs = get_video_image_and_screen_uvs()

		# Load video and get video properties
		self.video_width = video_image.shape[1]
		self.video_height = video_image.shape[0]
		self.video_texture = self.ctx.texture((self.video_width, self.video_height), video_image.shape[2], data=video_image.tobytes())

		# Create vertex attribute array with screen uvs with depth for the depth shader
		self.distance_ibo = self.ctx.buffer(tri_vertex_indices.tobytes())

		self.distance_vbo = self.ctx.buffer(screen_uvs.tobytes())
		self.distance_vao = self.ctx.vertex_array(
			self.distance_shader,
			[(self.distance_vbo, '3f', 'screen_uv')],
			self.distance_ibo
		)

		# Create a renderbuffer and associate it to a framebuffer for distance_shader output
		self.distance_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 1, dtype='f4')
		self.distance_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
		self.distance_shader_framebuffer = self.ctx.framebuffer(self.distance_shader_color_texture, self.distance_shader_depth_texture)

		# Create placeholder buffers for uv shader
		tri_screen_uvs = screen_uvs[tri_vertex_indices].reshape((-1, 3))
		tri_uvs_and_tri_screen_uvs = np.concatenate([tri_uvs, tri_screen_uvs], axis=1)
		self.uv_vbo = self.ctx.buffer(tri_uvs_and_tri_screen_uvs.tobytes())

		self.uv_vao = self.ctx.vertex_array(
			self.uv_shader,
			[(self.uv_vbo, '2f 3f', 'tri_uv', 'tri_screen_uv')],
		)

		# Set layout indices of textures
		self.uv_shader['video_texture'].value = 0
		self.uv_shader['distance_texture'].value = 1

		# User uv shader framebuffer to enable data viewing from cpu
		self.uv_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
		self.uv_shader_framebuffer = self.ctx.framebuffer(self.uv_shader_color_texture) # No depth texture is necessary

		# Create two output video captures to write to
		fps = 15
		# output_path =

		# magenta_occlusion_path = os.path.join(self.output_path, 'magenta_occlusion.avi')
		# self.magenta_occlusion_writer = cv2.VideoWriter(magenta_occlusion_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.video_width, self.video_height)) # This result can be imported to blender directly to give an idea of what areas are occluded

	def update_buffers(self, video_image, screen_uvs):
		global vertices, tri_vertex_indices, tri_uv_indices, tri_uvs

		# Update distance shader vbo with current frame screen_uvs
		self.distance_vbo.write(screen_uvs.tobytes())

		# Get screen_uv coordinates for each tri
		tri_screen_uvs = screen_uvs[tri_vertex_indices].reshape((-1, 3))

		# Interleave tri uvs and tri screen uvs
		tri_uvs_and_tri_screen_uvs = np.concatenate([tri_uvs, tri_screen_uvs], axis=1)

		# Update uv shader vbo with current frame tri_uvs_and_screen_uvs
		self.uv_vbo.write(tri_uvs_and_tri_screen_uvs.tobytes())

		# Update video frame
		self.video_texture.write(video_image.tobytes())

	def render(self, time, frametime):
		global vertices, tri_vertex_indices, tri_uv_indices, tri_uvs

		video_image, screen_uvs = get_video_image_and_screen_uvs()

		# These will be used to normalize the pixel depth to range (0, 1) for DEPTH_TEST
		pixel_min_depth = screen_uvs[:, 2].min()
		pixel_max_range = screen_uvs[:, 2].max() - pixel_min_depth
		screen_uvs[:, 2] = (screen_uvs[:, 2] - pixel_min_depth) / pixel_max_range

		# Clear buffers
		self.ctx.clear(0.0, 0.0, 0.0, 0.0)
		self.distance_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)
		self.uv_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)

		# Update buffers
		self.update_buffers(video_image, screen_uvs)

		# Find pixel distance from the blender camera
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.enable(moderngl.DEPTH_TEST)
		self.distance_shader_framebuffer.use()
		self.distance_vao.render()

		# self.ctx.disable(moderngl.CULL_FACE)
		# self.ctx.enable(moderngl.DEPTH_TEST)
		# self.wnd.use()
		# self.distance_vao.render()

		# Create a uv unwrapped image showing which parts of the model are visible
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.disable(moderngl.DEPTH_TEST)
		self.uv_shader_framebuffer.use()
		self.video_texture.use(0)
		self.distance_shader_color_texture.use(1)
		self.uv_vao.render()

		# Just redoing that last step but rendering to the screen and not a framebuffer
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.disable(moderngl.DEPTH_TEST)
		self.wnd.use()
		self.video_texture.use(0)
		self.distance_shader_color_texture.use(1)
		self.uv_vao.render()

		# Read result to numpy
		raw = self.uv_shader_color_texture.read()
		cpu_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))
		cpu_image = cv2.cvtColor(cpu_image, cv2.COLOR_BGRA2RGBA)

	def resize(self, width, height):
		self.ctx.viewport = (0, 0, width, height)

	def close(self):
		# self.magenta_occlusion_writer.release()
		return

def get_video_image_and_screen_uvs():
	global cap, face_mesh

	while True:
		success, video_image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			continue

		video_image.flags.writeable = False
		video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(video_image)

		if results.multi_face_landmarks:
			break

	screen_uvs = np.array([(l.x, l.y, l.z) for l in results.multi_face_landmarks[0].landmark])

	return video_image, screen_uvs.astype(np.float32)


if __name__ == '__main__':
	cap = cv2.VideoCapture(0) # Webcam

	dir_path = os.path.dirname(__file__)
	file_path = os.path.join(dir_path, '..', 'resources', 'canonical_face_mesh_info.pkl')
	with open(file_path, 'rb') as f:
		content = f.read()
	vertices, tri_vertex_indices, tri_uv_indices, tri_uvs = pickle.loads(content)
	vertices = vertices.astype(np.float32)
	tri_vertex_indices = tri_vertex_indices.astype(np.int32)
	tri_uv_indices = tri_uv_indices.astype(np.int32)
	tri_uvs = tri_uvs.astype(np.float32)


	with mp_face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as face_mesh:

		thread = threading.Thread(target=WebCamFaceUVRecorder.run, daemon=True)
		thread.start()
		thread.join()