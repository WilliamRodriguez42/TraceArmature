import moderngl
import moderngl_window
import numpy as np
import cv2
import os
import time

from helper_functions import print, start_video_at_frame, get_next_frame, read_pickled, write_pickled, look_at_mat4, get_keypoints_and_descriptors, match_descriptors

class Test(moderngl_window.WindowConfig):
	gl_version = (4, 4)

	title = 'Triangle'
	resizable = True
	aspect_ratio = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		(
			self.video_path, # Only used for getting the resolution
			texture_path,
			self.vertices,
			self.tri_uvs,
			projection_matrix,
			self.output_path,
		) = read_pickled()


		self.wnd.exit_key = None # Don't allow user to exit using escape

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
				in vec4 vertex;
				in vec2 vert_uv;

				out vec2 frag_uv;

				uniform mat4 model_view_matrix;
				uniform mat4 projection_matrix;

				void main() {
					frag_uv = vert_uv;

					mat4 mvp_matrix = model_view_matrix * projection_matrix;
					vec4 position = vertex * mvp_matrix;

					gl_Position = position;
				}
			''',
			fragment_shader='''
				#version 440
				in vec2 frag_uv;

				out vec4 color;

				uniform sampler2D skin_texture;

				void main() {
					color = texture(skin_texture, frag_uv);
				}
			''',
		)

		# Load video and get video properties
		self.video_capture = start_video_at_frame(self.video_path, 0)
		_, video_image = get_next_frame(self.video_capture, (    False,  False,  False))

		self.video_width  = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

		# Create vertex attribute array
		self.distance_vertex_vbo = self.ctx.buffer(self.vertices.astype(np.float32).tobytes())
		self.distance_uv_vbo = self.ctx.buffer(self.tri_uvs.astype(np.float32).tobytes())
		self.distance_vao = self.ctx.vertex_array(
			self.distance_shader,
			[
				(self.distance_vertex_vbo, '4f', 'vertex'),
				(self.distance_uv_vbo, '2f', 'vert_uv'),
			],
		)

		# Load and assign texture
		skin_texture = cv2.imread(texture_path)
		skin_texture = skin_texture[::-1, :, ::-1]

		self.texture_width = skin_texture.shape[1]
		self.texture_height = skin_texture.shape[0]

		self.skin_texture = self.ctx.texture((skin_texture.shape[1], skin_texture.shape[0]), skin_texture.shape[2], data=skin_texture.tobytes())
		self.distance_shader['skin_texture'].value = 0

		# Assign projection matrix
		self.projection_matrix = self.distance_shader["projection_matrix"]
		self.projection_matrix.value = tuple(projection_matrix.T.reshape(-1))
		print(projection_matrix)

		# Construct a list of model_view matrices that circle around the origin
		self.model_view_matrices = []

		radius = 2
		height = 1
		self.num_steps = 10

		steps = np.arange(self.num_steps)
		angle = steps / self.num_steps * np.pi * 2
		xy = np.column_stack([np.cos(angle) * radius, np.sin(angle) * radius])
		target_position = np.array([0, 0, height], dtype=np.float64)
		up_vector = np.array([0, 0, 1], dtype=np.float64)

		for x, y in xy:
			eye_position = np.array([x, y, height], dtype=np.float64)
			model_view_matrix = look_at_mat4(eye_position, target_position, up_vector)

			self.model_view_matrices.append(model_view_matrix)

		self.model_view_matrix = self.distance_shader['model_view_matrix']

		# Create a renderbuffer and associate it to a framebuffer for distance_shader output
		self.distance_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
		self.distance_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
		self.distance_shader_framebuffer = self.ctx.framebuffer(self.distance_shader_color_texture, self.distance_shader_depth_texture)

		# Create an akaze detector
		self.detector = cv2.AKAZE_create()

		# Capture test descriptors from first video frame
		self.test_frame_keypoints_descriptors = get_keypoints_and_descriptors(video_image, self.detector)
		self.test_frame = video_image

		self.frame_count = 0

	def render(self, _time, _frametime):
		# Move the camera around the origin
		self.model_view_matrix.value = tuple(self.model_view_matrices[self.frame_count].T.reshape(-1))

		# Clear buffers
		self.ctx.clear(0.0, 0.0, 0.0, 0.0)
		self.distance_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)

		# Find pixel distance from the blender camera
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.enable(moderngl.DEPTH_TEST)
		# self.distance_shader_framebuffer.use()
		self.skin_texture.use(0)
		self.distance_shader_framebuffer.use()
		self.distance_vao.render()

		# Read render data to cpu
		raw = self.distance_shader_color_texture.read()
		cpu_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))
		cpu_image = cv2.cvtColor(cpu_image, cv2.COLOR_RGBA2BGRA)
		cpu_image = cpu_image[::-1, :, :]


		# Find akaze points
		keypoints_descriptors = get_keypoints_and_descriptors(cpu_image, self.detector)

		match_descriptors(keypoints_descriptors, self.test_frame_keypoints_descriptors, (self.test_frame / 255, cpu_image[:, :, :3]))

		cv2.imshow('image', cpu_image[::4, ::4, :])

		self.frame_count += 1
		if self.frame_count == self.num_steps:
			self.wnd.close()

	def resize(self, width, height):
		self.ctx.viewport = (0, 0, width, height)

if __name__ == '__main__':
	Test.run()