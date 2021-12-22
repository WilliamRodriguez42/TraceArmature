import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import pdb

import helper_functions as hf

class MarkerGenerator():
	def __init__(
		self,
		num_bits=8,
		size=256,
		blank_radius=0.25,
		white_radius=0.4,
		orientation_region_radius=0.6,
	):
		self.size = size
		self.blank_radius = blank_radius
		self.white_radius = white_radius
		self.orientation_region_radius = orientation_region_radius

		axis_positions = np.arange(self.size)
		pixel_positions = np.zeros((self.size, self.size, 2), dtype=np.float32) # axis 2 contains the i, j coordinates of each pixel
		pixel_positions[:, :, 0] = axis_positions[:, np.newaxis]
		pixel_positions[:, :, 1] = axis_positions[np.newaxis, :]

		pixel_positions /= self.size-1 # Convert i, j coordinates to [0, 1] range
		pixel_positions -= 0.5 # Move origin to center of image
		pixel_positions *= 2 # Get to range [-1, 1]

		pixel_polar_angle = np.arctan2(pixel_positions[:, :, 1], pixel_positions[:, :, 0])
		pixel_polar_angle[pixel_polar_angle < 0] += 2*np.pi # Convert from [-pi, pi] to [0, 2*pi]
		self.pixel_polar_hypotenuse = np.linalg.norm(pixel_positions, axis=2)

		# If n bits were to be evenly distributed as wedges on a circle, which pixels would represent which bit offset?
		self.pixel_bit_mask = np.zeros((self.size, self.size), dtype=np.uint8) # Each value represents a bit mask for just 1 bit. This is the bit that the pixel represents

		for i in range(num_bits):
			bit_mask = 1 << i

			min_angle = 2*np.pi * i / num_bits
			max_angle = 2*np.pi * (i+1) / num_bits

			within_angle_range = np.logical_and(pixel_polar_angle > min_angle, pixel_polar_angle <= max_angle)
			self.pixel_bit_mask[within_angle_range] = bit_mask

		binary_array = np.zeros(num_bits, dtype=bool)
		binary_array[::2] = 1 # Alternate zeros and ones
		binary_array[0:2] = 0 # First two bits must be zero
		binary_arange = np.arange(num_bits)
		binary_powers = np.power(2, binary_arange)
		self.orientation_encoding = np.dot(binary_array, binary_powers)

		# self.orientation_encoding = 0b01010100 # Must satisfy the following requirements: first bit (furthest right) is 0, second bit is 0, third bit is 1, then alternate for the rest


	def generate(self, code):

		# Define the white region
		white_region_mask = np.logical_and(self.pixel_polar_hypotenuse > self.blank_radius, self.pixel_polar_hypotenuse <= self.white_radius)

		# Define orientation encoding region
		orientation_region_mask = np.logical_and(self.pixel_polar_hypotenuse > self.white_radius, self.pixel_polar_hypotenuse <= self.orientation_region_radius)

		# Define the bit encoding region
		encoding_region_mask = np.logical_and(self.pixel_polar_hypotenuse > self.orientation_region_radius, self.pixel_polar_hypotenuse <= 1)

		marker_image = np.zeros((self.size, self.size), dtype=np.uint8)
		marker_image[white_region_mask] = 255
		marker_image[orientation_region_mask] = (self.pixel_bit_mask[orientation_region_mask] & self.orientation_encoding).astype(bool) * 255
		marker_image[encoding_region_mask] = (self.pixel_bit_mask[encoding_region_mask] & code).astype(bool) * 255

		return marker_image

if __name__ == '__main__':
	size = 256
	margin = 10

	page_width = int(10 * (size + margin) + margin)
	page_height = int(7 * (size + margin) + margin)

	page_margin = 100

	page_with_border = np.zeros((page_height + page_margin*2, page_width + page_margin*2), dtype=np.uint8)
	page_with_border[1:-1, 1:-1] += 255
	page = page_with_border[page_margin:-page_margin, page_margin:-page_margin]
	marker_generator = MarkerGenerator(7, size)

	i = margin
	j = margin
	k = 0
	while i+size < page_height:
		j = margin
		while j+size < page_width:
			marker_image = 255 - marker_generator.generate(k) # Invert image so black on white, easier to print
			page[i:i+size, j:j+size] = marker_image

			j += size + margin
			k += 1
		i += size + margin

	cv2.imwrite('tmp/marker_test_page.png', page_with_border)

	marker_image = marker_generator.generate(0x55)

	cv2.imshow('marker_image', marker_image)
	cv2.waitKey(0)

	marker_image = marker_image[::-1, :] # Flip Y axis since moderngl expects 0, 0 to be in bottom left

	import moderngl
	import moderngl_window

	class Test(moderngl_window.WindowConfig):
		gl_version = (4, 4)

		title = 'Triangle'
		resizable = True
		aspect_ratio = None

		def __init__(self, **kwargs):
			super().__init__(**kwargs)


			self.color_shader = self.ctx.program(
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
						float texture_color = texture(skin_texture, frag_uv).r;
						color = vec4(texture_color, texture_color, texture_color, 1);
					}
				''',
			)

			# Create vertex attribute array
			vertices = np.array([
				[-25.4/1000, 0,  25.4/1000, 1],
				[ 25.4/1000, 0,  25.4/1000, 1],
				[-25.4/1000, 0, -25.4/1000, 1],
				[ 25.4/1000, 0,  25.4/1000, 1],
				[ 25.4/1000, 0, -25.4/1000, 1],
				[-25.4/1000, 0, -25.4/1000, 1],
			], dtype=np.float64)

			uvs = np.array([
				[0, 1],
				[1, 1],
				[0, 0],
				[1, 1],
				[1, 0],
				[0, 0],
			], dtype=np.float64)

			np.random.seed(0)

			# Modify vertices by a random rotation matrix and place them in a grid with random depth
			grid_shape = (20, 20)
			transformation_matrices = np.zeros((grid_shape[0], grid_shape[1], 4, 4), dtype=np.float64)
			forward = np.random.rand(grid_shape[0], grid_shape[1], 3) * 2 - 1
			forward[:, :, 1] = forward[:, :, 1] * 0.5 + 1.5
			forward /= np.linalg.norm(forward, axis=2)[:, :, np.newaxis]

			right = np.random.rand(grid_shape[0], grid_shape[1], 3) * 2 - 1
			component_in_forward = np.matmul(forward[:, :, np.newaxis, :], right[:, :, :, np.newaxis])
			right -= component_in_forward[:, :, 0, :] # Subtract component in forward direction
			right /= np.linalg.norm(right, axis=2)[:, :, np.newaxis]

			up = np.cross(right, forward)
			up /= np.linalg.norm(up, axis=2)[:, :, np.newaxis]

			translations = np.zeros((grid_shape[0], grid_shape[1], 3), dtype=np.float64)
			translations[:, :, 0] = np.arange(grid_shape[0])[:, np.newaxis] / (grid_shape[0] - 1) - 0.5
			translations[:, :, 2] = np.arange(grid_shape[1])[np.newaxis, :] / (grid_shape[1] - 1) - 0.5
			translations[:, :, 0] *= 3
			translations[:, :, 2] *= 6

			translations[:, :, 1] -= np.random.rand(*grid_shape) * 2

			transformation_matrices[:, :, :3, 0] = right
			transformation_matrices[:, :, :3, 1] = forward
			transformation_matrices[:, :, :3, 2] = up
			# transformation_matrices[:, :, 0, 0] = 1
			# transformation_matrices[:, :, 1, 1] = 1
			# transformation_matrices[:, :, 2, 2] = 1
			transformation_matrices[:, :, 3, 3] = 1
			transformation_matrices[:, :, 3, :3] = translations

			vertices = np.matmul(vertices[np.newaxis, np.newaxis, :, :], transformation_matrices)

			# pdb.set_trace()
			uvs = np.tile(uvs[np.newaxis, np.newaxis, :, :], (grid_shape[0], grid_shape[1], 1, 1))

			self.color_vertex_vbo = self.ctx.buffer(vertices.astype(np.float32).tobytes())
			self.color_uv_vbo = self.ctx.buffer(uvs.astype(np.float32).tobytes())
			self.color_vao = self.ctx.vertex_array(
				self.color_shader,
				[
					(self.color_vertex_vbo, '4f', 'vertex'),
					(self.color_uv_vbo, '2f', 'vert_uv'),
				],
			)

			# Load and assign texture
			# skin_texture = cv2.imread(texture_path)
			# skin_texture = skin_texture[::-1, :, ::-1]
			skin_texture = marker_image

			self.texture_width = skin_texture.shape[1]
			self.texture_height = skin_texture.shape[0]

			self.skin_texture = self.ctx.texture((skin_texture.shape[1], skin_texture.shape[0]), 1, data=skin_texture.tobytes())
			self.color_shader['skin_texture'].value = 0

			# Assign projection matrix
			projection_matrix = np.array([
				[ 2.8885038, 0.       , 0.       , 0.       ],
				[ 0.       , 1.5232345, 0.       , 0.       ],
				[ 0.       , 0.       ,-1.000528 ,-1.       ],
				[ 0.       , 0.       ,-0.2000528, 0.       ],
			], dtype=np.float32)
			self.projection_matrix = self.color_shader["projection_matrix"]
			self.projection_matrix.value = tuple(projection_matrix.T.reshape(-1))

			# Construct a list of model_view matrices that circle around the origin
			self.model_view_matrices = []

			radius = 23
			height = 0
			self.num_steps = 1

			xy = np.column_stack([0, -7])
			target_position = np.array([0, 0, height], dtype=np.float64)
			up_vector = np.array([0, 0, 1], dtype=np.float64)

			for x, y in xy:
				eye_position = np.array([x, y, height], dtype=np.float64)
				model_view_matrix = hf.look_at_mat4(eye_position, target_position, up_vector)

				self.model_view_matrices.append(model_view_matrix)

			self.model_view_matrix = self.color_shader['model_view_matrix']

			# Create a renderbuffer and associate it to a framebuffer for color_shader output
			self.video_width = 2160
			self.video_height = 4096
			self.color_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
			self.color_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
			self.color_shader_framebuffer = self.ctx.framebuffer(self.color_shader_color_texture, self.color_shader_depth_texture)

			self.frame_count = 0

		def render(self, _time, _frametime):
			# Move the camera around the origin
			self.model_view_matrix.value = tuple(self.model_view_matrices[self.frame_count].T.reshape(-1))

			# Clear buffers
			self.ctx.clear(0.0, 0.0, 0.0, 0.0)
			self.color_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)

			# Find pixel distance from the blender camera
			self.ctx.disable(moderngl.CULL_FACE)
			self.ctx.enable(moderngl.DEPTH_TEST)
			self.wnd.use()
			self.skin_texture.use(0)
			self.color_vao.render()

			# Find pixel distance from the blender camera
			self.ctx.disable(moderngl.CULL_FACE)
			self.ctx.enable(moderngl.DEPTH_TEST)
			self.color_shader_framebuffer.use()
			self.skin_texture.use(0)
			self.color_vao.render()

			# Read render data to cpu
			raw = self.color_shader_color_texture.read()
			cpu_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))
			# cpu_image = cv2.cvtColor(cpu_image, cv2.COLOR_RGBA2BGRA)
			cpu_image = cpu_image[::-1, :, :].copy()
			cpu_image *= 255
			cpu_image = cpu_image.astype(np.uint8)

			cv2.imwrite(f'tmp/marker_output/image_{self.frame_count}.png', cpu_image)

			self.frame_count += 1
			if self.frame_count == self.num_steps:
				self.wnd.close()

		def resize(self, width, height):
			self.ctx.viewport = (0, 0, width, height)

	Test.run()
