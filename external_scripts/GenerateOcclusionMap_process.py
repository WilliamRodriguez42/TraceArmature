import moderngl
import moderngl_window
import numpy as np
import cv2
import os

from helper_functions import print, start_video_at_frame, get_next_frame, read_pickled, write_pickled

class Test(moderngl_window.WindowConfig):
	gl_version = (4, 4)

	title = 'Triangle'
	resizable = True
	aspect_ratio = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

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
				in vec3 screen_uv; // vec3 because 3rd element is blender camera distance
				out float blender_depth; // AI approximated blender camera distance

				uniform float pixel_min_depth;
				uniform float pixel_max_range;

				void main() {
					blender_depth = screen_uv.z; // This guy will end up being our color output

					// This is what I was talking about, this guy will be our gl_Position depth
					float pixel_depth_0_to_1 = (blender_depth - pixel_min_depth) / pixel_max_range;

					vec2 screen_pos_2d = screen_uv.xy * 2 - 1;

					gl_Position = vec4(screen_pos_2d, pixel_depth_0_to_1, 1.0);
				}
			''',
			fragment_shader='''
				#version 440
				out float color;
				in float blender_depth;

				void main() {
					color = blender_depth;
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
					uv = tri_uv;
					screen_uv = tri_screen_uv;

					vec2 screen_coords = tri_uv * 2 - 1;

					gl_Position = vec4(screen_coords, 0, 1);
				}
			''',
			fragment_shader='''
				#version 440
				out vec4 color;
				in vec2 uv;
				in vec3 screen_uv;

				uniform sampler2D video_texture;
				uniform sampler2D green_screen_texture; // Needed for glove occlusions
				uniform sampler2D distance_texture;

				void main() {
					float min_distance_for_pixel = texture(distance_texture, screen_uv.xy).r;
					float distance_for_frag = screen_uv.z;

					vec4 green_screen_color = texture(green_screen_texture, screen_uv.xy);

					// Occluded if there is geometry in the way
					bool is_closest = distance_for_frag < min_distance_for_pixel + 0.1f; // Add a small value because distance_texture doesn't have a one to one mapping to this fragment, so we need a bit of wiggle room in the comparison to prevent aliasing

					bool is_glove = (green_screen_color.b > 0.5) || (green_screen_color.r > 0.5); // Or occluded if gloves are present

					if (is_closest && !is_glove) {
						color = texture(video_texture, screen_uv.xy).bgra; // Use video color
					} else if (is_glove) {
						color = vec4(1, 1, 0, 0); // Yellow with no alpha for glove occlusions
					} else {
						color = vec4(1, 0, 1, 0); // Any RGB value with no alpha will indicate an occluded area
					}
				}
			''',
		)

		self.head_shader = self.ctx.program(
			vertex_shader='''
				#version 440
				in vec3 screen_uv; // These are the screen uv coordinates of everything EXCEPT the head
				in float is_non_head; // This is the simplest way I could think of to remove a fragment if just one vertex belongs to the head
				out float is_non_head_frag;

				uniform float pixel_min_depth;
				uniform float pixel_max_range;

				void main() {
					is_non_head_frag = is_non_head;

					float pixel_depth_0_to_1 = (screen_uv.z - pixel_min_depth) / pixel_max_range;

					vec2 screen_pos_2d = screen_uv.xy * 2 - 1;

					gl_Position = vec4(screen_pos_2d, pixel_depth_0_to_1, 1.0);
				}
			''',
			fragment_shader='''
				#version 440
				out vec4 color;
				in float is_non_head_frag;

				void main() {
					if (is_non_head_frag < 0.99) {
						color = vec4(1, 0, 1, 1); // Any fragment that consists of any body vertices will be purple
					} else {
						color = vec4(0, 0, 0, 1); // Any fragment with a vertex belonging to the head will be 0s
					}
				}
			''',
		)


		(
			self.tri_vertex_indices, # tri_vertex_indices maps blender loops (which I call tris since they should all have length 3 now) to the corresponding set of vertices (or more importantly for this script the respective screen_uv)
			self.tri_uvs, # tri_uvs are blender's UVMap values for each blender loop, you can see this value in blender by going to the uv editor and viewing the UVMap for me.high_poly.optimized
			self.is_non_head,
			self.video_path,
			self.green_screen_path,
			self.video_crop_transpose,
			self.start_frame,
			self.end_frame,
			self.output_path,
			self.head_image_resolution,
		) = read_pickled()

		self.frame = self.start_frame # Set current frame to start frame

		# Load video and get video properties
		self.video_capture = start_video_at_frame(self.video_path, self.frame)
		self.green_screen_capture = start_video_at_frame(self.green_screen_path, self.frame)

		self.video_texture = None # This will be a moderngl texture once we load the first image
		self.green_screen_texture = None

		self.video_width  = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

		# Create vertex attribute array with screen uvs with depth for the depth shader
		self.distance_ibo = self.ctx.buffer(self.tri_vertex_indices.tobytes())

		self.distance_vbo = None # Cannot allocate vbo or vao without the first frame's screen_uvs, these will be created at first render call
		self.distance_vao = None

		# Create a renderbuffer and associate it to a framebuffer for distance_shader output
		self.distance_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 1, dtype='f4')
		self.distance_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
		self.distance_shader_framebuffer = self.ctx.framebuffer(self.distance_shader_color_texture, self.distance_shader_depth_texture)



		# Create placeholder buffers for uv shader
		self.uv_vbo = None # These will be created on first render call
		self.uv_vao = None

		# Set layout indices of textures
		self.uv_shader['video_texture'].value = 0
		self.uv_shader['green_screen_texture'].value = 1
		self.uv_shader['distance_texture'].value = 2

		# User uv shader framebuffer to enable data viewing from cpu
		self.uv_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
		self.uv_shader_framebuffer = self.ctx.framebuffer(self.uv_shader_color_texture) # No depth texture is necessary



		# Create placeholder buffers for head shader
		self.head_ibo = self.ctx.buffer(self.tri_vertex_indices.tobytes())

		self.head_vbo = None # This will be created on first render call
		self.head_vao = None

		# User uv shader framebuffer to enable data viewing from cpu
		self.head_shader_color_texture = self.ctx.texture((self.video_width, self.video_height), 4, dtype='f4')
		self.head_shader_depth_texture = self.ctx.depth_texture((self.video_width, self.video_height))
		self.head_shader_framebuffer = self.ctx.framebuffer(self.head_shader_color_texture, self.head_shader_depth_texture)



		# Create two output video captures to write to
		fps = self.video_capture.get(cv2.CAP_PROP_FPS)

		magenta_occlusion_path = os.path.join(self.output_path, 'magenta_occlusion.avi')
		self.magenta_occlusion_writer = cv2.VideoWriter(magenta_occlusion_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.video_width, self.video_height)) # This result can be imported to blender directly to give an idea of what areas are occluded


	def render(self, time, frametime):
		# Read frame args from parent process
		(
			screen_uvs, # screen_uvs are uv coordinates on the range [0, 1] generated by projecting the character's vertices by the intrinsic/extrinsic matrices, there is one per vertex
			bbox,
		) = read_pickled()

		# These will be used to normalize the pixel depth to range (0, 1) for DEPTH_TEST
		pixel_min_depth = screen_uvs[:, 2].min()
		pixel_max_range = screen_uvs[:, 2].max() - pixel_min_depth
		self.distance_shader['pixel_min_depth'].value = pixel_min_depth
		self.distance_shader['pixel_max_range'].value = pixel_max_range
		self.head_shader['pixel_min_depth'].value = pixel_min_depth
		self.head_shader['pixel_max_range'].value = pixel_max_range


		# Update distance shader vbo with current frame screen_uvs
		if self.distance_vbo is None:
			self.distance_vbo = self.ctx.buffer(screen_uvs.tobytes())
			self.distance_vao = self.ctx.vertex_array(
				self.distance_shader,
				[(self.distance_vbo, '3f', 'screen_uv')],
				self.distance_ibo
			)
		else:
			self.distance_vbo.write(screen_uvs.tobytes())


		# Get screen_uv coordinates for each tri
		tri_screen_uvs = screen_uvs[self.tri_vertex_indices]

		# Interleave tri uvs and tri screen uvs
		tri_uvs_and_tri_screen_uvs = np.concatenate([self.tri_uvs, tri_screen_uvs], axis=2)

		# Update uv shader vbo with current frame tri_uvs_and_screen_uvs
		if self.uv_vbo is None:
			self.uv_vbo = self.ctx.buffer(tri_uvs_and_tri_screen_uvs.tobytes())

			self.uv_vao = self.ctx.vertex_array(
				self.uv_shader,
				[(self.uv_vbo, '2f 3f', 'tri_uv', 'tri_screen_uv')],
			)
		else:
			self.uv_vbo.write(tri_uvs_and_tri_screen_uvs.tobytes())


		# Interleave screen_uvs with is non head
		screen_uvs_and_is_non_head = np.concatenate([screen_uvs, self.is_non_head[:, np.newaxis]], axis=1)

		# Update head shader vbo with current frame screen_uvs and is_non_head
		if self.head_vbo is None:
			self.head_vbo = self.ctx.buffer(screen_uvs_and_is_non_head.tobytes())
			self.head_vao = self.ctx.vertex_array(
				self.head_shader,
				[(self.head_vbo, '3f 1f', 'screen_uv', 'is_non_head')],
				self.head_ibo
			)
		else:
			self.head_vbo.write(screen_uvs_and_is_non_head.tobytes())


		# Clear buffers
		self.ctx.clear(0.0, 0.0, 0.0, 0.0)
		self.distance_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)
		self.uv_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)
		self.head_shader_framebuffer.clear(0.0, 0.0, 0.0, 0.0)

		# Load video frame
		ret, video_image = get_next_frame(self.video_capture, self.video_crop_transpose)
		_, green_screen_image = get_next_frame(self.green_screen_capture, self.video_crop_transpose)
		if not ret: # End of video
			self.wnd.close()

		if self.video_texture is None:
			self.video_texture = self.ctx.texture((self.video_width, self.video_height), video_image.shape[2], data=video_image.tobytes())
			self.green_screen_texture = self.ctx.texture((self.video_width, self.video_height), video_image.shape[2], data=green_screen_image.tobytes())
		else:
			self.video_texture.write(video_image.tobytes())
			self.green_screen_texture.write(green_screen_image.tobytes())


		# Find pixel distance from the blender camera
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.enable(moderngl.DEPTH_TEST)
		self.distance_shader_framebuffer.use()
		self.distance_vao.render()

		# Create a uv unwrapped image showing which parts of the model are visible
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.disable(moderngl.DEPTH_TEST)
		self.uv_shader_framebuffer.use()
		self.video_texture.use(0)
		self.green_screen_texture.use(1)
		self.distance_shader_color_texture.use(2)
		self.uv_vao.render()

		# Just redoing that last step but rendering to the screen and not a framebuffer
		self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.disable(moderngl.DEPTH_TEST)
		self.wnd.use()
		self.video_texture.use(0)
		self.green_screen_texture.use(1)
		self.distance_shader_color_texture.use(2)
		self.uv_vao.render()

		# Rendering head occlusions
		# self.ctx.disable(moderngl.CULL_FACE)
		# self.ctx.enable(moderngl.DEPTH_TEST)
		# self.head_shader_framebuffer.use()
		# self.head_vao.render()



		# Read result to numpy
		raw = self.uv_shader_color_texture.read()
		cpu_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))
		cpu_image = cv2.cvtColor(cpu_image, cv2.COLOR_BGRA2RGBA)

		# # An AI needs to fill in the occluded areas
		# # For now just replace them with magenta
		# occluded = np.logical_and(cpu_image.any(axis=2), cpu_image[:, :, 3] == 0) # Any RGB value with no alpha indicates an occluded area
		# cpu_image[occluded, :] = (1, 0, 1, 1) # At the end of this step, all pixel values with 0 alpha will also have 0 color, which is vital for the next steps

		# # Texture padding, requires alpha is 0 for background, or 1 for valid uv, no decimals
		# valid = cpu_image[:, :, 3] == 0

		# blurred = cv2.GaussianBlur(cpu_image, (0, 0), 5, 5) # (TODO: Config blur sigma)
		# color_corrected_blurred = blurred / (blurred[:, :, 3, np.newaxis] + 1e-20) # The edges of the image get darker with a gaussian blur on a black backround, but if we divide be the sum of the non-backround weights (a.k.a. the blurred alpha channel), the brightness should come back

		# # Overlay the original image on top of the color corrected blurred
		# valid = cpu_image.any(axis=2) # If there was any color or alpha in the original cpu_image, then we need to keep that value in the final result
		# padded = color_corrected_blurred # Just naming to something better
		# padded[valid] = cpu_image[valid] # Overlay

		# padded = (padded * 255).astype(np.uint8)
		# self.magenta_occlusion_writer.write(padded[::-1, :, :3])

		# cpu_image = (cpu_image * 255).astype(np.uint8)
		# self.magenta_occlusion_writer.write(cpu_image[::-1, :, :3])



		# # Read head shader data to numpy
		# raw = self.head_shader_color_texture.read()
		# head_shader_image = np.frombuffer(raw, dtype=np.float32).reshape((self.video_height, self.video_width, 4))

		# # Crop out head
		# top, right, bottom, left = bbox
		# head_video_image = video_image[top:bottom, left:right]
		# head_green_screen_image = green_screen_image[top:bottom, left:right]
		# head_shader_image = head_shader_image[top:bottom, left:right]

		# width = right - left
		# height = bottom - top

		# # Any green screen, gloves, or body parts should be masked out
		# # chroma_mask = head_green_screen_image.any(axis=2)
		# body_mask = np.logical_not(head_shader_image[:, :, :3].any(axis=2))
		# # crop_mask = np.zeros((height, width), dtype=bool)
		# # crop_mask[top:bottom, left:right] = False
		# # mask = np.logical_or(np.logical_or(chroma_mask, body_mask), crop_mask)
		# mask = body_mask

		# alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
		# head_image = np.concatenate([head_video_image, alpha], axis=2)
		# head_image[mask, :] = (255, 0, 255, 0) # Chroma'd out color will be replaced with purple, no alpha

		# # head_image = cv2.resize(head_image, self.head_image_resolution) # Resize the result to a standard
		# # mask_image = cv2.resize(mask_image, self.head_image_resolution)

		# cv2.imwrite(f"{self.output_path}/head_output/head_{self.frame}.png", head_image)

		# cv2.imshow('Head', head_image)
		# cv2.waitKey(1)



		# Increment frame counter
		if self.frame == self.end_frame:
			self.wnd.close()
		self.frame += 1

		print(f"Completed {self.frame} frames")

	def resize(self, width, height):
		self.ctx.viewport = (0, 0, width, height)

	def close(self):
		self.video_capture.release()
		self.green_screen_capture.release()
		self.magenta_occlusion_writer.release()


if __name__ == '__main__':

	Test.run()