import cv2
import numpy as np
import pickle
import requests
import mediapipe as mp
import os
import pdb
import helper_functions as hf
import pygame
print = hf.print

if __name__ == '__main__':

	pygame.init()

	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles
	mp_hands = mp.solutions.hands

	# Load the MVPs
	# parent_directory = os.path.dirname(__file__)
	# resources_directory = os.path.abspath(os.path.join(parent_directory, '..', 'resources'))

	# MVPs_file_path = os.path.join(resources_directory, 'MVPs.npy')
	# MVPs = np.load(MVPs_file_path)

	# MVP_is = np.linalg.inv(MVPs) # Get inverse model view projection matrices

	MVPs = hf.read_pickled()
	MVP_is = np.linalg.inv(MVPs)

	# Load images and find hands
	camera_landmarks = {} # Using a dict because some cameras might not pick up any hands

	with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=1,
		min_detection_confidence=0.7,
	) as hands:

		results = requests.post('http://localhost:5000/shoot')
		results = pickle.loads(results.content)

		images = [cv2.imdecode(result, cv2.IMREAD_COLOR) for result in results]
		image_surfs = []

		for i, image in enumerate(images):
			results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(
						image,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style(),
					)

					camera_landmarks[i] = np.array([(landmark.x, landmark.y) for landmark in hand_landmarks.landmark], dtype=np.float64)

			cv2.imwrite(rf'C:\Users\William\Desktop\HandPictures\image{i}.png', image)

			new_size = np.array(image.shape) / 3
			new_size = (int(new_size[1]), int(new_size[0]))

			image = cv2.resize(image, new_size)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.astype(np.uint8)
			image = np.swapaxes(image, 0, 1)

			image_surf = pygame.surfarray.make_surface(image)
			image_surfs.append(image_surf)

		# 	cv2.imshow(f'image {i}', cv2.resize(image, new_size[::-1]))

		# cv2.waitKey(0)


	# Select which cameras to use
	display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
	clock = pygame.time.Clock()
	running = True

	dark = pygame.Surface(image_surf.get_size()).convert_alpha()
	dark.fill((0, 0, 0, 128))

	disabled = [False] * 5

	while running:
		display.fill((0, 0, 0))

		just_pressed = False
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_RETURN:
					running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					just_pressed = True

		for i, image_surf in enumerate(image_surfs):
			if i < 3:
				position = (i * new_size[0], 0)
			else:
				position = ((i-3) * new_size[0], new_size[1])

			rect = image_surf.get_rect()
			rect.topleft = position

			display.blit(image_surf, position)
			if rect.collidepoint(pygame.mouse.get_pos()):
				display.blit(dark, position)

				if just_pressed:
					disabled[i] = not disabled[i]

			if disabled[i]:
				display.blit(dark, position)

		clock.tick(60)
		pygame.display.flip()

	# Filter cameras that were disabled
	for i in list(camera_landmarks.keys()):
		if disabled[i]:
			del camera_landmarks[i]

	# with open(r'C:\Users\William\Desktop\HandPictures\camera_landmarks.pkl', 'wb+') as f:
	# 	f.write(pickle.dumps(camera_landmarks))

	# with open(r'C:\Users\William\Desktop\HandPictures\camera_landmarks.pkl', 'rb') as f:
	# 	camera_landmarks = pickle.loads(f.read())

	# Create rays corresponding with the landmarks found by each camera
	num_usable_cameras = len(camera_landmarks.keys()) # Number of cameras that captured a hand
	print(num_usable_cameras)
	landmarks_os = np.zeros((21, num_usable_cameras, 3), dtype=np.float64) # Raycast origins
	landmarks_ds = np.zeros((21, num_usable_cameras, 3), dtype=np.float64) # Raycast directions

	current_camera = 0 # Index of the current USABLE camera
	for i, landmarks in list(camera_landmarks.items()):
		MVP_i = MVP_is[i] # Get inverse model view projection matrix associated with camera i

		# image = cv2.imread(rf'C:\Users\William\Desktop\HandPictures\image{i}.png')

		# landmarks *= (1920, 1080)
		# landmarks = landmarks.astype(np.int64)

		# for j in range(21):
		# 	landmark = landmarks[j]
		# 	cv2.circle(image, landmark, 10, (0, 0, 255), 4)

		# cv2.imshow('image', image)
		# cv2.waitKey(0)

		# Convert landmark range from [0, 1] to [-1, 1]
		landmarks *= 2
		landmarks -= 1
		landmarks[:, 1] *= -1

		p1, p2 = hf.ray(landmarks, MVP_i)

		for j in range(21):
			landmarks_os[j, current_camera, :] = p1[j, :3]
			landmarks_ds[j, current_camera, :] = p2[j, :3] - p1[j, :3] # Normalize this later

		current_camera += 1

	# Normalize landmark directions
	landmarks_ds /= np.linalg.norm(landmarks_ds, axis=2)[:, :, np.newaxis]

	# Iteratively improve a predicted answer
	def improving_vector(o, d, p):
		# temp = (d*(o - p[:, np.newaxis, :])).sum(axis=2)
		# improving_vectors = -d*temp[:, :, np.newaxis] + o - p[:, np.newaxis, :]

		ox = o[:, :, 0]
		oy = o[:, :, 1]
		oz = o[:, :, 2]

		dx = d[:, :, 0]
		dy = d[:, :, 1]
		dz = d[:, :, 2]

		px = p[:, np.newaxis, 0]
		py = p[:, np.newaxis, 1]
		pz = p[:, np.newaxis, 2]

		ix = -2*dx*dy*(dy*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oy + py) - 2*dx*dz*(dz*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oz + pz) - 2*(dx**2 - 1)*(dx*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - ox + px)
		iy = -2*dx*dy*(dx*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - ox + px) - 2*dy*dz*(dz*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oz + pz) - 2*(dy**2 - 1)*(dy*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oy + py)
		iz = -2*dx*dz*(dx*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - ox + px) - 2*dy*dz*(dy*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oy + py) - 2*(dz**2 - 1)*(dz*(dx*(ox - px) + dy*(oy - py) + dz*(oz - pz)) - oz + pz)

		improving_vectors = -np.concatenate([ix[:, :, np.newaxis], iy[:, :, np.newaxis], iz[:, :, np.newaxis]], axis=2)

		return improving_vectors.sum(axis=1)

	ps = np.zeros((21, 3), dtype=np.float64)
	for i in range(1000):
		ps += improving_vector(landmarks_os, landmarks_ds, ps) * 0.05

	hf.write_pickled(ps)