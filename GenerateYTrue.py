import bpy
import numpy as np
import pdb
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def generate_y_true(video_path):

	with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.9) as face_mesh:

		cap = cv2.VideoCapture(video_path)

		y_true = []
		valid_frames = []

		frame_count = 0
		while cap.isOpened():
			success, frame = cap.read()
			if not success:
				break

			face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

			face_image.flags.writeable = False
			results = face_mesh.process(face_image)
			face_image.flags.writeable = True

			face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
			if results.multi_face_landmarks:
				face_landmarks = results.multi_face_landmarks[0]

				vertices_camera = np.zeros((468, 2), dtype=np.float32)
				y_world = np.zeros(468, dtype=np.float32)

				for i, landmark in enumerate(face_landmarks.landmark):
					x = (landmark.x - 0.5) * 2
					y = (0.5 - landmark.y) * 2

					vertices_camera[i, 0] = x
					vertices_camera[i, 1] = y

				y_true.append(vertices_camera)
				valid_frames.append(frame_count)

				mp_drawing.draw_landmarks(
					image=face_image,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACE_CONNECTIONS,
					landmark_drawing_spec=drawing_spec,
					connection_drawing_spec=drawing_spec)

				cv2.imshow('MediaPipe FaceMesh', face_image)
				cv2.waitKey(1)

			frame_count += 1

		y_true = np.array(y_true, dtype=np.float32)
		valid_frames = np.array(valid_frames, dtype=np.int64)

		cv2.destroyAllWindows()

		return y_true, valid_frames