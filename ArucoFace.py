import cv2 # Apparently if you don't import cv2 before bpy, you can't access cv2.aruco :P
import bpy
import numpy as np

# Load the predefined dictionary
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
aruco_parameters = cv2.aruco.DetectorParameters_create()
# aruco_parameters.adaptiveThreshWinSizeMax = 400

# Generate the marker
marker_size = 200
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
cv2.aruco.drawMarker(aruco_dictionary, 33, marker_size, marker_image, 1)

pad_size = 100
padded_size = marker_size + 2 * pad_size
marker_image_padded = np.ones((padded_size, padded_size), dtype=np.uint8) * 255
marker_image_padded[pad_size:-pad_size, pad_size:-pad_size] = marker_image
# Add a small black border so we know where to cut them out
marker_image_padded[0, :] = 0
marker_image_padded[-1, :] = 0
marker_image_padded[:, 0] = 0
marker_image_padded[:, -1] = 0

cv2.imwrite(bpy.path.abspath('//external_scripts/tmp/marker_33.png'), marker_image_padded)

cap = cv2.VideoCapture(bpy.path.abspath('//external_scripts/tmp/aruco_test.mp4'))

while True:
	ret, image = cap.read()
	if not ret:
		break

	image = image[::20, ::20, :]

	corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dictionary, parameters=aruco_parameters)

	marked = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
	# rejected = cv2.aruco.drawDetectedMarkers(image.copy(), rejected, np.array([1] * len(rejected)))

	cv2.imshow("Marked", marked)
	cv2.waitKey(20)