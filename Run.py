from utils import config
from utils.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils, cv2, os
import streamlit as st

st.write("""# Social Distancing Detection""")
st.write("""Drop any image that contains humans and check whether social distancing is violated!""")

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = st.sidebar.file_uploader("Upload Image")

if img is not None:
	# Convert the files to an opencv images
	file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
	frame = cv2.imdecode(file_bytes, 1)

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the max/min social distance limits
	serious = set()
	abnormal = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of the centroid pairs
					serious.add(i)
					serious.add(j)
				# update our abnormal set if the centroid distance is below max distance limit
				if (D[i, j] < config.MAX_DISTANCE) and not serious:
					abnormal.add(i)
					abnormal.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation/abnormal sets, then update the color
		if i in serious:
			color = (0, 0, 255)
		elif i in abnormal:
			color = (0, 255, 255) #orange = (0, 165, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 2)

	# draw some of the parameters
	Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
	cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
	Threshold = "Threshold limit: {}".format(config.Threshold)
	cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

	# draw the total number of social distancing violations on the output frame
	text = "Total serious violations: {}".format(len(serious))
	cv2.putText(frame, text, (10, frame.shape[0] - 55),
		cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

	text1 = "Total abnormal violations: {}".format(len(abnormal))
	cv2.putText(frame, text1, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
 
	# convert colour back to RGB channel
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# display output image
	st.write("""## Output Image:""")
	st.image(frame)