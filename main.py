# Importing all the Libraries required
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np


# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,
	model_complexity=1,
	min_detection_confidence=0.75,
	min_tracking_confidence=0.75,
	max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
	# Read video frame by frame
	_,frame = cap.read()
	
	#Flip image since camera changes it by itself
	frame=cv2.flip(frame,1)
	
	# Changin the BGR --> RGB
	frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	
	# Process the RGB image
	Process = hands.process(frameRGB)
	
	landmarkList = []
	# Seeing if the hand is present in the frame
	if Process.multi_hand_landmarks:
		# detect handmarks
		for handlm in Process.multi_hand_landmarks:
			for _id,landmarks in enumerate(handlm.landmark):
				# store height and width of image
				height,width,color_channels = frame.shape
				
				# sotre x and y coords
				x,y = int(landmarks.x*width),int(landmarks.y*height)			
				landmarkList.append([_id,x,y])
			
			# Drawing theLandmarks
			Draw.draw_landmarks(frame,handlm,mpHands.HAND_CONNECTIONS)
	
	# If landmarks list is not empty
	if landmarkList != []:

		x_1,y_1 = landmarkList[4][1],landmarkList[4][2]
		
		x_2,y_2 = landmarkList[8][1],landmarkList[8][2]
		
		# draw circle on thumb and index finger tip
		cv2.circle(frame,(x_1,y_1),7,(0,255,0),cv2.FILLED)
		cv2.circle(frame,(x_2,y_2),7,(0,255,0),cv2.FILLED)
		
		# draw line from tip of thumb to tip of index finger
		cv2.line(frame,(x_1,y_1),(x_2,y_2),(0,255,0),3)
		
		# calculate square root of the sum
		L = hypot(x_2-x_1,y_2-y_1)
		
		b_level = np.interp(L,[15,220],[0,100])
	
		# set brightness
		sbc.set_brightness(int(b_level))

	# Display Video and when 'q' is entered, close the window
	cv2.imshow('Image', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

print('hello')