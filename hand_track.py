import cv2
import mediapipe as mp
from time import time
import pyautogui as pag

# READ VIDEO
vid_capt = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Baca video dari kamera webcam

# PROPERTIES VIDEO
resizer_factor = 0.75
frame_width, frame_height = (int(resizer_factor*vid_capt.get(3)), int(resizer_factor*vid_capt.get(4)))
xCenter, yCenter = (frame_width//2, frame_height//2)

# HAND TRACKER
hands = mp.solutions.hands.Hands(
  static_image_mode = False,
  max_num_hands = 1,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)

# FPS TRACKER
pTime = 0

rectColor = (0, 255, 0)
xMouse = 0
yMouse = 0
xP = 0
yP = 0

# DISPLAY VIDEO
while vid_capt.isOpened():
  # READ FRAME
  isFrameRead, frame = vid_capt.read()

  if isFrameRead:
      # MEASURE FPS
      fps = 1 // (time()-pTime)
      pTime=time()

      # FLIP, RECTANGLE, RESIZE
      flipped_frame = cv2.flip(src=frame, flipCode=1)
      frame = cv2.resize(src=flipped_frame, dsize=None, fx=resizer_factor, fy=resizer_factor, interpolation=cv2.INTER_AREA)
      cv2.rectangle(img=frame, pt1=(xCenter-100, yCenter-100), pt2=(xCenter+100, yCenter+100), color=rectColor, thickness=2)

      cv2.putText(
      frame,
      text=('FPS: '+str(fps)),
      org=(30,50),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.5,
      color=(0,255,255),
      thickness=1
    )

      # HAND PROCESSING
      processedHands = hands.process(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))
      if processedHands.multi_hand_landmarks:
        for handLms in processedHands.multi_hand_landmarks:
          for id, lm in enumerate(handLms.landmark):
            x, y = (int(lm.x*frame_width), int(lm.y*frame_height))
            print(lm.z)
            z = int(abs(lm.z) / 0.3 * 100) if lm.z > -0.3 else 100
            mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            if id == 8 and lm.x > 0 and lm.y > 0 and lm.x < 1 and lm.y < 1: # ID=8 berarti ujung telunjuk
              xMouse, yMouse = (x,y)
              isClicked = True if z >= 40 else False
              clickColor = (0,255,255) if isClicked else (255,255,0)
              cv2.putText(
                frame,
                text=('('+str(x)+','+str(y)+')'),
                org=(x-40, y-20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=clickColor,
                thickness=1
                )
              cv2.putText(
                frame,
                text=('KEDALAMAN: '+str(z)),
                org=(30,30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=clickColor,
                thickness=1
              )
              cv2.circle(frame, center=(x, y), radius=10, color=clickColor, thickness=-1, lineType=cv2.FILLED)

      # DISPLAY FRAME        
      cv2.imshow('Hand Track', frame)

      # MOVE MOUSE
      if xMouse > xCenter-100 and yMouse > yCenter-100 and xMouse < xCenter+100 and yMouse < yCenter+100:
        if isInTouchPad:
          xRel = xMouse-xP
          yRel = yMouse-yP
          pag.moveRel(2*xRel, 2*yRel)
        else:
          xRel = 0
          yRel = 0
        xP = xMouse
        yP = yMouse
        rectColor = (0, 255, 0)
        isInTouchPad = True
      else:
        isInTouchPad = False
        xMouse = 0
        yMouse = 0
        xP = 0
        yP = 0
        rectColor = (0, 127, 0)

      # EXIT LOOP
      isExit = cv2.waitKey(20)
      if isExit == ord('q'):
        break

  else:
    break

vid_capt.release()
cv2.destroyAllWindows()