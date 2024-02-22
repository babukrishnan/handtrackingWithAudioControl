import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode #Sets the mode for hand tracking.
        self.maxHands = maxHands #Specifies the maximum number of hands to detect and track.
        self.detectionCon = detectionCon #Sets the confidence threshold for hand detection.
        self.trackCon = trackCon # Sets the confidence threshold for hand tracking.
        self.mpHands = mp.solutions.hands #Imports the hand tracking module from the MediaPipe library.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands) #Initializes the hand tracking object with the specified mode and maximum number of hands.
        self.mpDraw = mp.solutions.drawing_utils #Imports the drawing utilities module from MediaPipe.
        self.results = None  # Initializes the results variable. This is likely to store the results of hand tracking later.
        self.tipIds = [4, 8, 12, 16, 20] #Specifies the landmark indices corresponding to the fingertips.

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)     #Uses the previously initialized hand tracking object (self.hands) to process the RGB 
                                                      #image and obtain hand landmarks.The results are stored in the self.results attribute.
        
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) <= 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()