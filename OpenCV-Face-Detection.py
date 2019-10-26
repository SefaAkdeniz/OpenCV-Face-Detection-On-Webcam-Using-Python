"""
ÖNEMLİ NOT: Çalıştırdıktan sonra programı kapatmak için "Q" tuşuna basın.
"""

import cv2

# Cascade yükleme
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

# Tanıma yapacak fonksiyon
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
        roi_gray = gray[y:y+height, x:x+width]
        roi_color = frame[y:y+height, x:x+width]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ewidth, eheight) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ewidth, ey+eheight), (0, 255, 0), 2)
    return frame


# Webcam ile tanıma yapılıyor.
# Eğer bilgisayarınızda birden fazla kamera bağlıysa 0'ı 1 yapabilirsiniz.
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    canvas = detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()