import cv2
import face_recognition

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

img = cv2.imread('face.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(50, 50)
)

for (x, y, w, h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

output_path = '/workspaces/Face-Detection-Using-Haar-Cascades/detected_face.jpg'
cv2.imwrite(output_path, img)

print(f"Face detection complete! Found {len(faces)} face(s). Image saved as 'detected_face.jpg'.")
