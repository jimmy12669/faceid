import cv2
import os

o_dir = "captured_images"
if not os.path.exists(o_dir):
    os.makedirs(o_dir)

d = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

r_img = None
r_hist = None
r_img_path = os.path.join(o_dir, "reference_image.jpg")

def capture_r_img(frame):
    global r_img, r_hist
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = d.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        r_img = gray[y:y+h, x:x+w]
        r_hist = cv2.calcHist([r_img], [0], None, [256], [0, 256])
        r_hist = cv2.normalize(r_hist, r_hist).flatten()

        cv2.imwrite(r_img_path, frame)
        print(f"Reference image saved to {r_img_path}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(500)

if os.path.exists(r_img_path):
    r_img_color = cv2.imread(r_img_path)
    capture_r_img(r_img_color)
else:
    print(f"No reference image found at {r_img_path}. Waiting to capture a reference image.")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("An error has occurred")
    exit()

print("Press 'c' to capture a picture, 'v' to verify, and 'q' to quit.")

image_counter = 0   

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    cv2.imshow('Camera', frame)

    if r_img is None:
        capture_r_img(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = d.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Captured Image', img)
        image_path = os.path.join(o_dir, f"captured_image_{image_counter}.jpg")
        cv2.imwrite(image_path, img)
        print(f"Image saved to {image_path}")

        capture_r_img(frame)
        image_counter += 1

    elif key == ord('v'):
        if r_img is None:
            print("No reference image available. Please wait for the system to capture a reference image.")
        else:
            img = frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = d.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                current_face = gray[y:y+h, x:x+w]
                current_face_hist = cv2.calcHist([current_face], [0], None, [256], [0, 256])
                current_face_hist = cv2.normalize(current_face_hist, current_face_hist).flatten()

                distance = cv2.compareHist(r_hist, current_face_hist, cv2.HISTCMP_CORREL)
                print(f"Similarity score: {distance:.2f}")
                similarity_text = f"Similarity: {distance:.2f}"
                if distance > 0.7:
                    match_text = "Match found!"
                else:
                    match_text = "No match found."

                cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, match_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Camera', frame)
                cv2.waitKey(0)

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()