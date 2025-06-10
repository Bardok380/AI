import face_recognition         # Load CMake onto you computer
import cv2
import os

# Load known faces
known_face_encodings = []
known_face_names = []

# Load images from the known_faces folder
for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join("known_faces", filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)

# Load an image with unknown faces
unknown_image = face_recognition.load_image_file("unknown_faces/unknown.jpg")
unknown_face_locations = face_recognition.face_locations(unknown_image) 
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

# Convert to BGR for OpenCV
image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_face_names[match_index]

    # Draw a rectangle around the face
    cv2.rectangle(image_bgr, (left,top), (right, bottom), (0, 255, 0), 2)
    # Draw a label
    cv2.putText(image_bgr, name, (left, top - 10), cv2.Font_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display result
cv2.imshow("Facial Recognition", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()