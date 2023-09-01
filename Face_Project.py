import cv2
import face_recognition

# Load known faces and their names
known_faces = []
known_names = []

# Example: Load known faces from images and associate them with names
image_path1 = "known_faces/person1.jpg"
image_path2 = "known_faces/person2.jpg"

person1_image = face_recognition.load_image_file(image_path1)
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]
known_faces.append(person1_face_encoding)
known_names.append("Person 1")

person2_image = face_recognition.load_image_file(image_path2)
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]
known_faces.append(person2_face_encoding)
known_names.append("Person 2")

# Initialize variables for marking attendance
attendance = {}

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            
            # Mark attendance
            attendance[name] = attendance.get(name, 0) + 1
        
        # Draw rectangle and label on the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

# Print attendance
print("Attendance:")
for name, count in attendance.items():
    print(f"{name}: {count} times")
