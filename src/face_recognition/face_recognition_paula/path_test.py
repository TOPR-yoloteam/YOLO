import os

known_faces_dir = "YOLO\\src\\face_recognition\\data\\img\\"

if os.path.exists(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        print(filename)
else:
    print(f"Verzeichnis nicht gefunden: {known_faces_dir}")
