import os
import cv2
import numpy as np
import time
import mediapipe as mp
import pickle
from datetime import datetime


class FaceRecognitionSystem:
    def __init__(self):
        # Pfade und Verzeichnisse
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.landmarks_dir = os.path.join(base_dir, "landmarks_data")

        # Verzeichnis erstellen, falls es nicht existiert
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)

        # MediaPipe Face Mesh initialisieren
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # Kamera initialisieren
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Kamera konnte nicht geöffnet werden")

        # Kameraauflösung setzen
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Bekannte Gesichtslandmarks und Namen
        self.known_face_names = []
        self.known_face_landmarks_collection = []  # Liste von Listen für mehrere Landmarks pro Person
        self.load_known_landmarks()

        # Schwellenwerte für Gesichtserkennung
        self.recognition_threshold = 0.15  # Für sichere Erkennung
        self.learning_threshold = 0.25     # Höherer Schwellenwert für kontinuierliches Lernen
        self.max_samples_per_person = 10   # Maximale Anzahl der Samples pro Person

        # Face-Click Handling
        self.button_area = []
        self.current_frame = None

        # UI-Status
        self.state = "normal"  # "normal" oder "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        print("Initialisierung abgeschlossen. Drücke 'q' zum Beenden.")

    def extract_face_landmarks(self, image):
        """Extraktion der Gesichtslandmarks mit MediaPipe"""
        # Bild in RGB konvertieren (MediaPipe benötigt RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe-Verarbeitung
        results = self.face_mesh.process(image_rgb)

        face_landmarks_list = []
        face_locations_list = []

        if results.multi_face_landmarks:
            h, w, _ = image.shape

            for face_landmarks in results.multi_face_landmarks:
                # Face location berechnen (Bounding Box)
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Landmarks in einen Vektor konvertieren
                # Wir nutzen eine reduzierte Anzahl von Landmarks für bessere Vergleichbarkeit
                # Diese Landmarks sind wichtige Gesichtsmerkmale
                key_landmarks_indices = [
                    # Augen
                    33, 133, 160, 158, 153, 144,  # Rechtes Auge
                    362, 263, 385, 380, 387, 373,  # Linkes Auge
                    # Nase
                    1, 2, 3, 4, 5, 6, 19, 94, 195,
                    # Mund
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                    # Kinn und Wangen
                    152, 377,
                    # Augenbrauen
                    70, 63, 105, 66, 107,
                    336, 296, 334, 293, 300
                ]

                landmarks_array = []
                for idx in key_landmarks_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        # Wir speichern x, y (normalisierte Koordinaten)
                        landmarks_array.extend([landmark.x, landmark.y])

                face_landmarks_list.append(np.array(landmarks_array))
                face_locations_list.append((y_min, x_max, y_max, x_min))  # top, right, bottom, left

        return face_landmarks_list, face_locations_list

    def load_known_landmarks(self):
        """Gespeicherte Landmarks laden"""
        self.known_face_landmarks_collection = []
        self.known_face_names = []

        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        if os.path.exists(landmarks_file):
            try:
                with open(landmarks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_landmarks_collection = data.get('landmarks_collection', [])
                    self.known_face_names = data.get('names', [])
                print(f"{len(self.known_face_names)} bekannte Personen mit insgesamt {sum(len(landmarks) for landmarks in self.known_face_landmarks_collection)} Landmark-Sets geladen")
            except Exception as e:
                print(f"Fehler beim Laden der Landmarks: {e}")
                # Korrupte Datei überschreiben
                self.save_landmarks_data()
        else:
            print("Keine gespeicherten Landmarks gefunden")

    def save_landmarks_data(self):
        """Bekannte Landmarks speichern"""
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        data = {
            'landmarks_collection': self.known_face_landmarks_collection,
            'names': self.known_face_names
        }
        try:
            with open(landmarks_file, 'wb') as f:
                pickle.dump(data, f)
            total_landmarks = sum(len(landmarks) for landmarks in self.known_face_landmarks_collection)
            print(f"Landmarks gespeichert: {len(self.known_face_names)} Personen mit insgesamt {total_landmarks} Landmark-Sets")
        except Exception as e:
            print(f"Fehler beim Speichern der Landmarks: {e}")

    def save_face(self, name, face_location):
        """Gesichtslandmarks mit dem angegebenen Namen speichern"""
        if not name:
            return False

        # Alle Gesichter im gesamten Frame extrahieren
        all_landmarks, all_locations = self.extract_face_landmarks(self.current_frame)

        # Versuche, das richtige Gesicht basierend auf Position zu finden
        for landmarks, loc in zip(all_landmarks, all_locations):
            if self._locations_are_close(loc, face_location):
                if name in self.known_face_names:
                    # Person existiert bereits, füge neues Landmark-Set hinzu
                    index = self.known_face_names.index(name)
                    self.known_face_landmarks_collection[index].append(landmarks)
                    print(f"Landmarks für bestehenden Namen '{name}' hinzugefügt")
                else:
                    # Neue Person anlegen
                    self.known_face_landmarks_collection.append([landmarks])  # Liste mit einem Landmark-Set
                    self.known_face_names.append(name)
                    print(f"Neue Person '{name}' hinzugefügt")

                self.save_landmarks_data()
                return True

        print(f"WARNUNG: Kein passendes Gesicht für '{name}' gefunden")
        return False

    def _locations_are_close(self, loc1, loc2, tolerance=30):
        """Vergleicht zwei Gesichtslokationen mit Toleranzbereich"""
        return all(abs(a - b) < tolerance for a, b in zip(loc1, loc2))

    def compare_landmarks(self, landmarks):
        """Landmarks mit bekannten Gesichtern vergleichen"""
        if not self.known_face_landmarks_collection or len(self.known_face_landmarks_collection) == 0:
            return "Unbekannt", False, 0

        if landmarks is None or len(landmarks) == 0:
            return "Unbekannt", False, 0

        # Euklidischer Abstand zwischen Landmarks berechnen
        min_distance = float('inf')
        best_match_index = -1
        best_match_landmark_index = -1

        # Für jede bekannte Person
        for i, person_landmarks_list in enumerate(self.known_face_landmarks_collection):
            # Für jedes Landmark-Set dieser Person
            for j, known_landmarks in enumerate(person_landmarks_list):
                if len(known_landmarks) != len(landmarks):
                    print(f"Warnung: Unterschiedliche Landmark-Längen - Bekannt: {len(known_landmarks)}, Aktuell: {len(landmarks)}")
                    continue

                # Euklidischen Abstand berechnen
                distance = np.linalg.norm(landmarks - known_landmarks)

                if distance < min_distance:
                    min_distance = distance
                    best_match_index = i
                    best_match_landmark_index = j

        if best_match_index == -1:
            return "Unbekannt", False, 0

        # Vertraulichkeitsberechnung
        confidence = 1.0 / (1.0 + min_distance)

        # Vergleich mit Schwellenwert
        if min_distance < self.recognition_threshold:
            return self.known_face_names[best_match_index], True, confidence
        elif min_distance < self.learning_threshold:
            # Gesicht mit geringerer Sicherheit erkannt, aber genug für kontinuierliches Lernen
            return self.known_face_names[best_match_index], True, confidence
        else:
            return "Unbekannt", False, confidence

    def add_landmark_to_person(self, name, landmarks):
        """Fügt ein neues Landmark-Set zu einer bereits bekannten Person hinzu"""
        if name in self.known_face_names:
            person_index = self.known_face_names.index(name)
            landmarks_list = self.known_face_landmarks_collection[person_index]
            
            # Überprüfen, ob wir nicht zu viele Samples für diese Person haben
            if len(landmarks_list) < self.max_samples_per_person:
                # Prüfen, ob das Landmark-Set nicht zu ähnlich zu einem bereits vorhandenen ist
                is_too_similar = False
                for existing_landmark in landmarks_list:
                    if len(existing_landmark) == len(landmarks):
                        similarity = np.linalg.norm(landmarks - existing_landmark)
                        if similarity < 0.05:  # Sehr ähnlich zu einem bestehenden Sample
                            is_too_similar = True
                            break
                
                if not is_too_similar:
                    # Landmark hinzufügen und Daten speichern
                    landmarks_list.append(landmarks)
                    print(f"Neues Landmark-Set für '{name}' automatisch hinzugefügt (jetzt {len(landmarks_list)})")
                    self.save_landmarks_data()
                    return True
        
        return False

    def mouse_callback(self, event, x, y, flags, param):
        """Mausklicks auf den 'Learn Face'-Button behandeln"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "normal":
                # Prüfen, ob ein Learn Face-Button angeklickt wurde
                for (button_left, button_top, button_right, button_bottom), face_location in self.button_area:
                    if button_left <= x <= button_right and button_top <= y <= button_bottom:
                        # Button geklickt, Name-Eingabemodus aktivieren
                        self.state = "entering_name"
                        self.selected_face_loc = face_location
                        self.current_text = ""
                        self.text_entry_active = True
                        break
            elif self.state == "entering_name" and not self.text_entry_active:
                # Wenn wir im Namenseingabemodus sind und außerhalb der Textbox klicken
                # Zurück zum normalen Modus
                self.state = "normal"
                self.text_entry_active = False

    def detect_and_recognize_faces(self, frame):
        """Gesichter mit MediaPipe erkennen und bekannte Gesichter wiedererkennen"""
        # Kopie des Frames für die Anzeige erstellen und aktuellen Frame speichern
        display_frame = frame.copy()
        self.current_frame = frame.copy()

        # Gesichtslandmarks mit MediaPipe erkennen
        face_landmarks_list, face_locations = self.extract_face_landmarks(frame)

        # Vorherige Button-Bereiche löschen
        self.button_area = []

        # Erkannte Gesichter durchgehen
        for i, (landmarks, face_loc) in enumerate(zip(face_landmarks_list, face_locations)):
            top, right, bottom, left = face_loc

            # Gesicht erkennen
            name, is_known_face, confidence = self.compare_landmarks(landmarks)

            # Kontinuierliches Lernen: Wenn Gesicht erkannt wurde und Konfidenz nicht perfekt ist
            if is_known_face and 0.5 < confidence < 0.95:
                # Automatisch neues Landmark-Set hinzufügen für dieses Gesicht
                self.add_landmark_to_person(name, landmarks)

            # Rahmen um das Gesicht zeichnen
            color = (0, 255, 0) if is_known_face else (0, 0, 255)  # Grün für erkannt, Rot für unbekannt
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

            # Label mit Namen unter dem Gesicht zeichnen
            label_top = bottom + 10
            label_bottom = bottom + 35
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            conf_text = f"{confidence:.2f}" if confidence > 0 else "N/A"
            cv2.putText(display_frame, f"{name} ({conf_text})", (left + 6, label_top + 20), font, 0.5,
                        (255, 255, 255), 1)

            # "Learn Face"-Button nur für unbekannte Gesichter hinzufügen
            if self.state == "normal" and not is_known_face:
                button_left = left
                button_top = top - 30
                button_right = right
                button_bottom = top

                if button_top > 0:  # Sicherstellen, dass Button innerhalb des Frames ist
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0),
                                  cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5,
                                (255, 255, 255), 1)

                    # Button-Bereich und zugehörigen Gesichtsbereich speichern
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), face_loc))

        return display_frame, face_locations

    def draw_text_input(self, frame):
        """Texteingabeoberfläche auf dem Frame zeichnen"""
        height, width = frame.shape[:2]

        # Texteingabebereich am unteren Rand des Frames erstellen
        input_height = 40
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)

        # Aktuellen Text und Anweisung anzeigen
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Name eingeben: {self.current_text}_"
        cv2.putText(frame, text, (10, height - 15), font, 0.7, (255, 255, 255), 1)

        # Anweisungen anzeigen
        instructions = "ENTER zum Speichern, ESC zum Abbrechen"
        cv2.putText(frame, instructions, (width - 300, height - 15), font, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """Hauptschleife für die Gesichtserkennung"""
        # Maus-Callback einmal einrichten, außerhalb der Schleife
        cv2.namedWindow('Gesichtserkennung mit MediaPipe')
        cv2.setMouseCallback('Gesichtserkennung mit MediaPipe', self.mouse_callback)

        while True:
            # Frame aufnehmen
            ret, frame = self.cap.read()

            if not ret:
                print("Fehler beim Aufnehmen des Frames")
                break

            # Frame spiegeln (Selfie-Ansicht)
            frame = cv2.flip(frame, 1)

            if self.state == "normal":
                # Normaler Betrieb: Gesichter erkennen und identifizieren
                display_frame, face_locations = self.detect_and_recognize_faces(frame)
                # Ergebnisframe anzeigen
                cv2.imshow('Gesichtserkennung mit MediaPipe', display_frame)

                # Tastatureingaben verarbeiten
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            elif self.state == "entering_name":
                # Texteingabemodus
                # Einfacheren Frame mit hervorgehobenem Gesicht anzeigen
                display_frame = frame.copy()

                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc

                    # Ausgewähltes Gesicht hervorheben
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)

                    # Anzeigen, was gespeichert wird
                    cv2.putText(display_frame, "Zu speicherndes Gesicht",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Texteingabeoberfläche zeichnen
                display_frame = self.draw_text_input(display_frame)

                # Frame anzeigen
                cv2.imshow('Gesichtserkennung mit MediaPipe', display_frame)

                # Tasteneingaben für Texteingabe verarbeiten
                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # ENTER-Taste - Gesicht mit Namen speichern
                    if self.current_text:
                        success = self.save_face(self.current_text, self.selected_face_loc)
                        if not success:
                            print("Fehler beim Speichern des Gesichts")
                    # Zurück zum normalen Modus
                    self.state = "normal"
                elif key == 27:  # ESC-Taste - abbrechen
                    self.state = "normal"
                elif key == 8:  # BACKSPACE - letztes Zeichen löschen
                    self.current_text = self.current_text[:-1]
                elif key == ord('q'):  # q-Taste - beenden
                    break
                elif 32 <= key <= 126:  # Druckbare ASCII-Zeichen
                    self.current_text += chr(key)

        # Kamera freigeben und Fenster schließen
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()
