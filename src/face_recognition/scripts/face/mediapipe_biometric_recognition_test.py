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
        self.known_face_landmarks = []
        self.known_face_names = []
        self.load_known_landmarks()

        # Face-Click Handling
        self.button_area = []
        self.current_frame = None

        # UI-Status
        self.state = "normal"  # "normal" oder "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        # Schwellenwert für Gesichtserkennung
        self.recognition_threshold = 0.15  # Angepasster Schwellenwert

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
        self.known_face_landmarks = []
        self.known_face_names = []

        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        if os.path.exists(landmarks_file):
            try:
                with open(landmarks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_landmarks = data.get('landmarks', [])
                    self.known_face_names = data.get('names', [])
                print(f"{len(self.known_face_names)} bekannte Gesichter geladen")
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
            'landmarks': self.known_face_landmarks,
            'names': self.known_face_names
        }
        try:
            with open(landmarks_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Landmarks gespeichert: {len(self.known_face_names)} Gesichter")
        except Exception as e:
            print(f"Fehler beim Speichern der Landmarks: {e}")

    def save_face(self, name, face_location):
        """Gesichtslandmarks mit dem angegebenen Namen speichern (verbessert)"""
        if not name:
            return False

        # Alle Gesichter im gesamten Frame extrahieren
        all_landmarks, all_locations = self.extract_face_landmarks(self.current_frame)

        # Versuche, das richtige Gesicht basierend auf Position zu finden
        for landmarks, loc in zip(all_landmarks, all_locations):
            if self._locations_are_close(loc, face_location):
                if name in self.known_face_names:
                    index = self.known_face_names.index(name)
                    self.known_face_landmarks[index] = landmarks
                    print(f"Landmarks für bestehenden Namen '{name}' aktualisiert")
                else:
                    self.known_face_landmarks.append(landmarks)
                    self.known_face_names.append(name)
                    print(f"Neue Landmarks für '{name}' hinzugefügt")

                self.save_landmarks_data()
                return True

        print(f"WARNUNG: Kein passendes Gesicht für '{name}' gefunden")
        return False

    def _locations_are_close(self, loc1, loc2, tolerance=30):
        """Vergleicht zwei Gesichtslokationen mit Toleranzbereich"""
        return all(abs(a - b) < tolerance for a, b in zip(loc1, loc2))

    def compare_landmarks(self, landmarks):
        """Landmarks mit bekannten Gesichtern vergleichen"""
        if not self.known_face_landmarks or len(self.known_face_landmarks) == 0:
            return "Unbekannt", False, 0

        if landmarks is None or len(landmarks) == 0:
            return "Unbekannt", False, 0

        # Euklidischer Abstand zwischen Landmarks berechnen
        min_distance = float('inf')
        best_match_index = -1

        # Längenüberprüfung für konsistente Vergleiche
        for i, known_landmarks in enumerate(self.known_face_landmarks):
            if len(known_landmarks) != len(landmarks):
                print(
                    f"Warnung: Unterschiedliche Landmark-Längen - Bekannt: {len(known_landmarks)}, Aktuell: {len(landmarks)}")
                continue

            # Euklidischen Abstand berechnen
            distance = np.linalg.norm(landmarks - known_landmarks)

            if distance < min_distance:
                min_distance = distance
                best_match_index = i

        if best_match_index == -1:
            return "Unbekannt", False, 0

        # Vertraulichkeitsberechnung
        confidence = 1.0 / (1.0 + min_distance)

        # Vergleich mit Schwellenwert
        if min_distance < self.recognition_threshold:
            return self.known_face_names[best_match_index], True, confidence
        else:
            return "Unbekannt", False, confidence

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
