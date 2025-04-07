import cv2
import time
from queue import Queue
from threading import Thread
import easyocr
import numpy as np
import os


class RaspberryPiOCRPipeline:
    def __init__(self, resolution=(320, 240), frame_interval=0.5, recording_duration=5, save_images=False):
        self.resolution = resolution
        self.frame_interval = frame_interval
        self.recording_duration = recording_duration
        self.save_images = save_images

        # Warteschlangen für die Verarbeitung
        self.frame_queue = Queue(maxsize=5)  # Begrenzte Größe (RAM-Effizienz)
        self.reader = easyocr.Reader(['en'], gpu=False)  # GPU deaktiviert für Raspberry Pi

        # Ordner für gespeicherte Bilder
        self.output_dir = "processed_images"
        if self.save_images:
            self.ensure_directory(self.output_dir)

    @staticmethod
    def ensure_directory(directory):
        """
        Erstellt das Verzeichnis, falls es nicht existiert.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def capture_frames(self):
        """
        Frames von der Kamera aufnehmen und in die Warteschlange einfügen.
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Fehler: Kamera konnte nicht geöffnet werden.")
            return

        print(f"Starte Aufnahme... (Auflösung: {self.resolution[0]}x{self.resolution[1]})")
        start_time = time.time()
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fehler: Frame konnte nicht gelesen werden.")
                break

            current_time = time.time()
            if current_time - start_time >= self.recording_duration:
                break

            # Skalieren auf gewünschte Auflösung
            frame = cv2.resize(frame, self.resolution)

            # Frames nur in bestimmten Intervallen speichern
            if current_time - start_time >= frame_counter * self.frame_interval:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    frame_counter += 1

        cap.release()
        self.frame_queue.put(None)  # Beendigungs-Signal
        print("Kameraaufnahme abgeschlossen.")

    def preprocess_and_ocr(self, frame, frame_index):
        """
        Bereitet Bilder vor und führt OCR aus.
        """
        # 1. Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Schwellenwert anwenden
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Bild speichern (optional)
        if self.save_images:
            processed_path = os.path.join(self.output_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(processed_path, thresh)
            print(f"Bild gespeichert: {processed_path}")

        # OCR ausführen
        results = self.reader.readtext(thresh)

        # Ergebnisse formatieren
        text_results = [{"text": text, "confidence": conf} for (_, text, conf) in results]
        return text_results

    def process_frames(self):
        """
        Frames aus der Warteschlange holen, verarbeiten und OCR ausführen.
        """
        frame_index = 0
        while True:
            frame = self.frame_queue.get()
            if frame is None:  # Beendigungs-Signal
                break

            # Bildverarbeitung & OCR
            ocr_results = self.preprocess_and_ocr(frame, frame_index)
            frame_index += 1

            # OCR-Ergebnisse ausgeben
            print("OCR Ergebnis:")
            for entry in ocr_results:
                print(f"- Text: {entry['text']}, Vertrauen: {entry['confidence']:.2f}")

    def run_pipeline(self):
        """
        Startet die gesamte OCR-Pipeline.
        """
        capture_thread = Thread(target=self.capture_frames)
        process_thread = Thread(target=self.process_frames)

        # Threads starten
        capture_thread.start()
        process_thread.start()

        # Warten, bis beide Threads beendet sind
        capture_thread.join()
        process_thread.join()

        print("Pipeline abgeschlossen.")


if __name__ == "__main__":
    pipeline = RaspberryPiOCRPipeline(
        resolution=(320, 240),
        frame_interval=1,
        recording_duration=10,
        save_images=True  # Gespeicherte Bilder werden erzeugt
    )
    pipeline.run_pipeline()
