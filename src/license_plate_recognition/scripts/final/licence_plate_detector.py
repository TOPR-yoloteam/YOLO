import os
import cv2
from ultralytics import YOLO


# Funktion zum Abrufen der vollständigen Pfade aller Bilder in einem Ordner
def get_images(image_folder="data/images", file_extension=".png"):
    """
    Sammelt alle Bilddateien mit der angegebenen Erweiterung aus einem Ordner.

    Args:
        image_folder (str): Pfad zum Ordner mit den Bildern.
        file_extension (str): Dateierweiterung, nach der gefiltert wird (z. B. ".png").

    Returns:
        list: Liste der absoluten Pfade aller gültigen Bilddateien.
    """
    return [
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.endswith(file_extension)
    ]


# Funktion zum Speichern eines Bildes in einem gewünschten Ordner
def save_image(image_name, image, subfolder="data/detected_plates"):
    """
    Speichert ein Bild in einem angegebenen Ordner.

    Args:
        image_name (str): Name der Bilddatei, die gespeichert werden soll.
        image (np.ndarray): Das Bild (als Numpy-Array), das gespeichert wird.
        subfolder (str): Zielpfad zum Speichern des Bildes.

    Returns:
        None
    """
    # Erstelle den Zielordner, falls er nicht existiert
    os.makedirs(subfolder, exist_ok=True)
    output_path = os.path.join(subfolder, image_name)

    # Speichere das Bild
    cv2.imwrite(output_path, image)


# Hauptfunktion zur Verarbeitung der Bilder und Erkennung der Nummernschilder
def process_images(model, images):
    """
    Verarbeitet eine Liste von Bildern: Nummernschilder werden erkannt, extrahiert und gespeichert.

    Args:
        model (YOLO): Das vortrainierte YOLO-Modell für die Objekterkennung.
        images (list): Liste von Dateipfaden der Bilder, die bearbeitet werden sollen.

    Returns:
        None
    """
    for file_path in images:
        # Lade das Bild mit OpenCV
        image = cv2.imread(file_path)
        if image is None:
            print(f"Fehler: Bild konnte nicht geladen werden: {file_path}. Überspringe...")
            continue  # Überspringe dieses Bild, falls es nicht geladen werden konnte

        # Lass das Modell die Erkennung durchführen
        results = model(file_path)

        plate_counter = 0  # Zähler für mehrere Nummernschilder in einem Bild

        for result in results:
            for box in result.boxes:
                # Extrahiere die Bounding-Box-Koordinaten und prüfe das Konfidenzniveau
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinaten in Ganzzahlen umwandeln
                confidence = box.conf[0].item()  # Konfidenzwert des Modells
                class_id = int(box.cls[0].item())  # Klassen-ID

                if confidence > 0.4 and class_id == 0:  # Überprüfe, ob es sich um ein Nummernschild handelt
                    # Extrahiere das Nummernschild aus dem Bild
                    license_plate = image[y1:y2, x1:x2]

                    # Generiere einen eindeutigen Dateinamen für das Nummernschild
                    plate_filename = f"plate_{os.path.splitext(os.path.basename(file_path))[0]}_{plate_counter}.png"

                    # Speichere das extrahierte Nummernschild
                    save_image(plate_filename, license_plate, subfolder="data/detected_plates/license_plates")
                    plate_counter += 1  # Zähler inkrementieren

                    # Zeichne die Bounding Box und die Beschriftung auf das ursprüngliche Bild
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Grüne Bounding Box
                    cv2.putText(
                        image,
                        f"license_plate {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

        # Speichere das bearbeitete Bild mit Bounding Boxen
        save_image(os.path.basename(file_path), image)


# Hauptprogramm: Modell laden und Bilder verarbeiten
if __name__ == "__main__":
    # YOLO-Modell laden (angepasst für Nummernschilderkennung)
    model = YOLO("model/license_plate_detector.pt")

    # Bilderverzeichnis sicherstellen und Bilder abrufen
    images = get_images()

    if not images:
        print("Keine Bilder gefunden. Stelle sicher, dass sich Bilder im Verzeichnis 'data/images' befinden.")
    else:
        # Verarbeitung der Bilder
        process_images(model, images)
