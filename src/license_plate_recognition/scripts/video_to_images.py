import cv2
import time

# Dauer der Aufnahme in Sekunden
recording_duration = 10

# Zeitintervall für das Speichern der Frames in Sekunden
frame_interval = 0.5

# Zugriff auf die Kamera (Standard ist 0 für die Standardkamera)
cap = cv2.VideoCapture(0)

# Überprüfen, ob die Kamera geöffnet werden konnte
if not cap.isOpened():
    print("Fehler: Kamera kann nicht geöffnet werden.")
    exit()

print("Aufnahme startet...")

# Startzeit der Aufnahme
start_time = time.time()
frame_counter = 0

# Variablen für FPS-Berechnung
previous_time = time.time()
fps = 0

while True:
    # Aktuellen Frame aus dem Videostream lesen
    ret, frame = cap.read()

    if not ret:
        print("Fehler: Frame konnte nicht gelesen werden.")
        break

    # Eine Kopie des Frames erstellen, die verändert angezeigt wird
    frame_display = frame.copy()

    # Aktuelle Zeit bestimmen
    current_time = time.time()

    # Verbleibende Zeit berechnen
    time_remaining = recording_duration - (current_time - start_time)
    if time_remaining > 0:
        timer_text = f"Zeit verbleibend: {int(time_remaining)}s"
    else:
        timer_text = "Zeit abgelaufen!"

    # Timer auf dem Anzeigeframe (oben links) zeichnen
    cv2.putText(
        frame_display,
        timer_text,
        (10, 30),  # Position (x, y) oben links
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),  # Grün
        2,
        cv2.LINE_AA
    )

    # FPS-Berechnung
    fps = 1 / (current_time - previous_time)  # FPS = 1 / Frame-Zeit
    previous_time = current_time

    # FPS-Text (oben rechts)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        frame_display,
        fps_text,
        (frame.shape[1] - 200, 30),  # Position (x, y) oben rechts
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),  # Blau
        2,
        cv2.LINE_AA
    )

    # Frame mit Timer und FPS anzeigen
    cv2.imshow("Videostream", frame_display)

    # Überprüfen, ob es Zeit ist, einen neuen Frame zu speichern
    if current_time - start_time >= frame_counter * frame_interval:
        # Original-Frame ohne Text speichern
        filename = f"frame_{frame_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"{filename} gespeichert.")
        frame_counter += 1

    # Überprüfen, ob die Aufnahmezeit überschritten wurde
    if current_time - start_time > recording_duration:
        break

    # Beenden der Aufnahme bei Drücken der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Aufnahme durch Benutzer beendet.")
        break

# Kamera freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()

print("Aufnahme beendet.")
