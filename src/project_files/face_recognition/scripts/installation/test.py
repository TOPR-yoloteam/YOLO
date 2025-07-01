import cv2
import numpy as np

# Erstelle ein leeres Bild (schwarzer Hintergrund)
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Zeige das Bild in einem Fenster an
cv2.imshow("Test Window", image)

# Warte, bis eine Taste gedrückt wird
cv2.waitKey(0)

# Schließe alle Fenster
cv2.destroyAllWindows()