import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Valentin.Talmon\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

print(pytesseract.get_tesseract_version())