from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time

# Carica il modello e il processore
model_caricato = VisionEncoderDecoderModel.from_pretrained('captcha_alfanumerici/modello_preaddestrato')
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Inizializza Selenium
driver = webdriver.Chrome()  # Assicurati di avere il driver Chrome configurato correttamente
driver.get("http://localhost:8000/captcha_alfanumerici/demo.html")  # Sostituisci con il percorso corretto

# Trova l'elemento dell'immagine CAPTCHA sulla pagina
captcha_image = driver.find_element(By.TAG_NAME, "img")

# Ottieni l'immagine come byte array
captcha_image_bytes = captcha_image.screenshot_as_png
image = Image.open(BytesIO(captcha_image_bytes)).convert("RGB")

# Prepara l'immagine per il modello
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Genera la predizione
generated_ids = model_caricato.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# Inserisci il testo previsto nella textbox del CAPTCHA
textbox = driver.find_element(By.ID, "captcha_input") 
textbox.send_keys(generated_text)

# Trova il pulsante di invio e cliccalo
verify_button = driver.find_element(By.ID, "submit_button")  
verify_button.click()

# Attendi qualche secondo per vedere il risultato 
time.sleep(10)

# Chiudi il browser
driver.quit()