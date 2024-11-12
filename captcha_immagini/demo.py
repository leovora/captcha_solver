import time
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from tensorflow.keras.models import load_model
import requests
import cv2

# Caricare il modello di deep learning
model = load_model("captcha_immagini/modello_immagini.h5")
model.compile(metrics=['accuracy'])

# Funzione per preprocessare l'immagine per il modello
def preprocess_image(img):
    # Converti l'immagine in un array NumPy (OpenCV la carica in BGR)
    img = np.array(img)

    # Converti l'immagine da BGR a RGB (poiché OpenCV carica in BGR per default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ridimensiona l'immagine a 120x120 (conservando i 3 canali)
    img = cv2.resize(img, (120, 120))

    # Aggiungere una dimensione batch
    img = np.expand_dims(img, axis=0)
    
    print(img)
    return img

# Funzione per caricare l'immagine da una URL (se non è in base64)
def load_image_from_url(url):
    response = requests.get(url)
    # Carica l'immagine direttamente con OpenCV
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return img

# Configurare Selenium WebDriver
driver = webdriver.Chrome()
url = "http://localhost:8000/captcha_immagini/demo.html"  # URL della tua pagina con CAPTCHA
driver.get(url)
time.sleep(3)  # Attendere il caricamento della pagina

# Acquisire l'immagine del CAPTCHA (assumendo che l'immagine sia in un tag <img>)
captcha_element = driver.find_element(By.TAG_NAME, "img")
captcha_src = captcha_element.get_attribute("src")

# Decodificare l'immagine se è codificata in base64
if "base64," in captcha_src:
    img_data = captcha_src.split("base64,")[1]
    image = Image.open(io.BytesIO(base64.b64decode(img_data)))  # Questa riga è stata modificata
else:
    # Se l'immagine non è codificata in base64, scaricarla dal percorso
    image = load_image_from_url(captcha_src)

# Preprocessare l'immagine
input_image = preprocess_image(image)

# Fare una previsione con il modello
prediction = model.predict(input_image)
predicted_label = np.argmax(prediction)  # Decodifica l'indice della previsione

# Decodifica la previsione nel formato delle etichette
class_labels = ["car", "bus", "crosswalk", "palm", "hydrant", "bicycle", "traffic light", "motorcycle", "bridge", "chimney", "stair"]
predicted_label_text = class_labels[predicted_label]

# Inserire la previsione nel campo di testo del CAPTCHA
textbox = driver.find_element(By.ID, "captcha_input")  
textbox.send_keys(predicted_label_text)

# Cliccare sul pulsante per inviare il modulo
submit_button = driver.find_element(By.ID, "submit_button")
submit_button.click()

# Attendere per verificare il risultato (facoltativo)
time.sleep(6000)
