from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import skimage.filters as filters

# Carica il modello
model_caricato = load_model('captcha_logici/modello_logici.h5')

# Inizializza Selenium
driver = webdriver.Chrome()  # Assicurati di avere il driver Chrome configurato correttamente
driver.get("http://localhost:8000/captcha_logici/demo.html")  # Sostituisci con il percorso corretto

# Trova l'elemento dell'immagine CAPTCHA sulla pagina
captcha_image = driver.find_element(By.TAG_NAME, "img")

# Ottieni l'immagine come byte array
captcha_image_bytes = captcha_image.screenshot_as_png
image = Image.open(BytesIO(captcha_image_bytes)).convert("RGB")

# Converti l'immagine PIL in un array NumPy per l'elaborazione con OpenCV
img = np.array(image)

# Definisci le etichette per i numeri e gli operatori
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', 'x', ':']
INPUT_SHAPE = (50, 200, 1)

def prepare_image(img, input_shape):
    # Ridimensiona l'immagine alle dimensioni specificate in INPUT_SHAPE
    img = cv2.resize(img, (input_shape[1], input_shape[0]))  # Usa INPUT_SHAPE per il ridimensionamento

    # Se l'immagine è a colori, convertila in scala di grigi
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inverti e riduci il rumore
    img = 255 - cv2.fastNlMeansDenoising(img)

    # Applica i filtri di levigatura
    smooth = cv2.medianBlur(img, 5)
    smooth = cv2.GaussianBlur(smooth, (5, 5), 0)

    # Aumenta il contrasto
    division = cv2.divide(img, smooth, scale=255)

    # Applica l'unsharp masking per aumentare la nitidezza
    sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5, channel_axis=False, preserve_range=False)
    sharp = (255 * sharp).clip(0, 255).astype(np.uint8)

    # Normalizza e ridimensiona per l'input del modello
    return np.array(sharp / 255.0, dtype=np.float32).reshape(input_shape)

def calculate_expression(digit1, digit2, operator):
    # Mappa l'operatore riconosciuto a una funzione matematica
    if operator == '+':
        return digit1 + digit2
    elif operator == '-':
        return digit1 - digit2
    elif operator == 'x' or operator == 'X':
        return digit1 * digit2
    elif operator == ':':
        return digit1 / digit2 if digit2 != 0 else 'Errore: divisione per zero'
    else:
        return 'Operatore sconosciuto'

def predict_image(model, img, input_shape):
    preprocessed_img = prepare_image(img, input_shape)

    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    predictions = model.predict(preprocessed_img)

    pred_digit1 = np.argmax(predictions[0])
    pred_digit2 = np.argmax(predictions[1])
    pred_operator = np.argmax(predictions[2])  # Previsione per l'operatore

    # Stampa le previsioni
    print(f"Previsione per Digit1: {LABELS[pred_digit1]}")
    print(f"Previsione per Digit2: {LABELS[pred_digit2]}")
    print(f"Previsione per Operatore: {OPERATORS[pred_operator]}")  # Supponendo che LABELS_OPERATOR sia definito

    # Calcola il risultato dell'espressione
    result = calculate_expression(int(LABELS[pred_digit1]), int(LABELS[pred_digit2]), OPERATORS[pred_operator])
    print(f"Risultato dell'espressione: {result}")
    
    # Inserisci il testo previsto nella textbox del CAPTCHA
    textbox = driver.find_element(By.ID, "captcha_input") 
    textbox.send_keys(result)

predict_image(model_caricato, img, INPUT_SHAPE)

# Trova il pulsante di invio e cliccalo
verify_button = driver.find_element(By.ID, "submit_button")  
verify_button.click()

# Attendi qualche secondo per vedere il risultato 
time.sleep(10)

# Chiudi il browser
driver.quit()