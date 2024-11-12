import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
from pathlib import Path
from tensorflow.keras import layers

# Definizione del layer CTC
class LayerCTC(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
    

def preprocess_image(image):
    image = image.convert("L")  # Converti in scala di grigi
    image = image.resize((200, 50))  # Usa le dimensioni appropriate per il modello
    # Converti l'immagine in un array numpy
    image_array = np.array(image)
    # Normalizza i valori dell'immagine per allinearlo al primo codice
    image_array = image_array / 255.0  # Ora i valori sono tra 0 e 1
    # Correggi l'orientamento dell'immagine (altezza, larghezza)
    image_array = np.expand_dims(image_array, axis=-1)  # (200, 50, 1)
    # Aggiungi la dimensione del batch
    image_array = np.expand_dims(image_array, axis=0)  # (1, 200, 50, 1)
    # Trasponi l'immagine per allinearla con il preprocessamento del training
    image_array = np.transpose(image_array, (0, 2, 1, 3))  # (1, 50, 200, 1)
    
    return image_array

# Funzione per decodificare le predizioni
def decode_single_prediction(pred, max_length, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# Carica i dati del captcha
direc = Path("captcha_alfanumerici/samples")

# Carica solo immagini PNG con nomi di 5 caratteri
dir_img = sorted(list(map(str, list(direc.glob("*.png")))))
dir_img = [img for img in dir_img if len(os.path.basename(img).split(".png")[0]) == 5]

img_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in dir_img]
char_img = set(char for label in img_labels for char in label)
char_img = sorted(list(char_img))

# Stampa informazioni
print("Number of dir_img found: ", len(dir_img))
print("Number of img_labels found: ", len(img_labels))
print("Number of unique char_img: ", len(char_img))
print("Characters present: ", char_img)

# Impostiamo la lunghezza massima
max_length = max([len(label) for label in img_labels])

# Char to integers
char_to_num = layers.StringLookup(vocabulary=list(char_img), mask_token=None)

# Integers to original characters
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Carica il modello
model = keras.models.load_model('captcha_alfanumerici/modello_alfanumerici.h5', custom_objects={'LayerCTC': LayerCTC})

# Creiamo un modello per la predizione, utilizzando solo l'immagine come input (senza etichette)
prediction_model = keras.models.Model(
    inputs=model.input[0],  
    outputs=model.get_layer(name="dense2").output  
)

# Vai al sito
driver = webdriver.Chrome()
driver.get("http://localhost:8000/captcha_alfanumerici/demo.html")

# Trova l'elemento dell'immagine CAPTCHA
captcha_image = driver.find_element(By.TAG_NAME, "img")

# Ottieni l'immagine come byte array
captcha_image_bytes = captcha_image.screenshot_as_png
image = Image.open(BytesIO(captcha_image_bytes))

# Preprocessa l'immagine
image_array = preprocess_image(image)
print(image_array)

# Usa il modello per fare una previsione (senza etichette)
pred = prediction_model.predict(image_array)

# Decodifica la previsione
pred_text = decode_single_prediction(pred, max_length, num_to_char)

# Trova la textbox per l'inserimento del CAPTCHA
textbox = driver.find_element(By.ID, "captcha_input")

# Scrivi la previsione nella textbox
textbox.send_keys(pred_text[0])

# Trova il pulsante di verifica e clicca su di esso
verify_button = driver.find_element(By.ID, "submit_button")
verify_button.click()

# Attendi un po' per vedere se la verifica ha avuto successo
time.sleep(10)

# Chiudi il browser
driver.quit()