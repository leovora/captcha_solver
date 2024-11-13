from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

counter = 0

while counter <= 5:
    # Inizializza Selenium
    driver = webdriver.Chrome()  # Assicurati di avere il driver Chrome configurato correttamente
    driver.get("http://localhost:8000/captcha_immagini/demo_yolo.html")  # Sostituisci con il percorso corretto

    # Trova l'elemento dell'immagine CAPTCHA sulla pagina
    captcha_image = driver.find_element(By.TAG_NAME, "img")

    # Ottieni l'immagine come byte array
    captcha_image = captcha_image.screenshot_as_png

    # Usa PIL per aprire l'immagine in memoria
    image = Image.open(BytesIO(captcha_image)).convert("RGB")

    # Converti l'immagine da PIL a formato numpy array leggibile da OpenCV
    img = np.array(image)

    # Cambia il formato del colore da RGB a BGR (utilizzato da OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Prepara l'immagine per il modello

    # Load names of classes and get random colors for them.
    classes = open('captcha_immagini/yolo/coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet('captcha_immagini/yolo/yolov3.cfg', 'captcha_immagini/yolo/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # First, get layer names
    ln = net.getLayerNames()

    # Get output layers
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # Extract the detected objects
    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Ensure at least one object is detected
    if len(classIDs) > 0:
        # Get the name of the first identified object
        first_object_name = classes[classIDs[0]]
        
        # Check for special case "fire hydrant"
        if first_object_name == "fire hydrant":
            first_object_name = "hydrant"
        
        print(f"First identified object: {first_object_name}")

        # Inserisci il testo previsto nella textbox del CAPTCHA
        textbox = driver.find_element(By.ID, "captcha_input") 
        textbox.send_keys(first_object_name)

        # Trova il pulsante di invio e cliccalo
        verify_button = driver.find_element(By.ID, "submit_button")  
        verify_button.click()

    # Attendi qualche secondo per vedere il risultato 
    time.sleep(3)

    # Chiudi il browser
    driver.quit()
        
    counter += 1