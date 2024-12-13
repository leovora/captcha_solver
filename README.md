# CAPTCHA Solver

Questo repository contiene il codice per creare, addestrare e testare modelli di machine learning progettati per risolvere diversi tipi di CAPTCHA. I modelli sono in grado di riconoscere e risolvere CAPTCHA alfanumerici, logici e basati su immagini.

## Struttura del Repository

```bash
captcha_solver
├── captcha_alfanumerici   -> contiene file per testare i modelli alfanumerici (demo.py, demo_preaddestrato.py)
├── captcha_immagini       -> contiene file per testare i modelli sulle immagini (demo.py, demo_preaddestrato.py)
├── captcha_logici         -> contiene file per testare i modelli logici (demo.py, demo_preaddestrato.py)
├── modelli                -> contiene gli script per la creazione e l’addestramento dei modelli
```

## Istruzioni

### Avvia il server

Per eseguire i test, è necessario avviare un server web locale utilizzando Python:

```bash
python3 -m http.server 8000
```

Esegui gli script

Apri un secondo terminale per eseguire gli script di test. Puoi testare i modelli utilizzando i seguenti script:
- demo.py - per testare i nostri modelli.
- demo_preaddestrato.py - per testare i modelli pre-addestrati.

### Requisiti

Installa le seguenti dipendenze per far funzionare il progetto:

```bash
pip install scikit-image scikit-learn tensorflow torch torchvision torchaudio matplotlibPillow opencv-python selenium transformers -q datasets jiwer evaluate wandb pandas pydot graphviz
```
#
### Progetto realizzato da:
- Leonardo Vorabbi
- Carlotta Nunziati
