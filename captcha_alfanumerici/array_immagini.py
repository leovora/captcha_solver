import os
import json

def get_all_paths(directory, base_directory):
    paths = []
    # os.walk traversa la directory e le sue sottodirectory
    for root, _, files in os.walk(directory):
        for file in files:
            # Controlla se il file è un'immagine .png e ha un nome di 5 caratteri
            if file.endswith('.png') and len(os.path.splitext(file)[0]) == 5:
                # Costruisce il percorso completo per ciascun file
                full_path = os.path.join(root, file)
                # Calcola il percorso relativo alla cartella base
                relative_path = os.path.relpath(full_path, base_directory)
                paths.append(relative_path)
    return paths

# costanti
base_directory = 'captcha_alfanumerici'  # La cartella madre da omettere nel percorso
directory_path = os.path.join(base_directory, 'samples')  # La cartella che contiene le immagini
all_paths = get_all_paths(directory_path, base_directory)

# Stampa l'array di percorsi
print(all_paths)

# Scrivi l'array su un file JSON
with open('files.json', 'w') as f:
    json.dump(all_paths, f, indent=4)  # Usa indent per formattare meglio il JSON