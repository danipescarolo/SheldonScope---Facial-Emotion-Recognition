# Sheldonscope – Facial Emotion Recognition
Un'applicazione Python che rileva landmark facciali e emozioni in tempo reale tramite webcam, video o immagini, combinando MediaPipe Face Mesh e FER (Facial Emotion Recognition).

## Funzionalità
- Rilevamento in tempo reale del volto tramite **MediaPipe**.
- Estrazione di occhi, naso, bocca e contorno facciale.
- Riconoscimento delle emozioni principali:  
  - Arrabbiato  
  - Disgustato  
  - Impaurito  
  - Felice  
  - Triste  
  - Sorpreso  
  - Neutro
- Ridimensionamento automatico di video e immagini a **640x480** mantenendo le proporzioni.
- Interfaccia semplice che mostra direttamente il video o l’immagine con annotazioni.

## Requisiti
Installare le librerie necessarie (testato con Python 3.11):
pip install -r requirements.txt


Nota: aprire il file
<.venv>/Lib/site-packages/fer/classes.py
e commentare la riga:
# from moviepy.editor import *


## Contenuto del file requirements.txt:
opencv-python==4.11.0.86
mediapipe==0.10.21
numpy==1.26.4
fer==22.5.1
tensorflow==2.18.0
mtcnn==0.1.1
Pillow==10.2.0


## Modelli usati:
- FER (CNN addestrata su FER2013) per il riconoscimento emozioni.
- MTCNN per il rilevamento accurato dei volti.
- MediaPipe Face Mesh per l’estrazione dei landmark facciali.


## Utilizzo
Avviare il programma principale:

Per webcam (default):
python main.py


Per un video:
python main.py video percorso_al_video.mp4


Per un’immagine:
python main.py image percorso_immagine.jpg


Premere q per uscire (webcam/video).
Per le immagini, il programma si chiuderà dopo la visualizzazione.


## Struttura del progetto
.
├── main.py                # Avvio del programma (Sheldonscope)
├── facial_recognition.py  # Rilevamento volto e landmark con MediaPipe
├── emotion_recognition.py # Riconoscimento emozioni con FER
├── requirements.txt       # Librerie richieste
└── README.md              # Questo file


## Licenza
Questo progetto è rilasciato sotto la licenza MIT.
Sentiti libero di utilizzarlo e modificarlo.

## Autore
Daniela Pescarolo

