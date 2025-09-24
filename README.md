# Sheldonscope – Facial Emotion Recognition
Un'applicazione Python che rileva landmark facciali e emozioni in tempo reale tramite webcam, combinando MediaPipe Face Mesh e FER (Facial Emotion Recognition).


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
- Interfaccia semplice che mostra direttamente il video con annotazioni.


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


## Modelli usati:
- FER (CNN addestrata su FER2013) per il riconoscimento emozioni.
- MTCNN per il rilevamento accurato dei volti.
- MediaPipe Face Mesh per l’estrazione dei landmark facciali.


## Utilizzo
Avviare il programma principale:
python main.py
Premere q per uscire.


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

