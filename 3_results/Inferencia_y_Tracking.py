import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import torch

# Verificar si hay GPU disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Cargar el modelo en la GPU
model = YOLO("Modelito_v11_best.pt")
model.to(device)  # Mover el modelo explícitamente

# Abrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hacer tracking en el frame capturado
    results = model.track(
        source=frame,
        conf=0.5,
        imgsz=640,  # <-- Forzar entrada a 640x640
        persist=True,
        stream=True,
        tracker="bytetrack.yaml",
        device=device  # usar GPU
    )

    # Como stream=True, results es un generador
    for result in results:
        annotated_frame = result.plot()  # Frame anotado

        # Dimensiones del frame
        h, w = frame.shape[:2]

        # Procesar cada box detectado
        for box in result.boxes:
            cls_id = int(box.cls[0])                     # ID de la clase
            class_name = model.names[cls_id]             # Nombre de la clase
            track_id = int(box.id[0]) if box.id is not None else -1  # ID de tracking

            # Mostrar info de detección
            print(f"Track ID: {track_id} | Clase: {class_name}")

        # Mostrar el frame anotado
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()