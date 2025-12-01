import cv2
import time
import numpy as np
import mediapipe as mp
import serial

# ============================
# CONFIGURAÇÃO DA SERIAL
# ============================
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    print("Arduino conectado com sucesso.")
except Exception as e:
    print("ERRO ao conectar no Arduino:", e)
    arduino = None

# ============================
# CONFIG FACE MESH
# ============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices dos pontos do olho direito
eye_right_idx = [33, 160, 158, 133, 153, 144]

def calcular_ear(landmarks, w, h):
    try:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_right_idx]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C)
    except:
        return 1.0

# ============================
# CAPTURA DE VÍDEO
# ============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

sono_contador = 0
EAR_LIMIAR = 0.23
FRAME_SONO = 15
alerta_enviado = False

print(">>> Sistema iniciado. Monitorando fadiga ocular...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(rgb)

    rosto = False

    if results.multi_face_landmarks:
        rosto = True
        for face_landmarks in results.multi_face_landmarks:

            # BOUNDING BOX
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # EAR
            ear = calcular_ear(face_landmarks.landmark, w, h)

            if ear < EAR_LIMIAR:
                sono_contador += 1
            else:
                sono_contador = 0
                alerta_enviado = False

    # ALARME DE RISCO
    if sono_contador >= FRAME_SONO:
        cv2.putText(frame, "PERIGO DE ACIDENTE!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if not alerta_enviado and arduino:
            arduino.write(b'1')
            print(">>> ALERTA enviado ao Arduino <<<")
            alerta_enviado = True
    else:
        if arduino:
            arduino.write(b'0')

    # SEM ROSTO
    if not rosto:
        cv2.putText(frame, "Rosto NAO detectado", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Detector de Sonolência", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ENCERRAR
cap.release()
face_mesh.close()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
