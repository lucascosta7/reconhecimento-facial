import cv2
import face_recognition

from face_database import (
    load_known_faces,
    recognize_faces,
)


# Carrega os rostos cadastrados
known_encodings, known_names = load_known_faces(
    "known_faces"
)

print()
print(f"Base carregada: {known_names}")


# Abre a webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Não foi possível acessar a webcam."
    )


while True:
    ret, frame = camera.read()

    if not ret:
        print("Não foi possível capturar o frame.")
        break

    # OpenCV utiliza BGR.
    # face_recognition utiliza RGB.
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    # Detecta os rostos
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    # Gera o encoding de cada rosto
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations,
    )

    # Processa cada rosto encontrado
    for face_location, face_encoding in zip(
        face_locations,
        face_encodings,
    ):

        name, distance = recognize_faces(
            face_encoding,
            known_encodings,
            known_names,
        )

        print(
            f"Rosto reconhecido: "
            f"{name} | "
            f"Distância: {distance:.3f}"
        )

        top, right, bottom, left = face_location

        # Caixa ao redor do rosto
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2,
        )

        # Texto que aparecerá na tela
        label = f"{name} ({distance:.2f})"

        cv2.putText(
            frame,
            label,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow(
        "Reconhecimento Facial",
        frame,
    )

    # ESC encerra o programa
    if cv2.waitKey(1) == 27:
        break


camera.release()
cv2.destroyAllWindows()