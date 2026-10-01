import cv2
import face_recognition

from face_database import (
    load_known_faces,
    recognize_faces,
)

known_encodings, known_names = load_known_faces(
    "known_faces"
)

print()
print(f"Base carregada: {known_names}")

image = cv2.imread(
    "test_images/image.png"
)

if image is None:
    raise FileNotFoundError(
        "Não foi possível encontrar a imagem"
    )

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

face_locations = face_recognition.face_locations(
    rgb_image
)

face_encodings = face_recognition.face_encodings(
    rgb_image,
    face_locations
)

print(
    f"Rostos encontrados: "
    f"{len(face_encodings)}"
)

for face_location, face_encoding in zip(
    face_locations,
    face_encodings
):

    name, distance = recognize_faces(
        face_encoding,
        known_encodings,
        known_names
    )

    print(
        f"Rosto reconhecido: {name} "
        f"| Distância: {distance:.3f}"
        if distance is not None
        else f"Rosto reconhecido: {name}"
    )

    top, right, bottom, left = face_location

    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        (0, 255, 0),
        2
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        name,
        font,
        font_scale,
        thickness,
    )

    label_top = max(
        0,
        top - text_height - 10,
    )

    label_bottom = top

    cv2.rectangle(
        image,
        (left, label_top),
        (left + text_width + 10, label_bottom),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        image,
        name,
        (left + 5, label_bottom - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

cv2.imshow(
    "Reconhecimento Facial",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()