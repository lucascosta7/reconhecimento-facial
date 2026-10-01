from pathlib import Path

import face_recognition
import numpy as np


def load_known_faces(
    folder: str,
):

    known_encodings = []
    known_names = []

    base_path = Path(folder)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {folder}"
        )

    for person_folder in sorted(base_path.iterdir()):

        if not person_folder.is_dir():
            continue

        person_name = person_folder.name

        for image_path in sorted(person_folder.iterdir()):

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            print(
                f"Processando: "
                f"{person_name}/{image_path.name}"
            )

            image = face_recognition.load_image_file(
                image_path
            )

            encodings = face_recognition.face_encodings(
                image
            )

            if not encodings:
                print(
                    f" Nenhum rosto encontrado em "
                    f"{image_path.name}"
                )
                continue

            encoding = encodings[0]

            known_encodings.append(encoding)
            known_names.append(person_name)

            print(
                f" Pessoa adicionada: "
                f"{person_name}"
            )

    return known_encodings, known_names


def recognize_faces(
    face_encoding,
    known_encodings,
    known_names,
    threshold: float = 0.5,
):

    if not known_encodings:
        return "Desconhecido", None

    distances = face_recognition.face_distance(
        known_encodings,
        face_encoding,
    )

    best_match_index = int(
        np.argmin(distances)
    )

    best_distance = float(
        distances[best_match_index]
    )

    if best_distance <= threshold:
        name = known_names[best_match_index]
    else:
        name = "Desconhecido"

    return name, best_distance