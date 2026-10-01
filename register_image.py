from pathlib import Path

import face_recognition


image_path = Path("test_images/cadastro.png")

if not image_path.exists():
    raise FileNotFoundError(
        f"Imagem não encontrada: {image_path}"
    )


print(f"Carregando: {image_path}")

image = face_recognition.load_image_file(
    image_path
)

face_locations = face_recognition.face_locations(
    image
)

print(
    f"Rostos encontrados: "
    f"{len(face_locations)}"
)


if len(face_locations) != 1:
    raise RuntimeError(
        "A imagem precisa conter exatamente "
        "um rosto."
    )


face_encodings = face_recognition.face_encodings(
    image,
    face_locations,
)

if not face_encodings:
    raise RuntimeError(
        "Não foi possível gerar o encoding facial."
    )


face_encoding = face_encodings[0]


print()
print("Rosto capturado com sucesso!")
print(
    f"Encoding encontrado: "
    f"{len(face_encoding)} valores"
)


name = input(
    "Digite o nome da pessoa: "
).strip()


if not name:
    raise ValueError(
        "O nome não pode estar vazio."
    )


print()
print(f"Pessoa cadastrada: {name}")