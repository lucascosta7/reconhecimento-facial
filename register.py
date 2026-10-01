import cv2
import face_recognition

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam.")

print("Webcam aberta.")
print("Posicione o rosto na câmera.")
print("Pressione ENTER para cadastrar.")
print("Pressione ESC para cancelar.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Rostos encontrados: {len(face_locations)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Cadastro Facial", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:
        break

    if key == 27:
        print("Cadastro cancelado.")
        camera.release()
        cv2.destroyAllWindows()
        raise SystemExit

camera.release()
cv2.destroyAllWindows()

if len(face_locations) != 1:
    raise RuntimeError(
        "O cadastro precisa ter exatamente ."
        "um rosto na imagem."               
    )

face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

if not face_encodings:
    raise RuntimeError("Não foi possível gerar o encoding facial.")

face_encoding = face_encodings[0]

print()
print("Rosto capturado com sucesso!")
print(
    f"Encoding encontrado: "
    f"{len(face_encoding)} valores"
)

name = input("Digite o nome da pessoa: ").strip()

if not name:
    raise ValueError("O nome não pode estar vazio.")

print()
print(f"Pessoa cadastrada: {name}")