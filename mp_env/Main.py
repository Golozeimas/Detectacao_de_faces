import cv2
import mediapipe as mp

# Inicializa a captura de vídeo
webcam = cv2.VideoCapture(0)

# Inicializa a solução do rostos
solucao_reconhecimento_de_rosto = mp.solutions.face_detection

# Reconhece os rostos, com a detectação de faces do mediapipe
reconhecedor_de_rostos = solucao_reconhecimento_de_rosto.FaceDetection()

# A solução que desenha no rosto as figuras
desenho = mp.solutions.drawing_utils

while True:
    # Ler as informações do webcam
    verdadeiro, frame = webcam.read() # Retorna duas respostas
    if not verdadeiro:
        break
    
    # Lista de rostos que são processados em cada frame
    lista_rostos = reconhecedor_de_rostos.process(frame)
    
    # Lista de rosto que detecta na cãmera
    if lista_rostos.detections:
        # Percorre todos os rostos na lista de rostos que podem aparecer
        for rosto in lista_rostos.detections:
            # Desenha o rosto na câmera
            desenho.draw_detection(frame, rosto)
    
    # Colocando o frame para ser mostrado, e dando nome para a janela
    cv2.imshow("câmera" ,  frame)
    
    # Espera 5 milessimos para sempre, e ao clicar em 'q' sai do programa
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Desliga a câmera pós uso
webcam.release()

cv2.destroyAllWindows()