import cv2
import numpy as np
import os

# Carregar as configurações do YOLO
net = cv2.dnn.readNet("lp-detection-layout-classification.weights", "lp-detection-layout-classification.cfg")
classes = []
with open("lp-detection-layout-classification.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Leitura das configurações do YOLO a partir do arquivo lpscr.data
with open("lp-detection-layout-classification.data", "r") as data_file:
    data_content = data_file.readlines()
    for line in data_content:
        if line.startswith("names"):
            names_file = line.split("=")[1].strip()

# Leitura das imagens da pasta "imagens"
image_folder = "resultados"
output_folder = "placas"

# Criar pasta de resultados se não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Processamento de cada imagem na pasta
for img_name in images:
    img_path = os.path.join(image_folder, img_name)

    # Leitura da imagem
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    # Conversão para o formato da rede neural
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Detecção e desenho das caixas delimitadoras
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Adicionar imagens recortadas à pasta de resultados
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cropped_img = img[y:y + h, x:x + w]
            result_path = os.path.join(output_folder, f"placa_{img_name}")
            cv2.imwrite(result_path, cropped_img)
