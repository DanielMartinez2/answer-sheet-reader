import cv2
import numpy as np

# Caminho do arquivo (use o caminho real do seu arquivo digitalizado)
image_path = "Gabarito_original.jpeg"

# 1. Leitura e pré-processamento
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# 2. Encontrar contornos (para detectar a tabela)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos por área (tabelas grandes)
table_contours = [c for c in contours if cv2.contourArea(c) > 20000]
table_contours = sorted(table_contours, key=cv2.contourArea, reverse=True)

# 3. Isolar a maior tabela (esperada: área das questões)
table = table_contours[0]
x, y, w, h = cv2.boundingRect(table)
table_img = img[y:y+h, x:x+w]

# 4. Desenhar a detecção para conferência
preview = img.copy()
cv2.rectangle(preview, (x, y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("Preview da tabela detectada", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Dividir em células (15 linhas x 5 colunas)
rows, cols = 15, 5
cell_h, cell_w = h / rows, w / cols
grid_preview = table_img.copy()

for i in range(rows):
    for j in range(cols):
        x1, y1 = int(j * cell_w), int(i * cell_h)
        x2, y2 = int((j + 1) * cell_w), int((i + 1) * cell_h)
        cv2.rectangle(grid_preview, (x1, y1), (x2, y2), (0,255,0), 1)

cv2.imshow("Grade projetada", grid_preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
