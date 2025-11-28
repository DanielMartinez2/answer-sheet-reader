
# -*- coding: utf-8 -*-
"""
Leitura de gabarito de múltipla escolha (A–E) - Versão Otimizada para Mobile/CamScanner
Correções Críticas aplicadas:
1. Remoção de sombras (Normalização de Divisão) no pré-processamento.
2. Detecção de linhas HORIZONTAIS mais robusta para curvatura da página.
3. Alinhamento de colunas VERTICAIS baseado em RECONSTRUÇÃO GEOMÉTRICA FIXA (23% para "Questão" e ancoragem por tinta).
4. Leitura de marcações com fechamento morfológico (MORPH_CLOSE) para ignorar lápis riscado e fraco.
"""

import os
import cv2
import json
import argparse
import numpy as np


# ---------- utilitários básicos ----------

def remove_shadows(gray):
    """
    Remove sombras e normaliza a iluminação usando Normalização de Divisão.
    """
    # 1. Estima o fundo (background) dilatando para remover texto e linhas finas
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_blur = cv2.medianBlur(dilated, 21)
    
    # 2. Diferença absoluta invertida -> (255 - |img - bg|) simula a divisão
    diff = 255 - cv2.absdiff(gray, bg_blur)
    
    # 3. Normaliza o resultado para usar todo o espectro 0-255
    norm_img = cv2.normalize(diff, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img


def angle_from_hough(binary):
    """Estima pequeno ângulo de skew a partir de linhas horizontais via Hough."""
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=200) 
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = theta * 180.0 / np.pi
        if deg > 90:
            deg -= 180
        if -45 <= deg <= 45:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def angle_from_vertical_hough(gray):
    """Estima o ângulo de inclinação das LINHAS VERTICAIS."""
    # Usado para correção de rotação vertical leve (deskew)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        deg = theta * 180.0 / np.pi
        # Queremos linhas que são quase verticais (90 graus)
        if 80.0 < deg < 100.0:
            angles.append(deg - 90.0) # Ângulo de desvio em relação à vertical
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image, angle_deg):
    """Rotaciona em torno do centro (deskew leve)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def cluster_positions(pos, tol):
    """Agrupa posições 1D próximas e retorna a mediana de cada grupo."""
    if not pos:
        return []
    pos = sorted(pos)
    clusters = [[pos[0]]]
    for p in pos[1:]:
        if abs(p - clusters[-1][-1]) > tol:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return [int(np.median(c)) for c in clusters]


# ---------- encontrar região da TABELA pela densidade de linhas ----------

def find_table_roi_by_line_density(gray):
    """Detecta a região da tabela (ROI) pela densidade de linhas."""
    H, W = gray.shape[:2]

    # Binarização adaptativa
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 5
    )

    # Kernels menores para tolerar curvatura/redução da imagem
    hscale = max(10, W // 60)
    vscale = max(10, H // 60)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))

    hor = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v, iterations=1)

    lines_map = cv2.bitwise_or(hor, ver)
    lines_map = cv2.dilate(lines_map, np.ones((9,9), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (lines_map > 0).astype("uint8"), connectivity=8
    )

    best_score = -1
    best_bbox = None

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        # Filtros de geometria
        if x < 10 or y < 10 or (x + w) > (W - 10) or (y + h) > (H - 10): continue
        if y < 0.25 * H: continue 
        aspect_ratio = w / float(h)
        #if not (0.7 <= aspect_ratio <= 2.5): continue
        if w < 0.15 * W or w > 0.95 * W: continue
        if h < 0.15 * H or h > 0.7 * H: continue

        # Score baseado na densidade
        hor_roi = hor[y:y + h, x:x + w]
        ver_roi = ver[y:y + h, x:x + w]
        sum_hor = np.sum(hor_roi > 0, axis=1)
        sum_ver = np.sum(ver_roi > 0, axis=0)

        if np.max(sum_hor) == 0 or np.max(sum_ver) == 0: continue

        h_count = int(np.sum(sum_hor > 0.3 * np.max(sum_hor)))
        if h_count < 8:
            continue
        v_count = int(np.sum(sum_ver > 0.3 * np.max(sum_ver)))

        score = h_count * v_count
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    # Fallback caso não ache
    if best_bbox is None:
        best_bbox = (int(W * 0.15), int(H * 0.40), int(W * 0.70), int(H * 0.45))

    x, y, w, h = best_bbox
    pad_x = 15
    pad_y = 60 
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + 2 * pad_y)
    
    return (x, y, w, h), th


# ---------- detectar cantos da tabela (Mantido para Perspectiva, mas opcional) ----------

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def find_table_corners_in_roi(tbl_gray):
    """Encontra o quadrilátero externo da tabela para correção de perspectiva."""
    h, w = tbl_gray.shape[:2]

    edges = cv2.Canny(tbl_gray, 30, 100, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1) # Ajuda a fechar bordas

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_quad = None
    min_rel_area = 0.10 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_rel_area * h * w:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx.reshape(4, 2)
            break 

    if best_quad is not None:
        return order_points(best_quad)
    
    return None # Retorna None se não achar um quadrilátero claro


# ---------- detectar grade DENTRO da região da tabela (CRÍTICA) ----------

def detect_grid_in_table(img_bgr, gray, th, table_bbox):
    """
    ### NOVO: Implementa Reconstrução Geométrica e Ancoragem por Tinta.
    Isso corrige o desalinhamento vertical (A, B, C...) e ignora o padding.
    """
    tx, ty, tw, th_h = table_bbox
    
    tbl_gray = gray[ty:ty + th_h, tx:tx + tw]
    tbl_th   = th[ty:ty + th_h, tx:tx + tw]
    tbl_bgr  = img_bgr[ty:ty + th_h, tx:tx + tw]

    # Correção de Skew Vertical
    skew_v = angle_from_vertical_hough(tbl_gray)
    if abs(skew_v) > 0.05:
        tbl_gray = rotate_image(tbl_gray, -skew_v)
        tbl_th   = rotate_image(tbl_th,   -skew_v)
        tbl_bgr  = rotate_image(tbl_bgr,  -skew_v)

    h2, w2 = tbl_th.shape[:2]
    
    # 1. Tenta corrigir perspectiva (Warp)
    corners = find_table_corners_in_roi(tbl_gray)

    if corners is not None:
        (tl, tr, br, bl) = corners
        maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(corners, dst)
        table_bgr = cv2.warpPerspective(tbl_bgr, M, (maxW, maxH))
        
        # Recalcula threshold na imagem retificada
        table_gray_warp = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
        table_th = cv2.adaptiveThreshold(
             table_gray_warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY_INV, 21, 5
        )
    else:
        # Fallback: Rotação simples
        angle = angle_from_hough(tbl_th) 
        table_bgr = rotate_image(tbl_bgr, angle)
        table_th  = rotate_image(tbl_th,  angle)

    h2, w2 = table_th.shape[:2]
    
    # ---------------------------------------------------------
    # 2. ANCORAGEM (Detecta onde a tinta REALMENTE começa e termina)
    # ---------------------------------------------------------
    col_sums = np.sum(table_th > 0, axis=0)
    ink_threshold = 0.3 * np.max(col_sums) if np.max(col_sums) > 0 else 10

    start_x = 0
    for x in range(w2):
        if col_sums[x] > ink_threshold:
            start_x = max(0, x - 2) # Recua 2px
            break
            
    end_x = w2 - 1
    for x in range(w2 - 1, 0, -1):
        if col_sums[x] > ink_threshold:
            end_x = min(w2, x + 2) # Avança 2px
            break
            
    table_width_real = end_x - start_x
    if table_width_real < 50:
        start_x = 0
        end_x = w2
        table_width_real = w2
        
    # ---------------------------------------------------------
    # 3. DIVISÃO MATEMÁTICA VERTICAL (COLUNAS)
    # ---------------------------------------------------------
    # Proporção CORRIGIDA para a coluna de números (Questão)
    num_col_ratio = 0.47  
    
    split_point = start_x + int(table_width_real * num_col_ratio)

    v_lines = []
    v_lines.append(start_x)     
    v_lines.append(split_point) # Fim da coluna "Questão" / Início "A"
    
    # Restante dividido por 5 (A, B, C, D, E)
    remaining_w = end_x - split_point
    col_width = remaining_w / 5.0
    
    for i in range(1, 6):
        line_x = split_point + int(i * col_width)
        v_lines.append(line_x)
        
    # ---------------------------------------------------------
    # 4. DETECÇÃO DE LINHAS HORIZONTAIS (LINHAS)
    # ---------------------------------------------------------
    hscale = max(5, w2 // 150) 
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    
    closed_th = cv2.morphologyEx(table_th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    hor = cv2.morphologyEx(closed_th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    
    sum_hor = np.sum(hor > 0, axis=1)
    h_indices = np.where(sum_hor > (0.15 * np.max(sum_hor)))[0].tolist()
    htol = max(5, h2 // 120)
    h_lines = cluster_positions(h_indices, htol)

    # Filtro de espaçamento mínimo (para não pegar cabeçalhos como linha de resposta)
    h_filtered = []
    if h_lines:
        h_filtered.append(h_lines[0])
        for i in range(1, len(h_lines)):
            if (h_lines[i] - h_filtered[-1]) > (h2 / 70):
                h_filtered.append(h_lines[i])
    h_lines = h_filtered
    
    # Fallback/garantia de número mínimo de linhas
    if len(h_lines) < 10:
        h_lines = [int(y) for y in np.linspace(0, h2 - 1, 17)]

    return 0, 0, w2, h2, v_lines, h_lines, table_bgr, table_th


# ---------- leitura das marcações (Ajustada) ----------

def read_marks(rot_img, rot_bin, bbox, v_lines, h_lines,
               min_mark_percent=5.0,  
               tie_margin_percent=3.0,
               max_rows=16):

    x, y, w, h = bbox
    table_vis  = rot_img[y:y + h, x:x + w].copy()
    
    # As colunas de respostas (A-E) estão entre v_lines[1] e v_lines[6]
    target_cols_indices = [1, 2, 3, 4, 5] 
    letras = ["A", "B", "C", "D", "E"]
    respostas = []

    if len(v_lines) < 7:
        print("Aviso: Número insuficiente de linhas verticais detectadas.")
        return [], table_vis

    # Assume que a primeira ou segunda linha horizontal é o início das questões (01)
    start_i = 0
    if len(h_lines) >= 16: 
        start_i = 2 # Pula o cabeçalho se ele foi detectado como uma linha
        
        
    count_read = 0
    
    for i in range(start_i, len(h_lines) - 1):
        if count_read >= max_rows: break
        
        y1, y2 = h_lines[i], h_lines[i + 1]
        if (y2 - y1) < 10: continue 
        
        densidades = []
        cell_boxes = []

        for k, col_idx in enumerate(target_cols_indices):
            x1 = v_lines[col_idx]
            x2 = v_lines[col_idx + 1]

            # Padding generoso para focar no centro da célula
            margin_x = int((x2 - x1) * 0.1) 
            margin_y = int((y2 - y1) * 0.1)
            
            roi = rot_bin[y + y1 + margin_y : y + y2 - margin_y, 
                          x + x1 + margin_x : x + x2 - margin_x]
            
            if roi.size == 0:
                densidades.append(0.0)
                cell_boxes.append((x1, y1, x2, y2))
                continue

            # Engrossar a marcação (lápis fraco)
            kernel_fill = np.ones((3,3), np.uint8)
            # Fechamento Morfológico para preencher riscos
            roi_filled = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
            
            ink = np.sum(roi_filled > 0)
            fill_percent = (ink / roi_filled.size) * 100.0            
            
            densidades.append(fill_percent)
            cell_boxes.append((x1, y1, x2, y2))

        if not densidades:
            respostas.append("")
            count_read += 1
            continue

        best_idx = np.argmax(densidades)
        best_val = densidades[best_idx]
        
        sorted_vals = sorted(densidades, reverse=True)
        second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
        
        # Visualização (Cinza nas células lidas)
        for bx1, by1, bx2, by2 in cell_boxes:
             cv2.rectangle(table_vis, (bx1, by1), (bx2, by2), (200,200,200), 1)

        if best_val < min_mark_percent:
            respostas.append("")
        elif (best_val - second_val) < tie_margin_percent and second_val > min_mark_percent:
            respostas.append("AMB")
            bx1, by1, bx2, by2 = cell_boxes[best_idx]
            cv2.rectangle(table_vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        else:
            letra = letras[best_idx]
            respostas.append(letra)
            bx1, by1, bx2, by2 = cell_boxes[best_idx]
            cv2.rectangle(table_vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(table_vis, f"{letra}", (bx1+5, by1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        count_read += 1

    return respostas, table_vis


# ---------- pipeline completo ----------

def run(input_path, outdir, show_stats=False):
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {input_path}")

    # 1) Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = remove_shadows(gray)
    cv2.imwrite(out("00_clean_shadows.png"), gray)

    # 2) Encontrar ROI da tabela
    table_bbox, th = find_table_roi_by_line_density(gray)
    tx, ty, tw, th_h = table_bbox
    
    debug_roi = img.copy()
    cv2.rectangle(debug_roi, (tx, ty), (tx+tw, ty+th_h), (0,0,255), 3)
    cv2.imwrite(out("01_table_roi.png"), debug_roi)

    # 3) Detectar grade e alinhar (Usa a geometria fixa)
    x, y, w, h, v_lines, h_lines, rot_tbl_bgr, rot_tbl_th = detect_grid_in_table(
        img, gray, th, table_bbox
    )

    # Overlays de debug da GRADE CORRIGIDA
    grid_overlay = rot_tbl_bgr.copy()
    for xl in v_lines: cv2.line(grid_overlay, (xl, 0), (xl, h), (0, 255, 0), 1)
    for yl in h_lines: cv2.line(grid_overlay, (0, yl), (w, yl), (0, 255, 0), 1)
    cv2.imwrite(out("02_grid_detected.png"), grid_overlay)

    # 4) Leitura
    respostas, marks_overlay = read_marks(
        rot_tbl_bgr, rot_tbl_th, (x, y, w, h), v_lines, h_lines
    )
    cv2.imwrite(out("03_final_marks.png"), marks_overlay)

    # Salva saídas
    respostas_limpa = [r if r in ["A","B","C","D","E"] else "" for r in respostas]
    
    with open(out("respostas.txt"), "w", encoding="utf-8") as f:
        f.write(str(respostas_limpa) + "\n")
        
    with open(out("respostas.json"), "w", encoding="utf-8") as f:
        json.dump({"respostas": respostas_limpa}, f, indent=2)

    if show_stats:
        print(f"Detectadas {len(respostas)} linhas.")
        print("Resultado:", respostas_limpa)
        print(f"Salvo em: {outdir}")

    return respostas, respostas_limpa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Imagem de entrada")
    ap.add_argument("--outdir", default="./out", help="Pasta de saída")
    ap.add_argument("--show-stats", action="store_true")
    args = ap.parse_args()
    
    try:
        run(args.input, args.outdir, args.show_stats)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()