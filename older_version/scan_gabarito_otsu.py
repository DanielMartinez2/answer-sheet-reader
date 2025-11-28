# scan_gabarito_gemini.py (Versão Corrigida - Híbrido)
# -*- coding: utf-8 -*-
import os
import cv2
import json
import argparse
import numpy as np

# ---------- utilitários básicos ----------

def remove_shadows(gray):
    """Remove sombras e normaliza a iluminação usando Normalização de Divisão."""
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_blur = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_blur)
    norm_img = cv2.normalize(diff, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def angle_from_hough(binary):
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=200) 
    if lines is None: return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = theta * 180.0 / np.pi
        if deg > 90: deg -= 180
        if -45 <= deg <= 45: angles.append(deg)
    return float(np.median(angles)) if angles else 0.0

def angle_from_vertical_hough(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None: return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        deg = theta * 180.0 / np.pi
        if 80.0 < deg < 100.0: angles.append(deg - 90.0) 
    return float(np.median(angles)) if angles else 0.0

def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

def cluster_positions(pos, tol):
    if not pos: return []
    pos = sorted(pos)
    clusters = [[pos[0]]]
    for p in pos[1:]:
        if abs(p - clusters[-1][-1]) > tol:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return [int(np.median(c)) for c in clusters]

# ---------- encontrar região da TABELA ----------

def find_table_roi_by_line_density(gray):
    H, W = gray.shape[:2]
    # Binarização suave (C=5) para pegar linhas fracas
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 5
    )

    hscale = max(10, W // 60)
    vscale = max(10, H // 60)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))

    hor = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    lines_map = cv2.bitwise_or(hor, ver)
    # Dilatação agressiva (9,9) para conectar linhas quebradas (cabeçalho/rodapé)
    lines_map = cv2.dilate(lines_map, np.ones((9,9), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (lines_map > 0).astype("uint8"), connectivity=8
    )

    best_score = -1
    best_bbox = None

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if x < 10 or y < 10 or (x + w) > (W - 10) or (y + h) > (H - 10): continue
        # Filtro de posição relaxado (0.05) para aceitar tabelas no topo
        if y < 0.05 * H: continue 
        
        if w < 0.15 * W or w > 0.95 * W: continue
        if h < 0.15 * H or h > 0.7 * H: continue

        hor_roi = hor[y:y + h, x:x + w]
        ver_roi = ver[y:y + h, x:x + w]
        sum_hor = np.sum(hor_roi > 0, axis=1)
        sum_ver = np.sum(ver_roi > 0, axis=0)

        if np.max(sum_hor) == 0 or np.max(sum_ver) == 0: continue

        h_count = int(np.sum(sum_hor > 0.3 * np.max(sum_hor)))
        if h_count < 8: continue
        v_count = int(np.sum(sum_ver > 0.3 * np.max(sum_ver)))

        score = h_count * v_count
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    if best_bbox is None:
        best_bbox = (int(W * 0.15), int(H * 0.40), int(W * 0.70), int(H * 0.45))

    x, y, w, h = best_bbox
    # Padding pequeno para evitar ruído externo
    pad_x = 5
    pad_y = 15 
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + 2 * pad_y)
    
    return (x, y, w, h), th

# ---------- auxiliares de geometria ----------

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def find_table_corners_in_roi(tbl_gray):
    h, w = tbl_gray.shape[:2]
    edges = cv2.Canny(tbl_gray, 30, 100, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1) 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_quad = None
    for cnt in contours:
        if cv2.contourArea(cnt) < 0.10 * h * w: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx.reshape(4, 2); break 
    return order_points(best_quad) if best_quad is not None else None

# ---------- detectar grade (Geometria Fixa) ----------

def detect_grid_in_table(img_bgr, gray, th, table_bbox):
    tx, ty, tw, th_h = table_bbox
    tbl_gray = gray[ty:ty + th_h, tx:tx + tw]
    tbl_th   = th[ty:ty + th_h, tx:tx + tw]
    tbl_bgr  = img_bgr[ty:ty + th_h, tx:tx + tw]

    skew_v = angle_from_vertical_hough(tbl_gray)
    if abs(skew_v) > 0.05:
        tbl_gray = rotate_image(tbl_gray, -skew_v)
        tbl_th   = rotate_image(tbl_th,   -skew_v)
        tbl_bgr  = rotate_image(tbl_bgr,  -skew_v)

    # Warp Perspective
    corners = find_table_corners_in_roi(tbl_gray)
    if corners is not None:
        (tl, tr, br, bl) = corners
        maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(corners, dst)
        table_bgr = cv2.warpPerspective(tbl_bgr, M, (maxW, maxH))
        table_gray_warp = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
        # Binarização robusta com Gaussian Adaptive Threshold
        table_th = cv2.adaptiveThreshold(
             table_gray_warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
             cv2.THRESH_BINARY_INV, 21, 5
        )
    else:
        angle = angle_from_hough(tbl_th) 
        table_bgr = rotate_image(tbl_bgr, angle)
        table_th  = rotate_image(tbl_th,  angle)

    h2, w2 = table_th.shape[:2]
    
    # --- ANCORAGEM ---
    col_sums = np.sum(table_th > 0, axis=0)
    ink_threshold = 0.3 * np.max(col_sums) if np.max(col_sums) > 0 else 10
    start_x = 0
    for x in range(w2):
        if col_sums[x] > ink_threshold: start_x = max(0, x - 2); break
    end_x = w2 - 1
    for x in range(w2 - 1, 0, -1):
        if col_sums[x] > ink_threshold: end_x = min(w2, x + 2); break
            
    table_width_real = end_x - start_x
    if table_width_real < 50: start_x=0; end_x=w2; table_width_real=w2
        
    # --- DIVISÃO VERTICAL (0.47) ---
    num_col_ratio = 0.47  
    split_point = start_x + int(table_width_real * num_col_ratio)
    v_lines = [start_x, split_point]
    remaining_w = end_x - split_point
    col_width = remaining_w / 5.0
    for i in range(1, 6):
        v_lines.append(split_point + int(i * col_width))
        
    # --- DETECÇÃO HORIZONTAL ---
    hscale = max(5, w2 // 150) 
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    closed_th = cv2.morphologyEx(table_th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    hor = cv2.morphologyEx(closed_th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    sum_hor = np.sum(hor > 0, axis=1)
    h_indices = np.where(sum_hor > (0.15 * np.max(sum_hor)))[0].tolist()
    h_lines = cluster_positions(h_indices, max(5, h2 // 120))

    h_filtered = []
    if h_lines:
        h_filtered.append(h_lines[0])
        for i in range(1, len(h_lines)):
            if (h_lines[i] - h_filtered[-1]) > (h2 / 70):
                h_filtered.append(h_lines[i])
    h_lines = h_filtered
    
    # Fallback para 17 linhas (Cabeçalho + 15 Questões)
    if len(h_lines) < 10:
        h_lines = [int(y) for y in np.linspace(0, h2 - 1, 17)]

    return 0, 0, w2, h2, v_lines, h_lines, table_bgr, table_th


# ---------- LEITURA DAS MARCAÇÕES (Adaptive Global + Morph Close Local) ----------

def read_marks(rot_img, rot_bin, bbox, v_lines, h_lines,
               min_mark_percent=5.0,  
               tie_margin_percent=3.0,
               max_questions=15):

    x, y, w, h = bbox
    table_vis = rot_img[y:y + h, x:x + w].copy()
    
    # rot_bin é a imagem BINÁRIA GLOBAL (Adaptive Threshold)
    
    target_cols_indices = [1, 2, 3, 4, 5] 
    letras = ["A", "B", "C", "D", "E"]
    respostas = []

    if len(v_lines) < 7: return [], table_vis

    # Lógica para pular cabeçalho
    start_i = 0
    if len(h_lines) >= 16: 
        start_i = 2 
        
    count_read = 0
    
    for i in range(start_i, len(h_lines) - 1):
        if count_read >= max_questions: break
        
        y1, y2 = h_lines[i], h_lines[i + 1]
        if (y2 - y1) < 10: continue 
        
        densidades = []
        cell_boxes = []

        for k, col_idx in enumerate(target_cols_indices):
            x1 = v_lines[col_idx]
            x2 = v_lines[col_idx + 1]

            # Margem segura (20%)
            margin_x = int((x2 - x1) * 0.20) 
            margin_y = int((y2 - y1) * 0.20)
            
            # Recorte na imagem BINÁRIA GLOBAL (Adaptive Threshold)
            cell_thresh = rot_bin[y1 + margin_y : y2 - margin_y, 
                                  x1 + margin_x : x2 - margin_x]
            
            if cell_thresh.size == 0:
                densidades.append(0.0); cell_boxes.append((x1, y1, x2, y2)); continue

            # --- MORPH CLOSE para preencher o "buraco de rosquinha" (AQUI ESTÁ A CORREÇÃO) ---
            # Preenche pequenos buracos nas marcas (resolvendo o Arquivo 06)
            kernel = np.ones((3, 3), np.uint8)
            cell_closed = cv2.morphologyEx(
                cell_thresh, 
                cv2.MORPH_CLOSE, 
                kernel, 
                iterations=1
            )
            
            ink = np.sum(cell_closed > 0)
            fill_percent = (ink / cell_closed.size) * 100.0
            
            densidades.append(fill_percent)
            cell_boxes.append((x1, y1, x2, y2))

        if not densidades:
            respostas.append(""); count_read += 1; continue

        best_idx = np.argmax(densidades)
        best_val = densidades[best_idx]
        sorted_vals = sorted(densidades, reverse=True)
        second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
        
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
            cv2.putText(table_vis, f"{letra} {int(best_val)}%", (bx1+2, by1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        count_read += 1

    return respostas, table_vis


# ---------- pipeline completo ----------

def run(input_path, outdir, show_stats=False):
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)

    img = cv2.imread(input_path)
    if img is None: raise FileNotFoundError(f"Erro ao abrir: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = remove_shadows(gray)
    cv2.imwrite(out("00_clean_shadows.png"), gray)

    table_bbox, th = find_table_roi_by_line_density(gray)
    tx, ty, tw, th_h = table_bbox
    
    debug_roi = img.copy()
    cv2.rectangle(debug_roi, (tx, ty), (tx+tw, ty+th_h), (0,0,255), 3)
    cv2.imwrite(out("01_table_roi.png"), debug_roi)

    x, y, w, h, v_lines, h_lines, rot_tbl_bgr, rot_tbl_th = detect_grid_in_table(
        img, gray, th, table_bbox
    )

    grid_overlay = rot_tbl_bgr.copy()
    for xl in v_lines: cv2.line(grid_overlay, (xl, 0), (xl, h), (0, 255, 0), 1)
    for yl in h_lines: cv2.line(grid_overlay, (0, yl), (w, yl), (0, 255, 0), 1)
    cv2.imwrite(out("02_grid_detected.png"), grid_overlay)

    respostas, marks_overlay = read_marks(
        rot_tbl_bgr, rot_tbl_th, (x, y, w, h), v_lines, h_lines
    )
    cv2.imwrite(out("03_final_marks.png"), marks_overlay)

    respostas_limpa = [r if r in ["A","B","C","D","E"] else "" for r in respostas]
    
    with open(out("respostas.txt"), "w", encoding="utf-8") as f:
        f.write(str(respostas_limpa) + "\n")
        
    with open(out("respostas.json"), "w", encoding="utf-8") as f:
        json.dump({"respostas": respostas_limpa}, f, indent=2)

    if show_stats:
        print(f"Lidas {len(respostas)} linhas. Resposta: {respostas_limpa}")

    return respostas, respostas_limpa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--show-stats", action="store_true")
    args = ap.parse_args()
    try: run(args.input, args.outdir, args.show_stats)
    except Exception as e: print(e)

if __name__ == "__main__":
    main()