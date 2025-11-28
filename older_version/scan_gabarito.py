# scan_gabarito.py
# -*- coding: utf-8 -*-
"""
Detecção da grade + leitura das marcações (A-E) em folha de gabarito.

Uso:
  python scan_gabarito.py --input Gabarito_preenchido.jpeg --outdir ./out --show-stats
"""

import os
import cv2
import json
import argparse
import numpy as np


# ------------ Helpers ------------
def angle_from_hough(binary):
    """Estima pequeno ângulo de skew a partir de linhas com Hough."""
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=250)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = (theta - np.pi / 2) * 180.0 / np.pi  # relativo ao vertical
        if -45 <= deg <= 45:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def cluster_positions(pos, tol):
    """Agrupa posições 1D próximas (tolerância em px) e retorna medianas."""
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


def find_table_bbox(line_map):
    """Maior contorno do mapa de linhas como bbox da tabela."""
    contours, _ = cv2.findContours(line_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)  # (x, y, w, h)


# ------------ Grade ------------
def detect_grid(img_bgr):
    """Deskew + morfologia + picos → retorna (x,y,w,h, v_lines, h_lines, rot_img, rot_bin)"""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)

    # deskew pela Hough (usando horizontais)
    kernel_h_tmp = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 80), 1))
    hor_tmp = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h_tmp, iterations=1)
    angle = angle_from_hough(hor_tmp)

    rot_img = rotate_image(img_bgr, angle)
    rot_th  = rotate_image(th, angle)
    H, W = rot_th.shape[:2]

    # morfologia
    hscale = max(12, W // 90)
    vscale = max(12, H // 90)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))
    hor = cv2.morphologyEx(rot_th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(rot_th, cv2.MORPH_OPEN, kernel_v, iterations=1)

    lines_map = cv2.addWeighted(hor, 0.5, ver, 0.5, 0)
    lines_map = cv2.morphologyEx(lines_map, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # bbox da tabela
    bbox = find_table_bbox(lines_map)
    if bbox is None:
        # fallback: página inteira
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = bbox

    # projeções no ROI
    hor_roi = hor[y:y + h, x:x + w]
    ver_roi = ver[y:y + h, x:x + w]

    sum_ver = np.sum(ver_roi > 0, axis=0)  # por coluna
    sum_hor = np.sum(hor_roi > 0, axis=1)  # por linha

    # limiares relativos aos picos
    v_indices = np.where(sum_ver > (0.60 * np.max(sum_ver)))[0].tolist()
    h_indices = np.where(sum_hor > (0.50 * np.max(sum_hor)))[0].tolist()

    vtol = max(5, w // 300)
    htol = max(5, h // 300)
    v_lines = cluster_positions(v_indices, vtol)
    h_lines = cluster_positions(h_indices, htol)

    # garantir bordas
    if 0 not in v_lines: v_lines = [0] + v_lines
    if (w - 1) not in v_lines: v_lines += [w - 1]
    if 0 not in h_lines: h_lines = [0] + h_lines
    if (h - 1) not in h_lines: h_lines += [h - 1]

    v_lines = sorted(set(v_lines))
    h_lines = sorted(set(h_lines))

    return (x, y, w, h, v_lines, h_lines, rot_img, rot_th)


# ------------ Leitura de marcações ------------
def read_marks(rot_img, rot_bin, bbox, v_lines, h_lines,
               min_mark_percent=20.0, tie_margin_percent=5.0,
               skip_top_rows=1, max_rows=15):
    """
    Lê marcações A–E.
    - skip_top_rows: quantas faixas (linhas) ignorar no topo (cabeçalho = 1)
    - max_rows: número de questões a ler (15)
    """
    x, y, w, h = bbox
    table_gray = cv2.cvtColor(rot_img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    table_vis  = rot_img[y:y + h, x:x + w].copy()

    letras = ["A", "B", "C", "D", "E"]
    respostas = []

    # ---------- definir quais colunas são QUESTÃO e A–E ----------
    # larguras entre linhas verticais
    widths = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines) - 1)]
    if not widths:
        return [], table_vis

    # considera "colunas de verdade" as que têm largura >= 20% da maior
    min_width = max(10, int(0.2 * max(widths)))
    big_cols = [i for i, wcol in enumerate(widths) if wcol >= min_width]

    # esperamos: big_cols = [ idx_Q, idx_A, idx_B, idx_C, idx_D, idx_E ]
    # pegamos a partir da segunda (1..) as 5 alternativas
    if len(big_cols) < 6:
        # fallback: mantém comportamento antigo (melhor do que quebrar)
        alt_col_indices = list(range(1, min(len(v_lines) - 1, 6)))
    else:
        alt_col_indices = big_cols[1:6]   # A..E

    # limites de linhas a processar (pula cabeçalho, lê 15 questões)
    start_i = min(skip_top_rows, len(h_lines) - 2)
    end_i   = min(start_i + max_rows, len(h_lines) - 1)

    for i in range(start_i, end_i):
        y1, y2 = h_lines[i], h_lines[i + 1]
        densidades, cell_boxes = [], []

        # percorre apenas colunas A–E
        for k, col_idx in enumerate(alt_col_indices):
            x1, x2 = v_lines[col_idx], v_lines[col_idx + 1]
            cell = table_gray[y1:y2, x1:x2]

            # cores de debug por alternativa
            debug_colors = [
                (0, 255, 0),    # A
                (255, 0, 0),    # B
                (0, 0, 255),    # C
                (0, 255, 255),  # D
                (255, 0, 255),  # E
            ]
            color_debug = debug_colors[k] if k < len(debug_colors) else (0, 255, 0)

            # binarização local + borda interna
            cell_blur = cv2.GaussianBlur(cell, (3, 3), 0)
            _, bin_cell = cv2.threshold(cell_blur, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            inner = bin_cell[3:-3, 3:-3]
            percent = 100.0 * np.sum(inner > 0) / max(1, inner.size)

            densidades.append(percent)
            cell_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(table_vis, (x1, y1), (x2, y2), color_debug, 1)

        # decisão da linha
        if not densidades:
            respostas.append("")
            continue

        order = np.argsort(densidades)[::-1]
        best_i = order[0]
        best   = densidades[best_i]
        second = densidades[order[1]] if len(order) > 1 else 0.0

        if best < min_mark_percent:
            respostas.append("")      # linha vazia: não desenhar seleção
            continue
        if (best - second) < tie_margin_percent and second >= min_mark_percent * 0.8:
            respostas.append("AMB")
        else:
            respostas.append(letras[best_i])
            bx1, by1, bx2, by2 = cell_boxes[best_i]
            cv2.rectangle(table_vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(table_vis, respostas[-1], (bx1 + 4, by1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return respostas, table_vis

# ------------ Orquestração ------------
def run(input_path, outdir, show_stats=False):
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {input_path}")

    # 1) Detecta grade
    x, y, w, h, v_lines, h_lines, rot_img, rot_bin = detect_grid(img)

    # 2) Visual: bbox
    overlay_bbox = rot_img.copy()
    cv2.rectangle(overlay_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(out("01_table_bbox_overlay.png"), overlay_bbox)

    # 3) Visual: linhas da grade
    grid_overlay = rot_img.copy()
    for xl in v_lines:
        cv2.line(grid_overlay, (x + xl, y), (x + xl, y + h), (0, 255, 0), 1)
    for yl in h_lines:
        cv2.line(grid_overlay, (x, y + yl), (x + w, y + yl), (0, 255, 0), 1)
    cv2.imwrite(out("02_grid_overlay.png"), grid_overlay)

    # 4) Visual: células
    cells_overlay = rot_img[y:y + h, x:x + w].copy()
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x1, x2 = v_lines[j], v_lines[j + 1]
            y1, y2 = h_lines[i], h_lines[i + 1]
            cv2.rectangle(cells_overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(out("03_cells_preview.png"), cells_overlay)

    # 5) Leitura das marcações
    respostas, marks_overlay = read_marks(rot_img, rot_bin, (x, y, w, h), v_lines, h_lines, min_mark_percent=20.0, tie_margin_percent=5.0,
    skip_top_rows=1,   # <- ignora cabeçalho
    max_rows=15        # <- garante 15 questões
    )
    cv2.imwrite(out("04_marks_overlay.png"), marks_overlay)

    # 6) Saídas estruturadas
    letras_validas = ["A", "B", "C", "D", "E"]
    respostas_limpa = [r if r in letras_validas else "" for r in respostas]  # vazio para não marcadas/ambíguas

    # .txt estilo lista python
    with open(out("respostas.txt"), "w", encoding="utf-8") as f:
        f.write(str(respostas_limpa) + "\n")

    # .csv
    with open(out("respostas.csv"), "w", encoding="utf-8") as f:
        f.write("questao,resposta\n")
        for i, r in enumerate(respostas, start=1):
            f.write(f"{i},{r}\n")

    # .json
    with open(out("respostas.json"), "w", encoding="utf-8") as f:
        json.dump({"respostas": respostas, "respostas_limpa": respostas_limpa}, f, ensure_ascii=False, indent=2)

    if show_stats:
        print(f"Table bbox: x={x}, y={y}, w={w}, h={h}")
        print(f"Verticais: {len(v_lines)} | Horizontais: {len(h_lines)}")
        print("Respostas (brutas):", respostas)
        print("Respostas (limpas):", respostas_limpa)
        print(f"Arquivos salvos em: {os.path.abspath(outdir)}")

    return respostas, respostas_limpa


# ------------ CLI ------------
def main():
    ap = argparse.ArgumentParser(description="Detecção de grade + leitura de marcações (A–E).")
    ap.add_argument("--input", required=True, help="Imagem do gabarito (JPEG/PNG)")
    ap.add_argument("--outdir", default="./out", help="Diretório de saída")
    ap.add_argument("--show-stats", action="store_true", help="Imprime estatísticas")
    args = ap.parse_args()

    run(args.input, args.outdir, show_stats=args.show_stats)


if __name__ == "__main__":
    main()
