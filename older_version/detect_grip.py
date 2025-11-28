# detect_grid.py
# -*- coding: utf-8 -*-
"""
Detecção robusta da grade (linhas/colunas) do gabarito.

Saídas:
  01_table_bbox_overlay.png  -> caixa da tabela detectada
  02_grid_overlay.png        -> linhas horizontais/verticais detectadas
  03_cells_preview.png       -> prévia com todas as células desenhadas

Uso:
  python detect_grid.py --input Gabarito_original.jpeg --outdir ./out
"""

import os
import cv2
import math
import argparse
import numpy as np


# ---------- Helpers ----------
def angle_from_hough(binary):
    """Estima pequeno ângulo de skew a partir de linhas com Hough."""
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=250)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        # queremos o desvio em relação ao vertical
        deg = (theta - np.pi / 2) * 180.0 / np.pi
        if -45 <= deg <= 45:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
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


# ---------- Pipeline ----------
def process(input_path, outdir, hough_thresh=250, morph_strength=90, close_kernel=3,
            ver_rel=0.6, hor_rel=0.5, vtol_div=300, htol_div=300, show_stats=False):
    os.makedirs(outdir, exist_ok=True)
    bn = lambda name: os.path.join(outdir, name)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {input_path}")
    H, W = img.shape[:2]

    # 1) Binarização adaptativa
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)

    # 2) Estimar skew com Hough nas horizontais
    kernel_h_tmp = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 80), 1))
    hor_tmp = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h_tmp, iterations=1)
    angle = angle_from_hough(hor_tmp)

    rot_img = rotate_image(img, angle)
    rot_th = rotate_image(th, angle)

    # 3) Morfologia para linhas horizontais/verticais (tamanho adaptativo)
    hscale = max(12, W // morph_strength)
    vscale = max(12, H // morph_strength)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))

    hor = cv2.morphologyEx(rot_th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(rot_th, cv2.MORPH_OPEN, kernel_v, iterations=1)

    lines_map = cv2.addWeighted(hor, 0.5, ver, 0.5, 0)
    lines_map = cv2.morphologyEx(
        lines_map,
        cv2.MORPH_CLOSE,
        np.ones((close_kernel, close_kernel), np.uint8),
        iterations=1
    )

    # 4) BBox da tabela
    bbox = find_table_bbox(lines_map)
    overlay1 = rot_img.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay1, (x, y), (x + w, y + h), (0, 255, 0), 3)
        table_roi = rot_img[y:y + h, x:x + w]
        hor_roi = hor[y:y + h, x:x + w]
        ver_roi = ver[y:y + h, x:x + w]
    else:
        # fallback
        x, y, w, h = 0, 0, W, H
        table_roi = rot_img.copy()
        hor_roi = hor.copy()
        ver_roi = ver.copy()

    cv2.imwrite(bn("01_table_bbox_overlay.png"), overlay1)

    # 5) Picos por projeção
    sum_ver = np.sum(ver_roi > 0, axis=0)   # por coluna
    sum_hor = np.sum(hor_roi > 0, axis=1)   # por linha

    v_indices = np.where(sum_ver > (ver_rel * np.max(sum_ver)))[0]  # noqa
    h_indices = np.where(sum_hor > (hor_rel * np.max(sum_hor)))[0]  # noqa

    vtol = max(5, w // vtol_div)
    htol = max(5, h // htol_div)
    v_lines = cluster_positions(v_indices.tolist(), vtol)
    h_lines = cluster_positions(h_indices.tolist(), htol)

    # garantir bordas
    if 0 not in v_lines:
        v_lines = [0] + v_lines
    if (w - 1) not in v_lines:
        v_lines = v_lines + [w - 1]
    if 0 not in h_lines:
        h_lines = [0] + h_lines
    if (h - 1) not in h_lines:
        h_lines = h_lines + [h - 1]

    v_lines = sorted(set(v_lines))
    h_lines = sorted(set(h_lines))

    # 6) Desenhar grade
    grid_overlay = table_roi.copy()
    for xl in v_lines:
        cv2.line(grid_overlay, (xl, 0), (xl, h), (0, 255, 0), 1)
    for yl in h_lines:
        cv2.line(grid_overlay, (0, yl), (w, yl), (0, 255, 0), 1)
    cv2.imwrite(bn("02_grid_overlay.png"), grid_overlay)

    # 7) Desenhar células
    cells_preview = table_roi.copy()
    if len(v_lines) >= 2 and len(h_lines) >= 2:
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x1, x2 = v_lines[j],   v_lines[j + 1]
                y1, y2 = h_lines[i],   h_lines[i + 1]
                cv2.rectangle(cells_preview, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(bn("03_cells_preview.png"), cells_preview)

    if show_stats:
        print(f"Skew (°): {angle:.3f}")
        print(f"Verticais detectadas: {len(v_lines)} → {v_lines[:10]}{' ...' if len(v_lines)>10 else ''}")
        print(f"Horizontais detectadas: {len(h_lines)} → {h_lines[:10]}{' ...' if len(h_lines)>10 else ''}")
        print(f"Saídas em: {os.path.abspath(outdir)}")


def main():
    ap = argparse.ArgumentParser(description="Detecção de grade da folha de gabarito.")
    ap.add_argument("--input", required=True, help="Caminho da imagem (ex.: Gabarito_original.jpeg)")
    ap.add_argument("--outdir", default="./out", help="Diretório de saída")
    ap.add_argument("--show-stats", action="store_true", help="Imprime estatísticas")
    args = ap.parse_args()

    process(
        input_path=args.input,
        outdir=args.outdir,
        show_stats=args.show_stats,
        # ↓ parâmetros ajustáveis caso precise refinar:
        hough_thresh=250,      # threshold das linhas na Hough
        morph_strength=90,     # maior -> kernels menores; menor -> kernels maiores
        close_kernel=3,        # fechamento para colar pequenos gaps
        ver_rel=0.60,          # fração do pico para aceitar vertical
        hor_rel=0.50,          # fração do pico para aceitar horizontal
        vtol_div=300,          # tolerância de cluster vertical: w // vtol_div
        htol_div=300           # tolerância de cluster horizontal: h // htol_div
    )


if __name__ == "__main__":
    main()
