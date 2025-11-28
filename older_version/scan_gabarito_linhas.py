# scan_gabarito_linhas.py
# -*- coding: utf-8 -*-
"""
Leitura de gabarito de múltipla escolha (A–E) a partir de imagem,
detectando a TABELA diretamente pela densidade de linhas horizontais
e verticais (sem usar quadrilátero do logo etc).

Uso:
  python scan_gabarito_linhas.py --input Gabarito_1.jpg --outdir ./out --show-stats
"""

import os
import cv2
import json
import argparse
import numpy as np


# ---------- utilitários básicos ----------

def angle_from_hough(binary):
    """Estima pequeno ângulo de skew a partir de linhas horizontais via Hough."""
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=250)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        # relativo ao eixo horizontal
        deg = theta * 180.0 / np.pi
        # linhas horizontais ~ 0 ou 180 graus
        if deg > 90:
            deg -= 180
        if -45 <= deg <= 45:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


### NOVO: detecção de skew pelas LINHAS VERTICAIS
def angle_from_vertical_hough(gray):
    """
    Estima o ângulo de inclinação das LINHAS VERTICAIS usando Hough.
    Retorna o ângulo em graus (positivo -> imagem precisa girar -ângulo).
    """
    # bordas para evidenciar as linhas
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        deg = theta * 180.0 / np.pi
        # verticais: em torno de 90 graus
        if 80.0 < deg < 100.0:
            # quanto falta para chegar em 90°
            angles.append(deg - 90.0)

    return float(np.median(angles)) if angles else 0.0
### FIM NOVO


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
    """
    Detecta a região da tabela usando densidade de linhas horizontais/verticais,
    com filtros adicionais para impedir captar bordas da folha.
    """
    H, W = gray.shape[:2]

    # binarização adaptativa (linhas ficam claras)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )

    # kernels para linhas grossas
    hscale = max(20, W // 55)
    vscale = max(20, H // 55)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))

    hor = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v, iterations=1)

    lines_map = cv2.bitwise_or(hor, ver)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (lines_map > 0).astype("uint8"), connectivity=8
    )

    best_score = -1
    best_bbox = None

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        # ❌ descartar regiões que encostam na borda da folha
        if x < 20 or y < 20 or (x + w) > (W - 20) or (y + h) > (H - 20):
            continue

        # ❌ descartar o topo — tabela nunca está antes de 40% da imagem
        if y < 0.35 * H:
            continue

        # ❌ descartar caixas desproporcionais à tabela real
        aspect_ratio = w / float(h)
        if not (0.8 <= aspect_ratio <= 2.2):
            continue

        # ❌ descartar regiões pequenas ou gigantes
        if w < 0.3 * W or w > 0.9 * W:
            continue
        if h < 0.15 * H or h > 0.6 * H:
            continue

        # intensidade de linhas
        hor_roi = hor[y:y + h, x:x + w]
        ver_roi = ver[y:y + h, x:x + w]
        sum_hor = np.sum(hor_roi > 0, axis=1)
        sum_ver = np.sum(ver_roi > 0, axis=0)

        if np.max(sum_hor) == 0 or np.max(sum_ver) == 0:
            continue

        h_thresh = 0.45 * np.max(sum_hor)
        v_thresh = 0.45 * np.max(sum_ver)
        h_count = int(np.sum(sum_hor > h_thresh))
        v_count = int(np.sum(sum_ver > v_thresh))

        score = h_count * v_count

        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    # fallback: caso tudo dê errado, tenta cortar metade inferior
    if best_bbox is None:
        best_bbox = (int(W * 0.15), int(H * 0.40), int(W * 0.70), int(H * 0.45))

    # --- AJUSTE: expandir um pouco o ROI para garantir que não corte a 1ª linha ---
    x, y, w, h = best_bbox
    pad_x = 10
    pad_y = 100
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + 2 * pad_y)
    best_bbox = (x, y, w, h)

    return best_bbox, th


# ---------- detectar cantos da tabela ----------

def order_points(pts):
    """
    Ordena 4 pontos no sentido:
    [top-left, top-right, bottom-right, bottom-left]
    para facilitar o cálculo da homografia.
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def find_table_corners_in_roi(tbl_gray):
    """
    Dentro do recorte da tabela (já aproximado pelo bbox),
    encontra o quadrilátero externo da TABELA usando contornos
    nas bordas (Canny + maior contorno com 4 vértices).
    Retorna um array (4,2) com os cantos em float32 ou None.
    """
    h, w = tbl_gray.shape[:2]

    # bordas básicas
    edges = cv2.Canny(tbl_gray, 50, 150, apertureSize=3)
    # engrossa um pouco as bordas
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # maior primeiro
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad = None
    best_area = 0.0
    min_rel_area = 0.05  # tabela pode ser bem menor que o ROI

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_rel_area * h * w:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            ar = ww / float(hh) if hh > 0 else 0.0
            # aspecto aproximado de uma tabela reta (descarta contornos muito verticais/estreitos)
            if 0.4 <= ar <= 3.0 and area > best_area:
                best_area = area
                best_quad = approx.reshape(4, 2)

    if best_quad is not None:
        return order_points(best_quad)

    # fallback: usa o maior contorno, mesmo sem 4 vértices
    cnt = contours[0]
    x, y, ww, hh = cv2.boundingRect(cnt)
    corners = np.array(
        [[x, y],
         [x + ww, y],
         [x + ww, y + hh],
         [x, y + hh]],
        dtype="float32"
    )
    return order_points(corners)


# ---------- detectar grade DENTRO da região da tabela ----------

def detect_grid_in_table(img_bgr, gray, th, table_bbox):
    """
    Acha as linhas da grade dentro da região da tabela.

    Fluxo:
      1) recorta o ROI aproximado da tabela (table_bbox)
      2) corrige o skew VERTICAL do ROI (novidade)
      3) dentro desse ROI, detecta o QUADRILÁTERO externo da tabela
         (4 cantos) com Canny + maior contorno
      4) aplica cv2.getPerspectiveTransform + warpPerspective
         para retificar a tabela (corrigir perspectiva)
      5) na tabela retificada, detecta linhas verticais e horizontais
         via morfologia + projeções.
    """
    tx, ty, tw, th_h = table_bbox

    # recorte aproximado da tabela (ROI)
    tbl_gray = gray[ty:ty + th_h, tx:tx + tw]
    tbl_th   = th[ty:ty + th_h, tx:tx + tw]
    tbl_bgr  = img_bgr[ty:ty + th_h, tx:tx + tw]

    # opcional: corta parte direita (onde pode ter texto fora da tabela)
    cut = int(tw * 0.82)
    tbl_gray = tbl_gray[:, :cut]
    tbl_th   = tbl_th[:,  :cut]
    tbl_bgr  = tbl_bgr[:, :cut]

    ### NOVO: corrigir inclinação VERTICAL do ROI da tabela
    skew_v = angle_from_vertical_hough(tbl_gray)
    if abs(skew_v) > 0.05:
        # giramos no sentido contrário para "desinclinar"
        tbl_gray = rotate_image(tbl_gray, -skew_v)
        tbl_th   = rotate_image(tbl_th,   -skew_v)
        tbl_bgr  = rotate_image(tbl_bgr,  -skew_v)
    ### FIM NOVO

    # 1) tenta encontrar os 4 cantos reais da tabela dentro do ROI
    corners = find_table_corners_in_roi(tbl_gray)

    if corners is not None:
        # ---- HOMOGRAFIA: corrige perspectiva da tabela ----
        (tl, tr, br, bl) = corners

        # largura/altura alvo baseadas nas distâncias dos lados
        widthA  = np.linalg.norm(br - bl)
        widthB  = np.linalg.norm(tr - tl)
        maxW    = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH    = int(max(heightA, heightB))

        dst = np.array(
            [[0, 0],
             [maxW - 1, 0],
             [maxW - 1, maxH - 1],
             [0, maxH - 1]], dtype="float32"
        )

        M = cv2.getPerspectiveTransform(corners, dst)
        table_bgr = cv2.warpPerspective(tbl_bgr, M, (maxW, maxH))
        table_th  = cv2.warpPerspective(tbl_th,  M, (maxW, maxH))
    else:
        # ---- fallback: usa apenas rotação leve (versão antiga) ----
        kernel_h_tmp = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(10, tw // 60), 1)
        )
        hor_tmp = cv2.morphologyEx(tbl_th, cv2.MORPH_OPEN, kernel_h_tmp,
                                   iterations=1)
        angle = angle_from_hough(hor_tmp)

        table_bgr = rotate_image(tbl_bgr, angle)
        table_th  = rotate_image(tbl_th,  angle)

    # A partir daqui, trabalhamos SEMPRE na tabela "reta"
    h2, w2 = table_th.shape[:2]

    # morfologia para pegar linhas bem definidas
    hscale = max(12, w2 // 80)
    vscale = max(12, h2 // 80)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (hscale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vscale))
    hor = cv2.morphologyEx(table_th, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(table_th, cv2.MORPH_OPEN, kernel_v, iterations=1)

    # projeções para achar posições de linhas
    sum_ver = np.sum(ver > 0, axis=0)
    sum_hor = np.sum(hor > 0, axis=1)

    v_indices = np.where(sum_ver > (0.60 * np.max(sum_ver)))[0].tolist()
    h_indices = np.where(sum_hor > (0.50 * np.max(sum_hor)))[0].tolist()

    vtol = max(5, w2 // 300)
    htol = max(5, h2 // 300)
    v_lines = cluster_positions(v_indices, vtol)
    h_lines = cluster_positions(h_indices, htol)

    # --- FILTRAGEM DE LINHAS HORIZONTAIS (remover ruído, manter bloco da tabela) ---

    # 1) remove linhas muito próximas (normalmente texto/ruído)
    h_filtered = []
    for i in range(len(h_lines)):
        if i == 0:
            h_filtered.append(h_lines[i])
        else:
            if (h_lines[i] - h_filtered[-1]) > 10:  # mínimo de 10 px entre linhas reais
                h_filtered.append(h_lines[i])
    h_lines = h_filtered

    # 2) pega o maior bloco com espaçamento mais regular (deve conter as 15 questões)
    if len(h_lines) >= 8:
        diffs = np.diff(h_lines).astype(float)
        med_gap = np.median(diffs)
        if med_gap > 0:
            lower = 0.6 * med_gap
            upper = 1.4 * med_gap
            good = (diffs >= lower) & (diffs <= upper)

            best_start = 0
            best_end = 0
            cur_start = None

            for i, ok in enumerate(good):
                if ok:
                    if cur_start is None:
                        cur_start = i
                    cur_end = i + 1
                    if (cur_end - cur_start) > (best_end - best_start):
                        best_start, best_end = cur_start, cur_end
                else:
                    cur_start = None

            block_lines = h_lines[best_start:best_end + 1]

            # Queremos pelo menos 16 linhas (15 linhas de questões + rodapé)
            if len(block_lines) >= 16:
                h_lines = block_lines[:16]
            else:
                h_lines = block_lines

    # organizar e remover duplicatas só por segurança
    v_lines = sorted(set(v_lines))
    h_lines = sorted(set(h_lines))

    # bbox dentro da imagem retificada da tabela (começa em 0,0)
    return 0, 0, w2, h2, v_lines, h_lines, table_bgr, table_th
# ---------- leitura das marcações ----------

def read_marks(rot_img, rot_bin, bbox, v_lines, h_lines,
               min_mark_percent=10.0,  # limiar base um pouco mais baixo
               tie_margin_percent=4.0,
               skip_top_rows=1, max_rows=15):
    """
    Lê marcações A–E a partir da grade detectada.

    Reconhece:
      - X
      - bolinha (círculo aberto ou quase aberto)
      - preenchimento total

    Usa dilatação mais forte, margem interna menor e limiar adaptativo por linha.
    """

    x, y, w, h = bbox
    table_gray = cv2.cvtColor(rot_img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    table_vis  = rot_img[y:y + h, x:x + w].copy()

    letras = ["A", "B", "C", "D", "E"]
    respostas = []

    widths = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines) - 1)]
    if not widths:
        return [], table_vis

    max_width = max(widths)
    min_width = max(10, int(0.2 * max_width))

    # Candidatas a colunas largas
    w_tbl = float(v_lines[-1] - v_lines[0])
    candidates = []
    for i, wcol in enumerate(widths):
        if wcol >= min_width:
            cx = 0.5 * (v_lines[i] + v_lines[i + 1])  # centro da coluna
            candidates.append((i, cx))

    # Remove o que está muito à esquerda (coluna dos números) e muito à direita (margem)
    # 0.30 * w_tbl funciona bem para tirar a coluna "01, 02, 03..."
    filtered = [(i, cx) for (i, cx) in candidates
                if cx > 0.30 * w_tbl and cx < 0.98 * w_tbl]

    if len(filtered) >= 5:
        # ordena da esquerda para a direita e pega as 5 primeiras = A..E
        filtered.sort(key=lambda p: p[1])
        alt_col_indices = [i for (i, _) in filtered[:5]]
    elif len(candidates) >= 5:
        # fallback: usa todas as largas, ainda baseado na posição
        candidates.sort(key=lambda p: p[1])
        alt_col_indices = [i for (i, _) in candidates[:5]]
    else:
        # fallback bem simples, se tudo der errado
        alt_col_indices = list(range(1, min(len(v_lines) - 1, 6)))


    # linhas da tabela (pula cabeçalho)
    start_i = min(skip_top_rows, len(h_lines) - 2)
    end_i   = min(start_i + max_rows, len(h_lines) - 1)

    h_t, w_t = table_gray.shape

    for i in range(start_i, end_i):
        y1, y2 = h_lines[i], h_lines[i + 1]
        densidades, cell_boxes = [], []

        for k, col_idx in enumerate(alt_col_indices):
            x1, x2 = v_lines[col_idx], v_lines[col_idx + 1]

            # pequeno padding para não cortar marcações encostadas na grade
            pad = 2
            y1p = max(0, y1 - pad)
            y2p = min(h_t, y2 + pad)
            x1p = max(0, x1 - pad)
            x2p = min(w_t, x2 + pad)

            cell = table_gray[y1p:y2p, x1p:x2p]

            debug_colors = [
                (0, 255, 0),    # A
                (255, 0, 0),    # B
                (0, 0, 255),    # C
                (0, 255, 255),  # D
                (255, 0, 255),  # E
            ]
            color_debug = debug_colors[k] if k < len(debug_colors) else (0, 255, 0)

            # 1) suaviza um pouco para tirar ruído
            cell_blur = cv2.GaussianBlur(cell, (3, 3), 0)

            # 2) binarização com Otsu + invertida (tinta vira 255)
            _, bin_cell = cv2.threshold(
                cell_blur, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 3) só o miolo da célula (evitar pegar as linhas da grade)
            margin = max(2, min(bin_cell.shape[0], bin_cell.shape[1]) // 10)
            inner = bin_cell[margin:-margin or None, margin:-margin or None]

            if inner.size == 0:
                densidades.append(0.0)
                cell_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(table_vis, (x1, y1), (x2, y2), color_debug, 1)
                continue

            # 4) dilatamos um pouco mais para engordar X, círculos e diagonais finas
            kernel = np.ones((3, 3), np.uint8)
            inner_dilated = cv2.dilate(inner, kernel, iterations=3)

            # 5) percentual de tinta no miolo
            ink_percent = 100.0 * np.sum(inner_dilated > 0) / inner_dilated.size

            densidades.append(ink_percent)
            cell_boxes.append((x1, y1, x2, y2))

            # retângulo de debug da célula
            cv2.rectangle(table_vis, (x1, y1), (x2, y2), color_debug, 1)

        # --- decide a resposta da linha ---
        if not densidades:
            respostas.append("")
            continue

        order = np.argsort(densidades)[::-1]
        best_i = order[0]
        best   = densidades[best_i]
        second = densidades[order[1]] if len(order) > 1 else 0.0
        row_max = best

        # limiar ADAPTATIVO para esta linha
        base_min = min_mark_percent
        if row_max < base_min:
            # quando a linha toda tem pouca tinta, aceita ~50% do máximo
            eff_min_mark = 0.5 * row_max
        else:
            eff_min_mark = base_min

        # nada marcado
        if best < eff_min_mark:
            respostas.append("")
            continue

        # empate (duas alternativas bem marcadas)
        if (best - second) < tie_margin_percent and second >= eff_min_mark * 0.8:
            respostas.append("AMB")
        else:
            respostas.append(letras[best_i])
            bx1, by1, bx2, by2 = cell_boxes[best_i]
            cv2.rectangle(table_vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(table_vis, respostas[-1], (bx1 + 4, by1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return respostas, table_vis


# ---------- pipeline completo ----------

def run(input_path, outdir, show_stats=False):
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) encontrar ROI da TABELA pela densidade de linhas
    table_bbox, th = find_table_roi_by_line_density(gray)
    tx, ty, tw, th_h = table_bbox

    debug_img = img.copy()
    # corrigido: usar ty na coordenada y do canto inferior
    cv2.rectangle(debug_img, (tx, ty), (tx + tw, ty + th_h), (0, 0, 255), 3)
    cv2.imwrite(out("00_table_roi_density.png"), debug_img)

    # 2) detectar grade DENTRO da tabela
    x, y, w, h, v_lines, h_lines, rot_tbl_bgr, rot_tbl_th = detect_grid_in_table(
        img, gray, th, table_bbox
    )

    # 3) overlays de debug
    overlay_bbox = rot_tbl_bgr.copy()
    cv2.rectangle(overlay_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(out("01_table_bbox_overlay.png"), overlay_bbox)

    grid_overlay = rot_tbl_bgr.copy()
    for xl in v_lines:
        cv2.line(grid_overlay, (xl, 0), (xl, h), (0, 255, 0), 1)
    for yl in h_lines:
        cv2.line(grid_overlay, (0, yl), (w, yl), (0, 255, 0), 1)

    cv2.imwrite(out("02_grid_overlay.png"), grid_overlay)

    cells_overlay = rot_tbl_bgr[y:y + h, x:x + w].copy()
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x1, x2 = v_lines[j], v_lines[j + 1]
            y1, y2 = h_lines[i], h_lines[i + 1]
            cv2.rectangle(cells_overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite(out("03_cells_preview.png"), cells_overlay)

    # 4) leitura das marcações – agora já retorna só as 15 questões
    respostas, marks_overlay = read_marks(
        rot_tbl_bgr, rot_tbl_th, (x, y, w, h), v_lines, h_lines,
        min_mark_percent=10.0,      # mais sensível para marcações fracas / nas bordas
        tie_margin_percent=4.0,
        skip_top_rows=0,
        max_rows=15
    )
    cv2.imwrite(out("04_marks_overlay.png"), marks_overlay)

    respostas_questoes = respostas

    letras_validas = ["A", "B", "C", "D", "E"]
    respostas_limpa = [r if r in letras_validas else "" for r in respostas_questoes]

    # TXT simples
    with open(out("respostas.txt"), "w", encoding="utf-8") as f:
        f.write(str(respostas_limpa) + "\n")

    # CSV (questão 1..15)
    with open(out("respostas.csv"), "w", encoding="utf-8") as f:
        f.write("questao,resposta\n")
        for i, r in enumerate(respostas_questoes, start=1):
            f.write(f"{i},{r}\n")

    # JSON com um pouco mais de info
    with open(out("respostas.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "respostas": respostas_questoes,
                "respostas_limpa": respostas_limpa,
                "table_bbox_density": {
                    "x": int(tx), "y": int(ty), "w": int(tw), "h": int(th_h)
                },
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    if show_stats:
        print(f"ROI tabela (densidade): x={tx}, y={ty}, w={tw}, h={th_h}")
        print(f"Verticais: {len(v_lines)} | Horizontais: {len(h_lines)}")
        print("Respostas (1..15):", respostas_questoes)
        print("Respostas limpas:", respostas_limpa)
        print(f"Arquivos salvos em: {os.path.abspath(outdir)}")

    return respostas_questoes, respostas_limpa


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Leitura de gabarito com detecção da tabela por densidade de linhas."
    )
    ap.add_argument("--input", required=True, help="Imagem do gabarito (JPG/PNG)")
    ap.add_argument("--outdir", default="./out", help="Diretório de saída")
    ap.add_argument("--show-stats", action="store_true", help="Imprime estatísticas")
    args = ap.parse_args()

    run(args.input, args.outdir, show_stats=args.show_stats)


if __name__ == "__main__":
    main()
