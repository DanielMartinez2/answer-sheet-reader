# processar_pasta.py
# -*- coding: utf-8 -*-

import os
import csv
import cv2
import argparse
import sys
from scan_gabarito_final import run   # importa sua função principal

def processar_pasta(pasta_input, arquivo_saida="resultados_consolidados.csv"):
    # extensões aceitas
    valid_ext = [".jpg", ".jpeg", ".png"]

    # lista de arquivos
    arquivos = sorted([
        f for f in os.listdir(pasta_input)
        if os.path.splitext(f.lower())[1] in valid_ext
    ])

    if not arquivos:
        print("Nenhuma imagem encontrada na pasta!")
        return

    resultados = []

    print("\n=== Iniciando processamento ===\n")

    for nome_arquivo in arquivos:
        caminho = os.path.join(pasta_input, nome_arquivo)
        print(f"Processando: {nome_arquivo} ...")

        try:
            respostas, _ = run(
                caminho,
                outdir=os.path.join("out_lote", nome_arquivo.split('.')[0]),
                show_stats=False
            )
        except Exception as e:
            print(f"❌ Erro ao processar {nome_arquivo}: {e}")
            respostas = ["ERRO"] * 15

        resultados.append([nome_arquivo] + respostas)

    # gerar CSV consolidado
    header = ["arquivo"] + [f"q{i}" for i in range(1, 16)]

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(resultados)

    print("\n=== Concluído ===")
    print(f"Arquivo salvo: {arquivo_saida}")

if __name__ == "__main__":
    

        parser = argparse.ArgumentParser(description="Processar imagens de gabarito em uma pasta.")
        parser.add_argument("-i", "--input", required=True, help="Pasta contendo imagens")
        parser.add_argument("-o", "--output", default="resultados_consolidados.csv", help="Arquivo CSV de saída")
        args = parser.parse_args()

        if not os.path.isdir(args.input):
            print(f"Pasta não encontrada: {args.input}")
            sys.exit(1)

        processar_pasta(args.input, arquivo_saida=args.output)
