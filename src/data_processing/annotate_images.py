import cv2
import os
import json
import numpy as np

# --- Configurações ---
ROI_COLOR = (0, 255, 0)
TEMP_ROI_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
INSTRUCTIONS = """
------------------------------------------------------------------
Ferramenta de Anotacao Responsiva

- A janela agora e redimensionavel.
- A imagem se ajustara automaticamente ao tamanho da janela.

- Arraste o mouse para desenhar um retangulo.
- Pressione 'n' para SALVAR e ir para a proxima imagem.
- Pressione 'z' para DESFAZER o ultimo retangulo.
- Pressione 'r' para RESETAR a imagem atual.
- Pressione 'q' para SAIR.
------------------------------------------------------------------
"""

# --- Variáveis Globais ---
drawing = False
rectangles_original = []
current_rect_original = None

def get_display_scale(img_shape, window_size):
    """Calcula a escala para ajustar a imagem à janela."""
    img_h, img_w = img_shape
    win_h, win_w = window_size
    # Calcula a escala baseada na altura para garantir que a imagem inteira seja visível
    scale = win_h / img_h
    return scale

def mouse_callback(event, x, y, flags, param):
    """Manipula eventos do mouse, convertendo coordenadas da tela para a imagem original."""
    global drawing, rectangles_original, current_rect_original
    
    # Obtem o tamanho atual da janela para calcular a escala
    win_h = cv2.getWindowImageRect(param['window_name'])[3]
    scale = get_display_scale((param['img_h'], param['img_w']), (win_h, 0))

    # Converte as coordenadas do mouse (tela) para coordenadas da imagem original
    original_x, original_y = int(x / scale), int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_rect_original = [(original_x, original_y), (original_x, original_y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect_original[1] = (original_x, original_y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_rect_original and abs(current_rect_original[0][0] - current_rect_original[1][0]) > 5:
            rectangles_original.append(tuple(current_rect_original))
        current_rect_original = None

def run_annotation_tool():
    """Função principal da ferramenta de anotação."""
    global rectangles_original, current_rect_original

    image_dir = input("Digite o caminho para o diretorio com as imagens de coluna: ")
    if not os.path.isdir(image_dir):
        print(f"Erro: Diretorio nao encontrado '{image_dir}'")
        return

    output_dir = os.path.join(os.path.dirname(image_dir), "masks")
    annotations_file = os.path.join(os.path.dirname(image_dir), "annotations.json")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(annotations_file, 'r') as f: annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): annotations = {}

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"Nenhuma imagem encontrada em '{image_dir}'.")
        return

    print(INSTRUCTIONS)

    current_image_index = 0
    while current_image_index < len(image_files):
        filename = image_files[current_image_index]
        if filename in annotations:
            print(f"Ja anotado: '{filename}'. Pulando.")
            current_image_index += 1
            continue

        image_path = os.path.join(image_dir, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Erro ao carregar imagem: {image_path}")
            current_image_index += 1
            continue

        rectangles_original = []
        window_name = f"Anotacao - [{current_image_index + 1}/{len(image_files)}] {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Janela redimensionável
        cv2.resizeWindow(window_name, 500, 750) # Altura inicial de 750px

        mouse_params = {
            'window_name': window_name,
            'img_h': original_image.shape[0],
            'img_w': original_image.shape[1]
        }
        cv2.setMouseCallback(window_name, mouse_callback, mouse_params)

        while True:
            # Tenta obter o tamanho da janela, mas com um fallback para evitar crashes
            try:
                win_rect = cv2.getWindowImageRect(window_name)
                win_h, win_w = win_rect[3], win_rect[2]
                # Se o sistema operacional retornar 0 (comum em certas configs de tela), usa um valor padrão
                if win_h <= 0 or win_w <= 0:
                    win_h = 750 # Fallback para a altura padrão
            except cv2.error:
                # A janela foi fechada pelo usuário, encerra o programa de forma segura
                print("Janela fechada. Saindo.")
                with open(annotations_file, 'w') as f: json.dump(annotations, f, indent=4)
                return

            scale = get_display_scale(original_image.shape[:2], (win_h, win_w))
            display_img = cv2.resize(original_image, (int(original_image.shape[1] * scale), win_h))

            # Desenha retângulos salvos
            for rect in rectangles_original:
                pt1 = (int(rect[0][0] * scale), int(rect[0][1] * scale))
                pt2 = (int(rect[1][0] * scale), int(rect[1][1] * scale))
                cv2.rectangle(display_img, pt1, pt2, ROI_COLOR, 2)

            # Desenha retângulo atual
            if drawing and current_rect_original:
                pt1 = (int(current_rect_original[0][0] * scale), int(current_rect_original[0][1] * scale))
                pt2 = (int(current_rect_original[1][0] * scale), int(current_rect_original[1][1] * scale))
                cv2.rectangle(display_img, pt1, pt2, TEMP_ROI_COLOR, 2)

            cv2.putText(display_img, f"Marcacoes: {len(rectangles_original)}", (10, 30), FONT, 1, (255, 255, 0), 2)
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                with open(annotations_file, 'w') as f: json.dump(annotations, f, indent=4)
                print(f"Progresso salvo em: {annotations_file}")
                return
            elif key == ord('n'):
                # Salva a máscara
                mask = np.zeros(original_image.shape[:2], dtype="uint8")
                for r in rectangles_original: cv2.rectangle(mask, r[0], r[1], 255, -1)
                mask_path = os.path.join(output_dir, f"mask_{os.path.splitext(filename)[0]}.png")
                cv2.imwrite(mask_path, mask)
                
                # Adiciona a anotação ao dicionário
                annotations[filename] = {"mask_path": mask_path, "rectangles": rectangles_original}
                
                # Salva o arquivo JSON imediatamente
                with open(annotations_file, 'w') as f:
                    json.dump(annotations, f, indent=4)
                print(f"Anotacao salva para '{filename}'. Progresso salvo no disco.")
                break
            elif key == ord('z'):
                if rectangles_original: 
                    rectangles_original.pop()
                    print("Ultima marcacao desfeita.")
            elif key == ord('r'):
                rectangles_original = []
                print("Marcacoes resetadas.")

        cv2.destroyWindow(window_name)
        current_image_index += 1

    print(f"Anotacao completa! Salvo em: {annotations_file}")

if __name__ == "__main__":
    run_annotation_tool()
