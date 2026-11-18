"""Segmentación del disco y la copa y medida de los biomarcadores a partir de una fotografía de fondo de ojo.

Reproduce el notebook `Segmentacion_SWIMT.ipynb` del proyecto, con el que se construyó la base de datos:

1. letterbox a 384 x 384 conservando la relación de aspecto (relleno negro);
2. red Swin Transformer de Bazi et al. (2024) con los pesos publicados por los autores
   (`TrainAll_Maghrabi84_50iteration_SWIN.pth.tar`, Hugging Face, espacio BigData-KSU/Glaucoma-Detection);
3. post-procesamiento morfológico (cierre, relleno de huecos, área mínima, circularidad, región más cercana al
   centro y copa dentro del disco);
4. áreas y diámetros reescalados a la imagen original, y grosores I, S, N, T por los cortes vertical y horizontal
   que pasan por el centroide de la copa, con redondeo aritmético (0.5 hacia arriba).

La arquitectura de la red (`RedSwin`) es la del archivo `pipline.py` de los autores, con la única diferencia de que el
backbone se crea sin descargar los pesos de ImageNet, porque todos los pesos vienen del archivo de Bazi et al.
Requiere timm 0.6.13, la versión que usan los autores.
"""
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk

TAMANO = 384
NOMBRE_PESOS = "TrainAll_Maghrabi84_50iteration_SWIN.pth.tar"
URL_PESOS = ("https://huggingface.co/spaces/BigData-KSU/Glaucoma-Detection/resolve/main/"
             "TrainAll_Maghrabi84_50iteration_SWIN.pth.tar")


# ---------------------------------------------------------------------------------------------------- red
def _construir_red():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from timm.models.swin_transformer import SwinTransformer, swin_base_patch4_window12_384_in22k

    # Igual que pipline.py: el backbone devuelve la salida de cada etapa
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        ocultas = []
        for capa in self.layers:
            x = capa(x)
            ocultas.append(x)
        return ocultas

    SwinTransformer.forward_features = forward_features
    SwinTransformer.forward = lambda self, x: self.forward_features(x)

    class BasicBlock(nn.Module):
        # Definido en pipline.py; no se usa en forward, pero sus pesos están en el archivo y se cargan igual
        def __init__(self, channel_num):
            super().__init__()
            self.conv_block1 = nn.Sequential(nn.Conv2d(channel_num, 48, 1, padding=0),
                                             nn.GroupNorm(num_groups=8, num_channels=48), nn.GELU())
            self.conv_block2 = nn.Sequential(nn.Conv2d(48, channel_num, 3, padding=1),
                                             nn.GroupNorm(num_groups=8, num_channels=channel_num), nn.GELU())
            self.relu = nn.GELU()

    class ASPP(nn.Module):
        def __init__(self, image_dim=384, head=1):
            super().__init__()
            self.image_dim = image_dim
            self.Residual2 = BasicBlock(channel_num=head)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.head = head

        def forward(self, x):
            return F.interpolate(x, size=(self.image_dim, self.image_dim), mode="bilinear", align_corners=True)

    class RedSwin(nn.Module):
        """Transformer_Regression de pipline.py (Bazi et al., 2024)."""

        def __init__(self, image_dim=384, num_classes=3):
            super().__init__()
            self.backbone = swin_base_patch4_window12_384_in22k(pretrained=False)
            self.ASPP1 = ASPP(image_dim, head=128)
            self.ASPP2 = ASPP(image_dim, head=128)
            self.Classifier_main = nn.Sequential(nn.Conv2d(128, num_classes, 3, bias=True, padding=1))
            self.Classifier_aux1 = nn.Sequential(nn.Conv2d(128, num_classes, 3, bias=True, padding=1))
            self.conv1 = nn.Sequential(nn.Conv2d(448, 128, kernel_size=(1, 1), padding=1), nn.GELU())
            self.pixelshufler1 = nn.PixelShuffle(2)
            self.pixelshufler2 = nn.PixelShuffle(4)

        def forward(self, x):
            h = self.backbone(x)
            x1 = [h[0].reshape(-1, 48, 48, 256), h[1].reshape(-1, 24, 24, 512), h[2].reshape(-1, 12, 12, 1024)]
            x1 = [t.permute(0, 3, 1, 2) for t in x1]
            x1[1] = self.pixelshufler1(x1[1])
            x1[2] = self.pixelshufler2(x1[2])
            f = self.conv1(torch.cat((x1[0], x1[1], x1[2]), 1))
            return {"seg": self.Classifier_main(self.ASPP1(f)), "seg_aux_1": self.Classifier_aux1(self.ASPP2(f))}

    return RedSwin


class Segmentador:
    """Carga la red una sola vez y segmenta imágenes."""

    def __init__(self, ruta_pesos):
        import torch
        if not os.path.exists(ruta_pesos):
            raise FileNotFoundError(
                f"No se encontraron los pesos de la red en {ruta_pesos}. Descárguelos de {URL_PESOS} "
                "y guárdelos en la carpeta 'pesos' de la aplicación.")
        self.torch = torch
        RedSwin = _construir_red()
        self.red = RedSwin(image_dim=TAMANO, num_classes=3)
        estado = torch.load(ruta_pesos, map_location="cpu", weights_only=False)
        estado = estado["state_dict"] if "state_dict" in estado else estado
        self.red.load_state_dict(estado, strict=True)
        self.red.eval()

    def mascara(self, imagen):
        """Máscara cruda 384 x 384 (0 fondo, 1 disco, 2 copa) y el letterbox usado."""
        lienzo, factor, ox, oy = letterbox(imagen)
        arreglo = np.asarray(lienzo, dtype=np.float32) / 255.0          # ToTensor
        arreglo = (arreglo - 0.5) / 0.5                                  # Normalize(0.5, 0.5)
        x = self.torch.from_numpy(arreglo.transpose(2, 0, 1)).unsqueeze(0)
        with self.torch.no_grad():
            salida = self.red(x)["seg"]
        return salida.argmax(dim=1).squeeze(0).numpy(), factor, ox, oy

    def analizar(self, imagen, ojo):
        """Segmenta una imagen PIL y devuelve (medidas, mascara_limpia, letterbox).

        medidas: dict con las 10 medidas de la base (I, S, N, T, áreas y diámetros en píxeles de la imagen original).
        """
        imagen = imagen.convert("RGB")
        cruda, factor, ox, oy = self.mascara(imagen)
        limpia = postprocesar_mascara(cruda)
        disco, copa = np.uint8(limpia > 0), np.uint8(limpia > 1)
        if disco.sum() == 0:
            raise ValueError("No se encontró un disco óptico confiable en la imagen.")
        if copa.sum() == 0:
            raise ValueError("No se encontró una copa confiable en la imagen.")
        s = 1 / factor
        bio = calcular_biomarcadores(limpia, s, s)
        I, S, N, T = calcular_isnt(disco, copa, ojo, s, s)
        medidas = {"I": I, "S": S, "N": N, "T": T,
                   "disc_area": bio["disc_area"], "cup_area": bio["cup_area"],
                   "Disco_X": bio["disco_x"], "Disco_Y": bio["disco_y"],
                   "Excavacion_X": bio["excavacion_x"], "Excavacion_Y": bio["excavacion_y"]}
        return medidas, limpia, (factor, ox, oy)


# ---------------------------------------------------------------------------------------------------- pasos
def letterbox(imagen, tamano=TAMANO):
    """Escala con el mismo factor en ambos ejes y centra sobre un lienzo negro cuadrado."""
    ancho, alto = imagen.size
    factor = min(tamano / ancho, tamano / alto)
    nuevo = (round(ancho * factor), round(alto * factor))
    lienzo = Image.new("RGB", (tamano, tamano), (0, 0, 0))
    ox, oy = (tamano - nuevo[0]) // 2, (tamano - nuevo[1]) // 2
    lienzo.paste(imagen.resize(nuevo), (ox, oy))
    return lienzo, factor, ox, oy


def postprocesar_mascara(mask_pred, radio_cierre=3, umbral_circularidad=0.5, area_minima_disco=100,
                         area_minima_copa=50):
    """Cierre, relleno de huecos, área mínima, circularidad y región más cercana al centro; copa dentro del disco."""
    alto, ancho = mask_pred.shape
    centro = np.array((alto / 2, ancho / 2))

    def limpiar(binaria, area_minima):
        rellena = ndimage.binary_fill_holes(binary_closing(binaria, disk(radio_cierre)))
        etiquetas = label(rellena)
        if etiquetas.max() == 0:
            return rellena
        regiones = [r for r in regionprops(etiquetas) if r.area >= area_minima]
        validas = [r for r in regiones
                   if r.perimeter > 0 and (4 * np.pi * r.area) / (r.perimeter ** 2) >= umbral_circularidad]
        if not validas:
            return np.zeros_like(rellena, dtype=bool)
        elegida = min(validas, key=lambda r: np.linalg.norm(np.array(r.centroid) - centro))
        return etiquetas == elegida.label

    disco = limpiar(mask_pred > 0, area_minima_disco)
    copa = limpiar(mask_pred > 1, area_minima_copa) & disco
    limpia = np.zeros_like(mask_pred)
    limpia[disco] = 1
    limpia[copa] = 2
    return limpia


def calcular_biomarcadores(mask, sx, sy):
    disco, copa = np.uint8(mask > 0), np.uint8(mask > 1)
    return {
        "disc_area": float(np.sum(disco) * sx * sy),
        "cup_area": float(np.sum(copa) * sx * sy),
        "excavacion_y": float(np.sum(copa, axis=0).max() * sy),
        "excavacion_x": float(np.sum(copa, axis=1).max() * sx),
        "disco_y": float(np.sum(disco, axis=0).max() * sy),
        "disco_x": float(np.sum(disco, axis=1).max() * sx),
    }


def redondeo_medico(x):
    """Redondeo aritmético: 0.5 hacia arriba (round() de Python redondea al par)."""
    return math.floor(x + 0.5)


def calcular_isnt(disco, copa, ojo, sx, sy):
    """Grosores del anillo por los cortes que pasan por el centroide de la copa.

    Misma convención que la base del proyecto: en un ojo DERECHO, N es el lado izquierdo de la imagen y T el derecho;
    en un ojo IZQUIERDO, al revés.
    """
    props = max(regionprops(label(copa)), key=lambda r: r.area)
    cy, cx = (int(v) for v in props.centroid)
    disco_col, copa_col = np.where(disco[:, cx])[0], np.where(copa[:, cx])[0]
    disco_fila, copa_fila = np.where(disco[cy, :])[0], np.where(copa[cy, :])[0]
    S = (copa_col.min() - disco_col.min()) * sy
    I = (disco_col.max() - copa_col.max()) * sy
    izq = (copa_fila.min() - disco_fila.min()) * sx
    der = (disco_fila.max() - copa_fila.max()) * sx
    N, T = (izq, der) if str(ojo).upper().startswith("DER") else (der, izq)
    return tuple(redondeo_medico(v) for v in (I, S, N, T))


def superponer_contornos(imagen, mask, letterbox_info, recorte=True):
    """Imagen original con los contornos del disco (verde) y la copa (amarillo), recortada alrededor del disco."""
    from PIL import ImageDraw
    from skimage.measure import find_contours
    factor, ox, oy = letterbox_info
    imagen = imagen.convert("RGB").copy()
    dibujo = ImageDraw.Draw(imagen)
    grosor = max(2, int(round(imagen.size[0] / 600)))
    for nivel, color in ((0.5, (60, 220, 90)), (1.5, (255, 215, 0))):
        for c in find_contours(mask.astype(float), nivel):
            pts = [((x - ox) / factor, (y - oy) / factor) for y, x in c]
            if len(pts) > 2:
                dibujo.line(pts + [pts[0]], fill=color, width=grosor)
    if recorte:
        filas, cols = np.where(mask > 0)
        if len(filas):
            y0, y1 = (filas.min() - oy) / factor, (filas.max() - oy) / factor
            x0, x1 = (cols.min() - ox) / factor, (cols.max() - ox) / factor
            margen = 1.2 * max(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            caja = (max(0, int(cx - margen)), max(0, int(cy - margen)),
                    min(imagen.size[0], int(cx + margen)), min(imagen.size[1], int(cy + margen)))
            imagen = imagen.crop(caja)
    return imagen
