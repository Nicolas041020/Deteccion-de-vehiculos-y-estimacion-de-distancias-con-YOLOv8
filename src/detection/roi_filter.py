import numpy as np
import cv2

class RoiFilter:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        TL = (int(self.ancho * 0.35), int(self.alto * 0.30))
        TR = (int(self.ancho * 0.85), int(self.alto * 0.30))
        BR = (int(self.ancho * 0.98), int(self.alto * 0.90))
        BL = (int(self.ancho * 0.02), int(self.alto * 0.90))
        self.contour = np.array([TL,TR,BR,BL], dtype=np.int32).reshape((-1,1,2))

    def filtro(self, detecciones):
        res = []
        clase_interes = [2,3,5,7]

        for dec in detecciones:
            x1,y1,x2,y2 = dec['bbox']
            central = ((x1+x2)/2,y2)
            resultado = cv2.pointPolygonTest(self.contour, central, False)
            if resultado > 0 and dec['clase_id'] in clase_interes:
                res.append(dec)
        return res
    