from ultralytics import YOLO

class YoloDetection:
    # NMS: agnostic evita que un mismo vehiculo sobreviva dos veces con clases
    # distintas (car + truck); iou=0.5 (default 0.7) elimina los duplicados de
    # la misma clase que quedaban con solape moderado.
    NMS_IOU = 0.5
    AGNOSTIC_NMS = True

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image,
                             iou=self.NMS_IOU,
                             agnostic_nms=self.AGNOSTIC_NMS)
        return results
    
    def parseToDictionary(self, result):
        detecciones = []
        for box in result[0].boxes:
            bounding = box.xyxy[0].tolist()
            clase = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            # box.id existe solo cuando se corre con tracking; en imagenes sueltas es None
            track_id = int(box.id[0].item()) if box.id is not None else None
            deteccion = {
                'bbox':bounding,
                'clase_id': clase,
                'conf': conf,
                'track_id': track_id
            }
            detecciones.append(deteccion)
        return detecciones
    
    def detectAndParse(self,image):
        results = self.detect(image)
        return self.parseToDictionary(results)

