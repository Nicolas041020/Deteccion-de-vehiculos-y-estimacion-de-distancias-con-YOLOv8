from ultralytics import YOLO

class YoloDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        return results
    
    @staticmethod
    def parseToDictionary(self,result):
        detecciones = []
        for box in result[0].boxes:
            bounding = box.xyxy[0].tolist()
            clase = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            deteccion = {
                'bbox':bounding,
                'clase_id': clase,
                'conf': conf
            }
            detecciones.append(deteccion)
        return detecciones
    
    def detectAndParse(self,image):
        results = self.detect(image)
        return self.parseToDictionary(results)

