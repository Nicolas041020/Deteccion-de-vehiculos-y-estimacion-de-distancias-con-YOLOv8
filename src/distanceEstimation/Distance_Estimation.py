import os 

class DistanceEstimation:
    
    @staticmethod
    def estimateDistance(real_height, pixel_height,focal_length):
        return focal_length * real_height / pixel_height
    
    @staticmethod
    def obtainPixelHeight(result):
        clases_interes = [2, 3]  # car y motorcycle
        lista_tamanos =[]
        for box in result:
            clase_id = int(box.cls[0])
            if clase_id not in clases_interes:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            H_px = y2 - y1
            lista_tamanos.append({
                'clase_id': clase_id,
                'H_px': float(H_px),
                'bbox': (float(x1), float(y1), float(x2), float(y2))
            })
        
        return lista_tamanos


    @staticmethod
    def returnFilename(filename):
        return os.path.splitext(os.path.basename(filename))[0]
    
    @staticmethod
    def getParameters(name):
        ruta = os.path.join('Dataset','data_object_calib', 'training', 'calib', f'{name}.txt')
        with open(ruta,'r') as f:
            lineas = f.readlines()
            for line in lineas:
                if line.startswith('P2:'):
                    partes = line.split()
                    fx = float(partes[1])
                    cx = float(partes[3])
                    cy = float(partes[7])
                    return fx,cx,cy
                

    
    @staticmethod
    def run(filename, result):
        distancias = []
        name = DistanceEstimation.returnFilename(filename)
        H_real = [1.42,1.88]
        fx, cx, cy = DistanceEstimation.getParameters(name)
        res = DistanceEstimation.obtainPixelHeight(result)
        
        for i,obj in enumerate(res):
            z = DistanceEstimation.estimateDistance(
                H_real[i],
                obj['H_px'],
                fx
            )
            distancias.append({
                'clase_id': obj['clase_id'],
                'bbox': obj['bbox'],
                'distancia_z': round(z, 2)
            })
        
        return distancias
    

    
