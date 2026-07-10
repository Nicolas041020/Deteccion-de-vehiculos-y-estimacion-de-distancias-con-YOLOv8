import os 
import random
import cv2
import numpy as np
import math
class DistanceEstimation:
    
    @staticmethod
    def estimateDistance(real_height, pixel_height,focal_length):
        return focal_length * real_height / pixel_height
    
    @staticmethod
    def returnFilename(filename):
        return os.path.splitext(os.path.basename(filename))[0]
    
    @staticmethod
    def getParametersValidation(name):
        
        ruta = os.path.join('dataset','data_object_calib', 'training', 'calib', f'{name}.txt')
        
        with open(ruta,'r') as f:
            lineas = f.readlines()
            for line in lineas:
                if line.startswith('P2:'):
                    partes = line.split()
                    fx = float(partes[1])
                    fy = float(partes[6])
                    cx = float(partes[3])
                    cy = float(partes[7])
                    #print("hOLA ",fx,fy,cx,cy)
                    return fx,fy,cx,cy
                

    
    @staticmethod
    def runValidation(filename, result):
        distancias = []
        name = DistanceEstimation.returnFilename(filename)
        H_real = [1.42,1.88]
        fx,fy, cx, cy = DistanceEstimation.getParametersValidation(name)
        res = DistanceEstimation.obtainPixelHeight(result)
        
        for i,obj in enumerate(res):
            z = DistanceEstimation.estimateDistance(
                #obj['H_mean'],
                H_real[i],
                obj['H_px'],
                fy
            )
            distancias.append({
                'clase_id': obj['clase_id'],
                'bbox': obj['bbox'],
                'distancia_z': round(z, 2)
            })
        
        return distancias
    
    @staticmethod
    def samplear():
        ruta = os.path.join('dataset','data_object_image_2', 'training', 'image_2')
        archivos = os.listdir(ruta)
        random.seed(42)
        imgs_muestra = random.sample(archivos,200)
        #DistanceEstimation.obtenerlabels_param(imgs_muestra)
        return imgs_muestra
    
    @staticmethod
    def obtenerlabels_param(lista):
        result = []
        for img in lista:
            name = DistanceEstimation.returnFilename(img)
            ruta = os.path.join('dataset','data_object_label_2', 'training', 'label_2', f'{name}.txt')

            objetos = []
            with open(ruta,'r') as f:
                for linea in f:
                    partes = linea.split()
                    if partes[0] != 'Car':
                        continue
                    x1 = float(partes[4])
                    y1 = float(partes[5])
                    x2 = float(partes[6])
                    y2 = float(partes[7])
                    altura_real = float(partes[8])
                    z = float(partes[13])

                    objetos.append({
                        'bbox': (x1, y1, x2, y2),
                        'h_real': altura_real,
                        'z_real': z
                    })

            params = DistanceEstimation.getParametersValidation(name)
            result.append({'imagen':img, 'objetos_label': objetos, 'params': params})
        return result




    @staticmethod
    def calcular_iou(boxA, boxB):
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        
        union = areaA + areaB - inter_area
        return inter_area / union if union > 0 else 0

    @staticmethod
    def emparejar_detecciones(detecciones_yolo, objetos_label, umbral_iou=0.5):
        pares = []
        for det in detecciones_yolo:
            mejor_iou = 0
            mejor_match = None
            for obj in objetos_label:
                iou = DistanceEstimation.calcular_iou(det['bbox'], obj['bbox'])
                if iou > mejor_iou:
                    mejor_iou = iou
                    mejor_match = obj
            
            if mejor_iou > umbral_iou:
                pares.append((det, mejor_match))
        
        return pares

    @staticmethod
    def calcular_error(res, modelo_yolo, ruta_imagenes):
        errores = []
        errores_lbl =[]
        z_reales = []
        
        for r in res:
            ruta_img = os.path.join(ruta_imagenes, r['imagen'])
            print(ruta_img)
            imagen = cv2.imread(ruta_img)
            resultado_yolo = modelo_yolo.detectAndParse(imagen)  # detecciones de ESTA imagen
            
            objetos_label = r['objetos_label']
            pares = DistanceEstimation.emparejar_detecciones(resultado_yolo, objetos_label)
            
            for p in pares:
                z_real = p[1]['z_real']
                fx= r['params'][0]
                fy = r['params'][1]
                cx = r['params'][2]
                altura_pixeles = p[0]['bbox'][3] - p[0]['bbox'][1]
                altura_pixeles_lbl = p[1]['bbox'][3] - p[1]['bbox'][1]
                altura_real = p[1]['h_real']
                z_est = DistanceEstimation.estimateDistance(altura_real, altura_pixeles, fy)
                #x1,y1,x2,y2 = p[1]['bbox']
                #u = (x1+x2)/2
                #angulo = math.atan((u-cx)/fx)
                #z_est = z_est/math.cos(angulo)
                z_est_lbl = DistanceEstimation.estimateDistance(altura_real,altura_pixeles_lbl,fy)
                #z_est_lbl = z_est_lbl/math.cos(angulo)
                err = abs(z_real - z_est)
                err_lbl = abs(z_real - z_est_lbl)
                errores_lbl.append(err_lbl)
                errores.append(err)
                z_reales.append(z_real)
        
        return errores, z_reales, errores_lbl
    
    @staticmethod
    def metricas_error(errores):
        arr = np.array(errores)
        mae  = np.mean(arr)
        rmse = np.sqrt(np.mean(arr**2))
        std  = np.std(arr)
        max_err = np.max(arr)
        return mae, rmse, std, max_err

    @staticmethod
    def validacionKITTI(modelo_yolo):
        lote = DistanceEstimation.samplear()
        labels = DistanceEstimation.obtenerlabels_param(lote)
        ruta = os.path.join('dataset','data_object_image_2', 'training', 'image_2')

        errores, z_reales, errores_lbl = DistanceEstimation.calcular_error(labels, modelo_yolo, ruta)
        rangos_data = DistanceEstimation.erroresPorRango(errores, errores_lbl, z_reales)
        mae, rmse, std, max_err = DistanceEstimation.metricas_error(errores)
        maee, rmsee, stdd, maxe = DistanceEstimation.metricas_error(errores_lbl)
        n = len(errores)
        muestras = len(lote)
        print(f"NMuestras: {muestras} N: {n}  MAE: {mae:.4f}  RMSE: {rmse:.4f}  Std: {std:.4f}  Max: {max_err:.4f}")
        print(f"NMuestras: {muestras} N: {n}  MAE: {maee:.4f}  RMSE: {rmsee:.4f}  Std: {stdd:.4f}  Max: {maxe:.4f}")
        return muestras, n, mae, rmse, std, max_err, maee, rmsee, stdd, maxe, rangos_data
    

    @staticmethod
    def erroresPorRango(errores_yolo, errores_kitti, z_reales):
        rangos = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        resultado = []

        for rmin, rmax in rangos:
            idx = [i for i, z in enumerate(z_reales) if rmin <= z < rmax]
            n = len(idx)
            if n == 0:
                resultado.append({
                    'rango': f"{rmin}-{rmax}m", 'n': 0,
                    'mae_yolo': None, 'rmse_yolo': None,
                    'mae_kitti': None, 'rmse_kitti': None,
                })
                continue
            ey = [errores_yolo[i] for i in idx]
            ek = [errores_kitti[i] for i in idx]
            fila = {
                'rango': f"{rmin}-{rmax}m",
                'n': n,
                'mae_yolo':  np.mean(ey),
                'rmse_yolo': np.sqrt(np.mean(np.array(ey) ** 2)),
                'mae_kitti':  np.mean(ek),
                'rmse_kitti': np.sqrt(np.mean(np.array(ek) ** 2)),
            }
            resultado.append(fila)
            print(f"{fila['rango']} | n={n:3d} | "
                  f"MAE_YOLO={fila['mae_yolo']:.2f}m RMSE_YOLO={fila['rmse_yolo']:.2f}m | "
                  f"MAE_KITTI={fila['mae_kitti']:.2f}m RMSE_KITTI={fila['rmse_kitti']:.2f}m")

        return resultado
    
    #USADO EN IMPLEMENTACION DE MODULO
    @staticmethod
    def obtenerParametrosCamara():
        ruta = os.path.join('models', 'calibracion_camara.npz')
        archivo = np.load(ruta)
        fx = float(archivo['fx'])
        fy = float(archivo['fy'])
        cx = float(archivo['cx'])
        cy = float(archivo['cy'])
        return fx, fy, cx, cy
        
    #USADO EN IMPLEMENTACION DE MODULO
    @staticmethod
    def obtenerVehiculoMasCercano(det,fx,fy,cx):
        
        r_temp = 100.0
        obj_min = None
        angulo_min = 0.0
        idx_min = -1

        for i, obj in enumerate(det):
            z = DistanceEstimation.estimateDistance(
                obj['H_mean'], obj['H_px'], fy)
            x1, y1, x2, y2 = obj['bbox']
            u = (x1 + x2) / 2
            angulo = math.atan((u - cx) / fx)
            r = z / math.cos(angulo)
            if r_temp > r:
                r_temp = r
                angulo_min = angulo
                obj_min = obj
                idx_min = i
            
        print("Vehiculo más cercano")
        print(r_temp,obj_min)

        return r_temp, angulo_min, obj_min, idx_min
    
    #USADO EN IMPLEMENTACION DE MODULO
    @staticmethod
    def obtainPixelHeight(detecciones):
        clases_interes = [2,3,5,7]  # car, motorcycle, Bus y Truck
        lista_tamanos =[]
        H_mean = 0.0
        for dec in detecciones:
            clase_id = int(dec['clase_id'])
            if clase_id not in clases_interes:
                continue

            if clase_id == 2:
                H_mean = 1.55
            elif clase_id == 3:
                H_mean = 0.80
            elif clase_id == 5 or clase_id == 7:
                H_mean = 3.0

            x1, y1, x2, y2 = dec['bbox']
            H_px = y2 - y1
            lista_tamanos.append({
                'clase_id': clase_id,
                'H_px': float(H_px),
                'H_mean':float(H_mean),
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'conf': float(dec['conf'])
            })
        
        return lista_tamanos

    @staticmethod
    def obtenerDosVehiculosMasCercanos(det, idx_min, r_min, cx, fx, fy):
        vecinos = []
        for i, obj in enumerate(det):
            if i == idx_min:
                continue
            z = DistanceEstimation.estimateDistance(
                obj['H_mean'], obj['H_px'], fy)
            x1, y1, x2, y2 = obj['bbox']
            u = (x1 + x2) / 2
            angulo = math.atan((u - cx) / fx)
            r = z / math.cos(angulo)
            diff = abs(r - r_min)
            vecinos.append((diff, r, angulo, obj))

        vecinos.sort(key=lambda x: x[0])
        return vecinos[:2]  
    

    @staticmethod
    def calcularDistanciaIntervehicular(r_A, angulo_A, r_B, angulo_B):
        theta = abs(angulo_A - angulo_B)
        d_AB = math.sqrt(
            r_A**2 + r_B**2 - 2 * r_A * r_B * math.cos(theta)
        )
        return d_AB
    
    @staticmethod
    def distanciasIntervehiculares(detecciones):
        det = DistanceEstimation.obtainPixelHeight(detecciones)
        fx,fy,cx,cy = DistanceEstimation.obtenerParametrosCamara()
        r_temp, angulo_min, obj_min, idx_min = DistanceEstimation.obtenerVehiculoMasCercano(det,fx,fy,cx)
        vecinos = DistanceEstimation.obtenerDosVehiculosMasCercanos(det,idx_min,r_temp,cx,fx,fy)
        distancias =[]
        for v in vecinos:
            distancias.append((DistanceEstimation.calcularDistanciaIntervehicular(r_temp,angulo_min,v[1],v[2]),v[3]))

        return distancias





            
            


    

    
