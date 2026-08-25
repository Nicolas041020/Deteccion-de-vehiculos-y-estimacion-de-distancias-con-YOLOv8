import sys
import os
import time
import cv2
import numpy as np
import psutil
import platform
import torch

# Ir a la raíz del proyecto
ruta_proyecto = os.path.abspath(os.path.dirname(__file__))
os.chdir(ruta_proyecto)
sys.path.insert(0, os.path.join(ruta_proyecto, 'src'))

from detection.yolo_detection import YoloDetection
from distanceEstimation.Distance_Estimation import DistanceEstimation

# GPU monitoring — pynvml opcional
GPU_DISPONIBLE = False
GPU_MODO       = None
gpu_handle     = None
gpu_name       = "N/A"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handle     = pynvml.nvmlDeviceGetHandleByIndex(0)
        GPU_DISPONIBLE = True
        GPU_MODO       = "pynvml"
    except Exception:
        print("pynvml no disponible, GPU no se medirá")

if GPU_DISPONIBLE:
    print(f"GPU detectada ({GPU_MODO}): {gpu_name}")
else:
    print("GPU no disponible — solo se mide CPU/RAM")

print(f"Plataforma : {platform.machine()} — {platform.system()} {platform.release()}")
print(f"Python     : {platform.python_version()}")
print(f"Directorio : {os.getcwd()}")

# ── Cargar modelo y video ──────────────────────────────────────────────────
PLATAFORMA = "Raspberry Pi"   # cambiá a "PC" si corrés en PC

rutaModelo = os.path.join(ruta_proyecto, 'models', 'yolov8n.pt')
rutaVideo  = os.path.join(ruta_proyecto, 'data', 'samples', 'valCorto.mp4')

detector = YoloDetection(rutaModelo)
detector.export(format="ncnn")

cap = cv2.VideoCapture(rutaVideo)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video    = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames a {fps_video:.1f} FPS")

# Warm-up
N_WARMUP = 20
for _ in range(N_WARMUP):
    ret, frame_warmup = cap.read()
    if ret:
        detector.detectAndParse(frame_warmup)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print(f"Warm-up completado ({N_WARMUP} frames)")

# ── Loop de benchmark ──────────────────────────────────────────────────────
tiempos_deteccion = []
tiempos_distancia = []
tiempos_umbral    = []
tiempos_total     = []
uso_cpu           = []
uso_ram_mb        = []
uso_gpu_pct       = []
uso_gpu_mem_mb    = []

proceso = psutil.Process(os.getpid())
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_inicio = time.perf_counter()

    t1 = time.perf_counter()
    detecciones = detector.detectAndParse(frame)
    t2 = time.perf_counter()

    z_ref, obj_min, distancias_vec = None, None, []
    try:
        z_ref, obj_min, distancias_vec = DistanceEstimation.distanciasIntervehiculares(detecciones)
    except Exception:
        pass
    t3 = time.perf_counter()

    try:
        if z_ref is not None:
            DistanceEstimation.clasificacionDeDistancia(z_ref)
        for d_AB, _ in distancias_vec:
            DistanceEstimation.clasificacionDeDistancia(d_AB)
    except Exception:
        pass
    t4 = time.perf_counter()

    tiempos_deteccion.append(t2 - t1)
    tiempos_distancia.append(t3 - t2)
    tiempos_umbral.append(t4 - t3)
    tiempos_total.append(t4 - t_inicio)
    uso_cpu.append(psutil.cpu_percent(interval=None))
    uso_ram_mb.append(proceso.memory_info().rss / 1024**2)

    if GPU_DISPONIBLE:
        if GPU_MODO == "pynvml":
            uso_gpu_pct.append(pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu)
            uso_gpu_mem_mb.append(pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1024**2)
        else:
            uso_gpu_pct.append(torch.cuda.utilization(0))
            uso_gpu_mem_mb.append(torch.cuda.memory_allocated(0) / 1024**2)

cap.release()
print(f"Frames procesados: {len(tiempos_total)}")

# ── Resultados ─────────────────────────────────────────────────────────────
def resumen(nombre, datos_s):
    datos_ms = np.array(datos_s) * 1000
    print(f"  {nombre}")
    print(f"    Media   : {datos_ms.mean():.3f} ms")
    print(f"    Mediana : {np.median(datos_ms):.3f} ms")
    print(f"    P95     : {np.percentile(datos_ms, 95):.3f} ms")
    print(f"    P99     : {np.percentile(datos_ms, 99):.3f} ms")
    print(f"    Mín     : {datos_ms.min():.3f} ms")
    print(f"    Máx     : {datos_ms.max():.3f} ms")

n = len(tiempos_total)
fps_real = 1 / np.mean(tiempos_total)

print(f"\n========== BENCHMARK — {PLATAFORMA} ==========")
print(f"Frames procesados : {n}")
print()
resumen("1. Detección YOLO",            tiempos_deteccion)
print()
resumen("2. Estimación de distancias",   tiempos_distancia)
print()
resumen("3. Clasificación por umbrales", tiempos_umbral)
print()
resumen("Pipeline completo (1+2+3)",     tiempos_total)
print(f"    FPS estimados : {fps_real:.2f}")
print()
print(f"  CPU / RAM")
print(f"    CPU media : {np.mean(uso_cpu):.1f}%")
print(f"    RAM media : {np.mean(uso_ram_mb):.1f} MB")
print(f"    RAM pico  : {np.max(uso_ram_mb):.1f} MB")

if GPU_DISPONIBLE and uso_gpu_pct:
    print()
    print(f"  GPU ({gpu_name})")
    print(f"    Utilización media : {np.mean(uso_gpu_pct):.1f}%")
    print(f"    Utilización pico  : {np.max(uso_gpu_pct):.1f}%")
    print(f"    VRAM media        : {np.mean(uso_gpu_mem_mb):.1f} MB")
    print(f"    VRAM pico         : {np.max(uso_gpu_mem_mb):.1f} MB")
