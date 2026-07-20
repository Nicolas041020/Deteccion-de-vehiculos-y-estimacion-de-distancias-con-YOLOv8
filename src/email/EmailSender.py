import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailSender:
    
    def __init__(self, email_origen, password, email_destino):
        self.email_origen = email_origen
        self.password = password
        self.email_destino = email_destino
        
        # Contadores de sesión
        self.total_frames = 0
        self.frames_riesgo_alto = 0
        self.eventos_riesgo = []  # lista de eventos
        self.inicio_sesion = datetime.now()
    
    def registrar_frame(self, d_AB, clase_ref, clase_vecino):
        self.total_frames += 1
        if d_AB is not None and d_AB < 5:
            self.frames_riesgo_alto += 1
            self.eventos_riesgo.append({
                'timestamp': datetime.now(),
                'd_AB': d_AB,
                'clase_ref': clase_ref,
                'clase_vecino': clase_vecino
            })
    
    def generar_informe(self):
        ahora = datetime.now()
        duracion = ahora - self.inicio_sesion
        porcentaje = (self.frames_riesgo_alto / 
                      self.total_frames * 100) \
                      if self.total_frames > 0 else 0
        
        cuerpo = f"""
        INFORME DE MONITOREO - SISTEMA ADAS
        =====================================
        Fecha: {ahora.strftime('%Y-%m-%d %H:%M:%S')}
        Duración sesión: {str(duracion).split('.')[0]}
        
        RESUMEN
        -------
        Total frames procesados: {self.total_frames}
        Eventos de riesgo alto:  {self.frames_riesgo_alto}
        Porcentaje en riesgo:    {porcentaje:.1f}%
        
        DETALLE DE EVENTOS
        ------------------
        """

        for i, e in enumerate(self.eventos_riesgo, 1):
            vehiculo_1 = "Car" if e['clase_ref'] == 2 else "motorcycle" if e['clase_ref'] == 3 else "Bus/Truck"
            vehiculo_2 = "Car" if e['clase_vecino'] == 2 else "motorcycle" if e['clase_vecino'] == 3 else "Bus/Truck" 

            cuerpo += f"""
        Evento {i}:
          Hora:      {e['timestamp'].strftime('%H:%M:%S')}
          Distancia: {e['d_AB']:.2f} m
          Vehículos: {vehiculo_1} → {vehiculo_2}
            """
        
        cuerpo += """
        =====================================
        Sistema de Detección Vehicular YOLOv8
        Universidad Piloto de Colombia - 2026
        """
        return cuerpo
    
    def enviar_informe(self):
        if self.total_frames == 0:
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.email_origen
        msg['To'] = self.email_destino
        msg['Subject'] = f"Informe ADAS - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        cuerpo = self.generar_informe()
        msg.attach(MIMEText(cuerpo, 'plain'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(self.email_origen, self.password)
            smtp.send_message(msg)