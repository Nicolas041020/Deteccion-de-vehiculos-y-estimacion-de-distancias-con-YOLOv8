import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailSender:

    DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    NOMBRES_CLASE = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}

    CLASE_MOTO = 3
    GRACIA_FRAMES = 8        # frames que un evento sobrevive sin verse antes de cerrarse
    MIN_FRAMES_EVENTO = 10   # duracion minima para que un evento cuente como real

    def __init__(self, email_origen, password, email_destino):
        self.email_origen = email_origen
        self.password = password
        self.email_destino = email_destino
        self.iniciar_sesion()

    # ── Sesion ────────────────────────────────────────────────────────────────
    def iniciar_sesion(self, fps=30.0):
        """Reinicia los contadores. Se llama al comenzar cada video."""
        self.fps = fps if fps and fps > 0 else 30.0
        self.n_frame = 0
        self.total_frames = 0
        self.frames_con_riesgo = 0
        self.eventos_abiertos = {}
        self.eventos_cerrados = []
        self.inicio_sesion = datetime.now()

    # ── Registro por frame ────────────────────────────────────────────────────
    def registrar_frame(self, pares):
        """Procesa todos los pares evaluados en un frame.

        pares: lista de dicts {'ids', 'clases', 'd_AB', 'nivel'}.
        Debe llamarse una vez por frame, incluso con la lista vacia, para que
        los eventos abiertos envejezcan correctamente.
        """
        self.n_frame += 1
        self.total_frames += 1

        vistos = set()
        for par in pares:
            if not self._es_par_reportable(par):
                continue
            clave = tuple(sorted(par['ids']))
            vistos.add(clave)
            if clave in self.eventos_abiertos:
                self._actualizar(clave, par)
            else:
                self._abrir(clave, par)

        # Los pares que no aparecieron este frame acumulan ausencia
        for clave in list(self.eventos_abiertos):
            if clave in vistos:
                continue
            evento = self.eventos_abiertos[clave]
            evento['frames_ausente'] += 1
            if evento['frames_ausente'] > self.GRACIA_FRAMES:
                self._cerrar(clave)

        # Se cuenta una vez por frame, no una por par
        if vistos:
            self.frames_con_riesgo += 1

    def _es_par_reportable(self, par):
        if par.get('nivel') != 'Riesgo':
            return False
        if None in par['ids']:                      # sin track no se puede agrupar
            return False
        return self.CLASE_MOTO in par['clases']     # el informe es sobre motos

    def _abrir(self, clave, par):
        id_a, id_b = par['ids']
        clase_a, clase_b = par['clases']
        self.eventos_abiertos[clave] = {
            'vehiculos': {id_a: clase_a, id_b: clase_b},
            'frame_inicio': self.n_frame,
            'frame_fin': self.n_frame,
            'd_min': par['d_AB'],
            'frames_vistos': 1,
            'frames_ausente': 0,
            'hora_inicio': datetime.now(),
        }

    def _actualizar(self, clave, par):
        evento = self.eventos_abiertos[clave]
        evento['frame_fin'] = self.n_frame      # ultimo frame realmente visto
        evento['frames_vistos'] += 1
        evento['frames_ausente'] = 0
        if par['d_AB'] < evento['d_min']:
            evento['d_min'] = par['d_AB']

    def _cerrar(self, clave):
        evento = self.eventos_abiertos.pop(clave)
        if evento['frames_vistos'] >= self.MIN_FRAMES_EVENTO:
            self.eventos_cerrados.append(evento)

    def _cerrar_todos(self):
        for clave in list(self.eventos_abiertos):
            self._cerrar(clave)

    # ── Formato ───────────────────────────────────────────────────────────────
    def _formato_dia(self, fecha):
        hora12 = fecha.hour % 12 or 12
        ampm = 'am' if fecha.hour < 12 else 'pm'
        return f"{self.DIAS[fecha.weekday()]} a las {hora12}:{fecha.minute:02d} {ampm}"

    def _duracion(self, evento):
        return (evento['frame_fin'] - evento['frame_inicio'] + 1) / self.fps

    def _etiqueta(self, evento):
        vehiculos = evento['vehiculos']
        motos = [i for i, c in vehiculos.items() if c == self.CLASE_MOTO]
        otros = [i for i, c in vehiculos.items() if c != self.CLASE_MOTO]
        return " <-> ".join(
            f"{self.NOMBRES_CLASE.get(vehiculos[i], 'Vehiculo')} #{i}"
            for i in motos + otros
        )

    # ── Informe ───────────────────────────────────────────────────────────────
    def generar_informe(self):
        ahora = datetime.now()
        duracion_sesion = ahora - self.inicio_sesion

        motos_en_riesgo = set()
        for evento in self.eventos_cerrados:
            motos_en_riesgo.update(
                i for i, c in evento['vehiculos'].items() if c == self.CLASE_MOTO)

        tiempo_riesgo = sum(self._duracion(e) for e in self.eventos_cerrados)
        d_global = min((e['d_min'] for e in self.eventos_cerrados), default=None)
        porcentaje = (self.frames_con_riesgo / self.total_frames * 100) \
            if self.total_frames > 0 else 0
        texto_d_global = f"{d_global:.2f} m" if d_global is not None else "—"

        cuerpo = f"""
        INFORME DE MONITOREO - SISTEMA ADAS
        =====================================
        Fecha: {self._formato_dia(ahora)} ({ahora.strftime('%Y-%m-%d %H:%M:%S')})
        Duración sesión: {str(duracion_sesion).split('.')[0]}

        RESUMEN
        -------
        Total frames procesados:   {self.total_frames}
        Motos distintas en riesgo: {len(motos_en_riesgo)}
        Eventos de riesgo:         {len(self.eventos_cerrados)}
        Tiempo total en riesgo:    {tiempo_riesgo:.1f} s
        Distancia mínima global:   {texto_d_global}
        Frames con riesgo:         {self.frames_con_riesgo} / {self.total_frames} ({porcentaje:.1f}%)

        DETALLE DE EVENTOS
        ------------------
        """

        if not self.eventos_cerrados:
            cuerpo += """
        Sin eventos de riesgo registrados.
            """

        for i, evento in enumerate(self.eventos_cerrados, 1):
            cuerpo += f"""
        Evento {i}: {self._etiqueta(evento)}
          Inicio:    {self._formato_dia(evento['hora_inicio'])}
          Duración:  {self._duracion(evento):.1f} s ({evento['frames_vistos']} frames)
          Dist. mín: {evento['d_min']:.2f} m
            """

        cuerpo += """
        =====================================
        Sistema de Detección Vehicular YOLOv8
        Universidad Piloto de Colombia - 2026
        """
        return cuerpo

    def enviar_informe(self):
        self._cerrar_todos()          # los eventos en curso al terminar el video
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
