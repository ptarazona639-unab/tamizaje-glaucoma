"""Aplicación de escritorio (Tkinter) para tamizaje de glaucoma con biomarcadores de la segmentación.

Uso:  python app.py
Herramienta de apoyo a la investigación. No reemplaza el criterio del oftalmólogo ni es un dispositivo de diagnóstico.
"""
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
import urllib.request
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from tamizaje_glaucoma.modelo import Modelo
from tamizaje_glaucoma.variables import MEDIDAS, NOMBRES_MEDIDAS, calcular_variables, tabla_desde_medidas

CARPETA = os.path.dirname(os.path.abspath(__file__))
MODELO_POR_DEFECTO = os.path.join(CARPETA, "modelos", "modelo_provisional.joblib")
CARPETA_PESOS = os.path.join(CARPETA, "pesos")
NOMBRE_PESOS = "TrainAll_Maghrabi84_50iteration_SWIN.pth.tar"
URL_PESOS = ("https://huggingface.co/spaces/BigData-KSU/Glaucoma-Detection/resolve/main/" + NOMBRE_PESOS)
LADO_VISTA = 430
AVISO = ("Herramienta de apoyo a la investigación. No reemplaza el criterio del oftalmólogo "
         "ni es un dispositivo de diagnóstico.")
# Ojo derecho con glaucoma del conjunto de desarrollo (imagen 390.jpg de ORIGA), para probar la aplicación.
# Como el modelo provisional se entrenó con el conjunto de desarrollo, este ejemplo no mide su desempeño.
EJEMPLO = {"ojo": "DERECHO", "I": 80, "S": 56, "N": 40, "T": 64, "disc_area": 144256, "cup_area": 67264,
           "Disco_X": 408, "Disco_Y": 440, "Excavacion_X": 304, "Excavacion_Y": 304}

COLOR_POSITIVO = "#b3261e"
COLOR_NEGATIVO = "#1e6b35"
COLOR_SUAVE = "#5f6368"


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tamizaje de glaucoma · Semillero SEIA (UNAB)")
        self.geometry("1040x760")
        self.minsize(980, 720)
        self.modelo = None
        self.resultados_lote = None
        self.segmentador = None
        self.ruta_pesos = os.path.join(CARPETA_PESOS, NOMBRE_PESOS)
        self.imagen = None
        self.ruta_imagen = None
        self.foto_tk = None
        self.medidas_foto = None
        self.cola = queue.Queue()
        self.ocupado = False
        self._estilos()
        self._construir()
        if os.path.exists(MODELO_POR_DEFECTO):
            self._cargar_modelo(MODELO_POR_DEFECTO)
        else:
            self._estado("No hay modelo cargado. Use «Cargar modelo…».")

    # ------------------------------------------------------------------ interfaz
    def _estilos(self):
        estilo = ttk.Style(self)
        if "clam" in estilo.theme_names():
            estilo.theme_use("clam")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 15, "bold"))
        estilo.configure("Aviso.TLabel", foreground=COLOR_SUAVE, font=("Segoe UI", 9))
        estilo.configure("Seccion.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        estilo.configure("Accion.TButton", font=("Segoe UI", 10, "bold"))

    def _construir(self):
        cabecera = ttk.Frame(self, padding=(16, 12, 16, 4))
        cabecera.pack(fill="x")
        ttk.Label(cabecera, text="Tamizaje de glaucoma con biomarcadores del disco óptico",
                  style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(cabecera, text=AVISO, style="Aviso.TLabel").pack(anchor="w", pady=(2, 0))

        marco_modelo = ttk.LabelFrame(self, text="Modelo", padding=10, style="Seccion.TLabelframe")
        marco_modelo.pack(fill="x", padx=16, pady=8)
        self.txt_modelo = tk.StringVar(value="Sin modelo")
        self.lbl_modelo = ttk.Label(marco_modelo, textvariable=self.txt_modelo, justify="left", wraplength=760)
        self.lbl_modelo.pack(side="left", fill="x", expand=True)
        ttk.Button(marco_modelo, text="Cargar modelo…", command=self._elegir_modelo).pack(side="right", anchor="n")
        ttk.Button(marco_modelo, text="Ver detalle", command=self._detalle_modelo).pack(side="right", anchor="n", padx=6)

        pestanas = ttk.Notebook(self)
        pestanas.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.pestanas = pestanas
        pestanas.add(self._pestana_foto(pestanas), text="  Fotografía  ")
        self.pestana_manual = self._pestana_un_ojo(pestanas)
        pestanas.add(self.pestana_manual, text="  Medidas manuales  ")
        pestanas.add(self._pestana_archivo(pestanas), text="  Archivo con varios ojos  ")

        self.txt_estado = tk.StringVar()
        ttk.Label(self, textvariable=self.txt_estado, relief="sunken", anchor="w",
                  padding=(8, 3)).pack(fill="x", side="bottom")

    def _pestana_foto(self, padre):
        marco = ttk.Frame(padre, padding=12)
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=1)
        marco.rowconfigure(0, weight=1)

        izq = ttk.LabelFrame(marco, text="Fotografía de fondo de ojo", padding=10, style="Seccion.TLabelframe")
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        controles = ttk.Frame(izq)
        controles.pack(fill="x")
        ttk.Button(controles, text="Abrir fotografía…", command=self._abrir_foto).pack(side="left")
        ttk.Label(controles, text="   Ojo:").pack(side="left")
        self.ojo_foto = tk.StringVar(value="")
        ttk.Radiobutton(controles, text="Derecho", variable=self.ojo_foto, value="DERECHO").pack(side="left", padx=4)
        ttk.Radiobutton(controles, text="Izquierdo", variable=self.ojo_foto, value="IZQUIERDO").pack(side="left")
        self.btn_analizar = ttk.Button(controles, text="Analizar", style="Accion.TButton", command=self._analizar_foto)
        self.btn_analizar.pack(side="right")
        self.vista = tk.Label(izq, bg="#111111", fg="#bbbbbb", width=LADO_VISTA, height=LADO_VISTA,
                              text="Abra una fotografía centrada en el disco óptico\n(como las de ORIGA)")
        self.vista.pack(fill="both", expand=True, pady=(10, 4))
        self.txt_vista = tk.StringVar()
        ttk.Label(izq, textvariable=self.txt_vista, style="Aviso.TLabel", wraplength=440,
                  justify="left").pack(anchor="w")

        der = ttk.LabelFrame(marco, text="Resultado", padding=10, style="Seccion.TLabelframe")
        der.grid(row=0, column=1, sticky="nsew")
        self.txt_resultado_foto = tk.StringVar(value="Abra una fotografía, elija el ojo y pulse «Analizar».")
        self.lbl_resultado_foto = tk.Label(der, textvariable=self.txt_resultado_foto, font=("Segoe UI", 16, "bold"),
                                           fg=COLOR_SUAVE, anchor="w", justify="left", wraplength=420)
        self.lbl_resultado_foto.pack(fill="x")
        self.txt_probabilidad_foto = tk.StringVar()
        ttk.Label(der, textvariable=self.txt_probabilidad_foto, font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 8))
        self.tabla_foto = ttk.Treeview(der, columns=("valor",), show="tree headings", height=14)
        self.tabla_foto.heading("#0", text="Medida o biomarcador")
        self.tabla_foto.heading("valor", text="Valor")
        self.tabla_foto.column("#0", width=270)
        self.tabla_foto.column("valor", width=110, anchor="e")
        self.tabla_foto.pack(fill="both", expand=True)
        self.btn_pasar = ttk.Button(der, text="Pasar las medidas al formulario manual", state="disabled",
                                    command=self._pasar_a_manual)
        self.btn_pasar.pack(anchor="w", pady=(8, 0))
        return marco

    def _pestana_un_ojo(self, padre):
        marco = ttk.Frame(padre, padding=12)
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=1)

        entradas = ttk.LabelFrame(marco, text="Medidas de la segmentación (píxeles de la imagen original)",
                                  padding=10, style="Seccion.TLabelframe")
        entradas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.ojo = tk.StringVar(value="DERECHO")
        fila_ojo = ttk.Frame(entradas)
        fila_ojo.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(fila_ojo, text="Ojo:").pack(side="left")
        ttk.Radiobutton(fila_ojo, text="Derecho", variable=self.ojo, value="DERECHO").pack(side="left", padx=6)
        ttk.Radiobutton(fila_ojo, text="Izquierdo", variable=self.ojo, value="IZQUIERDO").pack(side="left")
        self.campos = {}
        for i, (clave, etiqueta) in enumerate(MEDIDAS, start=1):
            ttk.Label(entradas, text=etiqueta).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            ttk.Entry(entradas, textvariable=var, width=12, justify="right").grid(row=i, column=1, sticky="e", pady=2)
            self.campos[clave] = var
        botones = ttk.Frame(entradas)
        botones.grid(row=len(MEDIDAS) + 1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(botones, text="Evaluar", style="Accion.TButton", command=self._evaluar_un_ojo).pack(side="left")
        ttk.Button(botones, text="Limpiar", command=self._limpiar).pack(side="left", padx=6)
        ttk.Button(botones, text="Cargar ejemplo", command=self._cargar_ejemplo).pack(side="left")
        self.bind("<Return>", lambda _e: self._evaluar_un_ojo())

        salida = ttk.LabelFrame(marco, text="Resultado", padding=10, style="Seccion.TLabelframe")
        salida.grid(row=0, column=1, sticky="nsew")
        self.txt_resultado = tk.StringVar(value="Ingrese las medidas y pulse «Evaluar».")
        self.lbl_resultado = tk.Label(salida, textvariable=self.txt_resultado, font=("Segoe UI", 16, "bold"),
                                      fg=COLOR_SUAVE, anchor="w", justify="left", wraplength=380)
        self.lbl_resultado.pack(fill="x")
        self.txt_probabilidad = tk.StringVar()
        ttk.Label(salida, textvariable=self.txt_probabilidad, font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 10))
        ttk.Label(salida, text="Biomarcadores calculados", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tabla_bio = ttk.Treeview(salida, columns=("valor",), show="tree headings", height=8)
        self.tabla_bio.heading("#0", text="Biomarcador")
        self.tabla_bio.heading("valor", text="Valor")
        self.tabla_bio.column("#0", width=250)
        self.tabla_bio.column("valor", width=110, anchor="e")
        self.tabla_bio.pack(fill="both", expand=True, pady=(4, 0))
        self.txt_usadas = tk.StringVar()
        ttk.Label(salida, textvariable=self.txt_usadas, style="Aviso.TLabel", wraplength=380,
                  justify="left").pack(anchor="w", pady=(6, 0))
        return marco

    def _pestana_archivo(self, padre):
        marco = ttk.Frame(padre, padding=12)
        ayuda = ("El archivo (CSV o Excel) debe tener una fila por ojo y las columnas "
                 + ", ".join(NOMBRES_MEDIDAS) + " y Tipo_Ojo (DERECHO / IZQUIERDO). "
                 "Si trae una columna «Archivo», se conserva como identificador. "
                 "La base del proyecto ya tiene este formato.")
        ttk.Label(marco, text=ayuda, wraplength=900, justify="left").pack(anchor="w")
        botones = ttk.Frame(marco)
        botones.pack(fill="x", pady=8)
        ttk.Button(botones, text="Abrir archivo…", style="Accion.TButton", command=self._abrir_archivo).pack(side="left")
        self.btn_guardar = ttk.Button(botones, text="Guardar resultados…", command=self._guardar_resultados,
                                      state="disabled")
        self.btn_guardar.pack(side="left", padx=6)
        self.txt_lote = tk.StringVar()
        ttk.Label(botones, textvariable=self.txt_lote).pack(side="left", padx=10)

        columnas = ("id", "ojo", "resultado", "probabilidad", "vcdr", "razon")
        self.tabla_lote = ttk.Treeview(marco, columns=columnas, show="headings")
        for col, titulo, ancho in zip(columnas, ("Identificador", "Ojo", "Resultado", "Probabilidad", "VCDR",
                                                  "Razón de áreas copa/disco"), (160, 90, 200, 100, 80, 170)):
            self.tabla_lote.heading(col, text=titulo)
            self.tabla_lote.column(col, width=ancho, anchor="center")
        barra = ttk.Scrollbar(marco, orient="vertical", command=self.tabla_lote.yview)
        self.tabla_lote.configure(yscrollcommand=barra.set)
        self.tabla_lote.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        self.tabla_lote.tag_configure("pos", foreground=COLOR_POSITIVO)
        self.tabla_lote.tag_configure("neg", foreground=COLOR_NEGATIVO)
        return marco

    # ------------------------------------------------------------------ modelo
    def _elegir_modelo(self):
        ruta = filedialog.askopenfilename(title="Cargar modelo", initialdir=os.path.join(CARPETA, "modelos"),
                                          filetypes=[("Modelo", "*.joblib"), ("Todos", "*.*")])
        if ruta:
            self._cargar_modelo(ruta)

    def _detalle_modelo(self):
        if self.modelo is None:
            messagebox.showinfo("Modelo", "No hay modelo cargado.")
            return
        messagebox.showinfo("Detalle del modelo", self.modelo.resumen(completo=True))

    def _cargar_modelo(self, ruta):
        try:
            self.modelo = Modelo.cargar(ruta)
        except Exception as e:  # noqa: BLE001 - se muestra al usuario
            messagebox.showerror("No se pudo cargar el modelo",
                                 f"{e}\n\nSi el modelo se guardó con otras versiones de las librerías, "
                                 "vuelva a generarlo con «python entrenar_modelo.py» (tarda unos segundos).")
            return
        self.txt_modelo.set(self.modelo.resumen())
        self.lbl_modelo.configure(foreground=COLOR_POSITIVO if self.modelo.get("provisional") else "")
        self._estado(f"Modelo cargado: {os.path.basename(ruta)}")

    # ------------------------------------------------------------------ un ojo
    def _leer_medidas(self):
        medidas = {}
        for clave, etiqueta in MEDIDAS:
            texto = self.campos[clave].get().strip().replace(",", ".")
            if not texto:
                raise ValueError(f"Falta: {etiqueta}")
            try:
                medidas[clave] = float(texto)
            except ValueError:
                raise ValueError(f"No es un número: {etiqueta}") from None
        return medidas

    def _evaluar_un_ojo(self):
        if self.modelo is None:
            messagebox.showwarning("Sin modelo", "Primero cargue un modelo.")
            return
        try:
            variables = calcular_variables(self._leer_medidas(), self.ojo.get())
        except ValueError as e:
            messagebox.showerror("Revise las medidas", str(e))
            return
        resultado = self.modelo.predecir(pd.DataFrame([variables])).iloc[0]
        positivo = int(resultado["prediccion"]) == 1
        self.txt_resultado.set("Sospecha de glaucoma" if positivo else "Sin sospecha de glaucoma")
        self.lbl_resultado.configure(fg=COLOR_POSITIVO if positivo else COLOR_NEGATIVO)
        if "probabilidad" in resultado:
            texto = f"Probabilidad estimada de glaucoma: {resultado['probabilidad']:.0%}"
            if self.modelo.get("umbral") is not None:
                texto += f"  (umbral del modelo: {self.modelo.get('umbral'):.0%})"
            self.txt_probabilidad.set(texto)
        else:
            self.txt_probabilidad.set("")
        self._mostrar_biomarcadores(variables)
        self.txt_usadas.set(f"El modelo usa {len(self.modelo.variables)} variables (ver «Ver detalle»).")
        self._estado("Evaluación lista.")

    def _mostrar_biomarcadores(self, v):
        self.tabla_bio.delete(*self.tabla_bio.get_children())
        filas = [
            ("Razón copa/disco vertical (VCDR)", f"{v['VCDR']:.3f}"),
            ("Razón copa/disco horizontal (HCDR)", f"{v['HCDR']:.3f}"),
            ("Razón de áreas copa/disco", f"{v['Cup_Disc_Area_ratio']:.3f}"),
            ("Razón de áreas anillo/disco", f"{v['Rim_Disc_Area_ratio']:.3f}"),
            ("Área del anillo neurorretiniano (px²)", f"{v['rim_area']:,.0f}"),
            ("Cumple la regla ISNT (I ≥ S ≥ N ≥ T)", "Sí" if v["ISNT"] else "No"),
            ("Razón I/S", f"{v['I/S']:.3f}"),
            ("Razón N/S", f"{v['N/S']:.3f}"),
        ]
        for nombre, valor in filas:
            self.tabla_bio.insert("", "end", text=nombre, values=(valor,))

    def _limpiar(self):
        for var in self.campos.values():
            var.set("")
        self.txt_resultado.set("Ingrese las medidas y pulse «Evaluar».")
        self.lbl_resultado.configure(fg=COLOR_SUAVE)
        self.txt_probabilidad.set("")
        self.txt_usadas.set("")
        self.tabla_bio.delete(*self.tabla_bio.get_children())

    def _cargar_ejemplo(self):
        self.ojo.set(EJEMPLO["ojo"])
        for clave in NOMBRES_MEDIDAS:
            self.campos[clave].set(str(EJEMPLO[clave]))
        self._estado("Ejemplo cargado: imagen 390.jpg de ORIGA (ojo derecho con glaucoma).")

    # ------------------------------------------------------------------ fotografía
    def _abrir_foto(self):
        ruta = filedialog.askopenfilename(title="Abrir fotografía de fondo de ojo",
                                          filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                                                     ("Todos", "*.*")])
        if not ruta:
            return
        try:
            from PIL import Image
            imagen = Image.open(ruta)
            imagen.load()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("No se pudo abrir la imagen", str(e))
            return
        self.imagen, self.ruta_imagen = imagen.convert("RGB"), ruta
        self._mostrar_en_vista(self.imagen)
        self.txt_vista.set(f"{os.path.basename(ruta)} · {imagen.size[0]} × {imagen.size[1]} px")
        self.txt_resultado_foto.set("Elija el ojo y pulse «Analizar».")
        self.lbl_resultado_foto.configure(fg=COLOR_SUAVE)
        self.txt_probabilidad_foto.set("")
        self.tabla_foto.delete(*self.tabla_foto.get_children())
        self.btn_pasar.configure(state="disabled")

    def _mostrar_en_vista(self, imagen):
        from PIL import ImageTk
        copia = imagen.copy()
        copia.thumbnail((LADO_VISTA, LADO_VISTA))
        self.foto_tk = ImageTk.PhotoImage(copia)
        self.vista.configure(image=self.foto_tk, text="", width=LADO_VISTA, height=LADO_VISTA)

    def _asegurar_pesos(self):
        """Devuelve True si hay pesos disponibles; si no, ofrece descargarlos o buscarlos."""
        if os.path.exists(self.ruta_pesos):
            return True
        if messagebox.askyesno(
                "Faltan los pesos de la red",
                "Para segmentar se necesita el archivo de pesos de Bazi et al. (2024), de unos 430 MB.\n\n"
                "¿Descargarlo ahora desde Hugging Face? (Sí)\n"
                "¿O ya lo tiene en el computador y quiere buscarlo? (No)"):
            self._descargar_pesos()
            return False
        ruta = filedialog.askopenfilename(title="Buscar " + NOMBRE_PESOS,
                                          filetypes=[("Pesos", "*.tar *.pth"), ("Todos", "*.*")])
        if ruta:
            self.ruta_pesos = ruta
            return True
        return False

    def _descargar_pesos(self):
        os.makedirs(CARPETA_PESOS, exist_ok=True)
        destino = os.path.join(CARPETA_PESOS, NOMBRE_PESOS)
        self._ocupar(True, "Descargando los pesos de la red…")

        def tarea():
            try:
                temporal = destino + ".parcial"
                with urllib.request.urlopen(URL_PESOS, timeout=60) as r, open(temporal, "wb") as f:
                    total = int(r.headers.get("Content-Length", 0))
                    leido = 0
                    while True:
                        bloque = r.read(1 << 20)
                        if not bloque:
                            break
                        f.write(bloque)
                        leido += len(bloque)
                        if total:
                            self.cola.put(("estado", f"Descargando los pesos de la red… {leido / total:.0%}"))
                os.replace(temporal, destino)
                self.cola.put(("descarga_lista", destino))
            except Exception as e:  # noqa: BLE001
                self.cola.put(("error", "No se pudo descargar los pesos: " + str(e) +
                               "\n\nPuede descargarlos con el navegador desde:\n" + URL_PESOS +
                               "\ny guardarlos en la carpeta «pesos» de la aplicación."))
        threading.Thread(target=tarea, daemon=True).start()
        self.after(150, self._revisar_cola)

    def _analizar_foto(self):
        if self.ocupado:
            return
        if self.modelo is None:
            messagebox.showwarning("Sin modelo", "Primero cargue un modelo.")
            return
        if self.imagen is None:
            messagebox.showwarning("Sin fotografía", "Primero abra una fotografía.")
            return
        if not self.ojo_foto.get():
            messagebox.showwarning("Falta el ojo", "Indique si la fotografía es del ojo derecho o del izquierdo. "
                                                  "Se necesita para saber cuál lado del disco es nasal y cuál temporal.")
            return
        if not self._asegurar_pesos():
            return
        imagen, ojo = self.imagen, self.ojo_foto.get()
        primera = self.segmentador is None
        self._ocupar(True, "Cargando la red de segmentación (la primera vez tarda hasta un minuto)…" if primera
                     else "Segmentando…")

        def tarea():
            try:
                if self.segmentador is None:
                    from tamizaje_glaucoma.segmentacion import Segmentador
                    self.segmentador = Segmentador(self.ruta_pesos)
                    self.cola.put(("estado", "Segmentando…"))
                medidas, mascara, info = self.segmentador.analizar(imagen, ojo)
                from tamizaje_glaucoma.segmentacion import superponer_contornos
                vista = superponer_contornos(imagen, mascara, info)
                self.cola.put(("foto_lista", (medidas, vista, ojo)))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.cola.put(("error", str(e)))
        threading.Thread(target=tarea, daemon=True).start()
        self.after(150, self._revisar_cola)

    def _revisar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "estado":
                    self._estado(dato)
                elif tipo == "error":
                    self._ocupar(False, "Ocurrió un error.")
                    messagebox.showerror("Error", dato)
                    return
                elif tipo == "descarga_lista":
                    self.ruta_pesos = dato
                    self._ocupar(False, "Pesos descargados. Pulse «Analizar».")
                    return
                elif tipo == "foto_lista":
                    self._ocupar(False, "Análisis listo. Verifique los contornos antes de interpretar el resultado.")
                    self._mostrar_resultado_foto(*dato)
                    return
        except queue.Empty:
            pass
        self.after(150, self._revisar_cola)

    def _ocupar(self, ocupado, texto):
        self.ocupado = ocupado
        self.btn_analizar.configure(state="disabled" if ocupado else "normal")
        self.configure(cursor="watch" if ocupado else "")
        self._estado(texto)

    def _mostrar_resultado_foto(self, medidas, vista, ojo):
        self.medidas_foto = (medidas, ojo)
        self._mostrar_en_vista(vista)
        self.txt_vista.set("Contornos: verde = disco óptico, amarillo = copa. Si no coinciden con la imagen, "
                           "el resultado no es confiable.")
        try:
            variables = calcular_variables(medidas, ojo)
        except ValueError as e:
            messagebox.showerror("Medidas no válidas", f"La segmentación produjo medidas no válidas: {e}")
            return
        resultado = self.modelo.predecir(pd.DataFrame([variables])).iloc[0]
        positivo = int(resultado["prediccion"]) == 1
        self.txt_resultado_foto.set("Sospecha de glaucoma" if positivo else "Sin sospecha de glaucoma")
        self.lbl_resultado_foto.configure(fg=COLOR_POSITIVO if positivo else COLOR_NEGATIVO)
        if "probabilidad" in resultado:
            texto = f"Probabilidad estimada de glaucoma: {resultado['probabilidad']:.0%}"
            if self.modelo.get("umbral") is not None:
                texto += f"  (umbral {self.modelo.get('umbral'):.0%})"
            self.txt_probabilidad_foto.set(texto)
        self.tabla_foto.delete(*self.tabla_foto.get_children())
        filas = [
            ("Razón copa/disco vertical (VCDR)", f"{variables['VCDR']:.3f}"),
            ("Razón de áreas copa/disco", f"{variables['Cup_Disc_Area_ratio']:.3f}"),
            ("Razón copa/disco horizontal (HCDR)", f"{variables['HCDR']:.3f}"),
            ("Cumple la regla ISNT (I ≥ S ≥ N ≥ T)", "Sí" if variables["ISNT"] else "No"),
            ("Grosor inferior del anillo, I (px)", f"{medidas['I']:.0f}"),
            ("Grosor superior del anillo, S (px)", f"{medidas['S']:.0f}"),
            ("Grosor nasal del anillo, N (px)", f"{medidas['N']:.0f}"),
            ("Grosor temporal del anillo, T (px)", f"{medidas['T']:.0f}"),
            ("Área del disco (px²)", f"{medidas['disc_area']:,.0f}"),
            ("Área de la copa (px²)", f"{medidas['cup_area']:,.0f}"),
            ("Diámetros del disco, h × v (px)", f"{medidas['Disco_X']:.0f} × {medidas['Disco_Y']:.0f}"),
            ("Diámetros de la copa, h × v (px)", f"{medidas['Excavacion_X']:.0f} × {medidas['Excavacion_Y']:.0f}"),
        ]
        for nombre, valor in filas:
            self.tabla_foto.insert("", "end", text=nombre, values=(valor,))
        self.btn_pasar.configure(state="normal")

    def _pasar_a_manual(self):
        if not self.medidas_foto:
            return
        medidas, ojo = self.medidas_foto
        self.ojo.set(ojo)
        for clave in NOMBRES_MEDIDAS:
            self.campos[clave].set(f"{medidas[clave]:.0f}")
        self.pestanas.select(self.pestana_manual)
        self._estado("Medidas copiadas al formulario manual: puede corregirlas y volver a evaluar.")

    # ------------------------------------------------------------------ archivo
    def _abrir_archivo(self):
        if self.modelo is None:
            messagebox.showwarning("Sin modelo", "Primero cargue un modelo.")
            return
        ruta = filedialog.askopenfilename(title="Abrir archivo de medidas",
                                          filetypes=[("CSV o Excel", "*.csv *.xlsx *.xls"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            df = pd.read_csv(ruta) if ruta.lower().endswith(".csv") else pd.read_excel(ruta)
            variables, errores = tabla_desde_medidas(df)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("No se pudo leer el archivo", str(e))
            return
        resultado = self.modelo.predecir(variables) if len(variables) else pd.DataFrame()
        salida = df.copy()
        salida["resultado"] = "No evaluado"
        if len(variables):
            salida.loc[resultado.index, "resultado"] = resultado["prediccion"].map(
                {1: "Sospecha de glaucoma", 0: "Sin sospecha de glaucoma"})
            if "probabilidad" in resultado:
                salida.loc[resultado.index, "probabilidad_glaucoma"] = resultado["probabilidad"].round(4)
        salida["observacion"] = ""
        for i, msg in errores:
            salida.loc[i, "observacion"] = msg
        salida["modelo"] = self.modelo.nombre
        self.resultados_lote = salida
        self._llenar_tabla_lote(salida, variables)
        n_pos = int((salida["resultado"] == "Sospecha de glaucoma").sum())
        self.txt_lote.set(f"{len(salida)} ojos · {n_pos} con sospecha · {len(errores)} con errores")
        self.btn_guardar.configure(state="normal")
        self._estado(f"Archivo evaluado: {os.path.basename(ruta)}")

    def _llenar_tabla_lote(self, salida, variables):
        self.tabla_lote.delete(*self.tabla_lote.get_children())
        for i, fila in salida.iterrows():
            ident = fila["Archivo"] if "Archivo" in salida.columns else str(i + 1)
            prob = fila.get("probabilidad_glaucoma")
            prob = "" if prob is None or pd.isna(prob) else f"{prob:.0%}"
            vcdr = f"{variables.loc[i, 'VCDR']:.3f}" if i in variables.index else ""
            razon = f"{variables.loc[i, 'Cup_Disc_Area_ratio']:.3f}" if i in variables.index else ""
            etiqueta = ("pos",) if fila["resultado"] == "Sospecha de glaucoma" else ("neg",)
            self.tabla_lote.insert("", "end", values=(ident, fila["Tipo_Ojo"], fila["resultado"], prob, vcdr, razon),
                                   tags=etiqueta)

    def _guardar_resultados(self):
        if self.resultados_lote is None:
            return
        ruta = filedialog.asksaveasfilename(title="Guardar resultados", defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not ruta:
            return
        try:
            if ruta.lower().endswith(".csv"):
                self.resultados_lote.to_csv(ruta, index=False)
            else:
                self.resultados_lote.to_excel(ruta, index=False)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("No se pudo guardar", str(e))
            return
        self._estado(f"Resultados guardados en {ruta}")

    def _estado(self, texto):
        self.txt_estado.set(texto)


def main():
    try:
        Aplicacion().mainloop()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
