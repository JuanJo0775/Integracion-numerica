# Aplicación Flask para Métodos de Integración Numérica

Esta aplicación permite resolver ecuaciones diferenciales utilizando los métodos numéricos de Euler y Heun, generando gráficas de las soluciones aproximadas.

## Estructura del proyecto

```
integracion-numerica/
│
├── app.py                # Archivo principal de la aplicación Flask
├── requirements.txt      # Dependencias del proyecto
│
└── templates/            # Carpeta de plantillas HTML
    ├── index.html        # Página principal con formulario
    └── result.html       # Página de resultados
```

## Requisitos previos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Crea un entorno virtual (recomendado):

```bash
python -m venv venv
```

2. Activa el entorno virtual:

**En Windows:**
```bash
venv\Scripts\activate
python app.py

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución de la aplicación

1. Con el entorno virtual activado, ejecuta:

```bash
python app.py
```

2. Abre tu navegador web y accede a la dirección:

```
http://127.0.0.1:5000/
```

## Uso de la aplicación

1. En la página principal, ingresa la función diferencial en la forma `y' = f(x,y)` (sólo debes ingresar la parte derecha).
2. Especifica los valores iniciales (x₀, y₀) y el punto final xf.
3. Indica el número de pasos para la aproximación.
4. Selecciona el método de integración (Euler o Heun).
5. Haz clic en "Resolver" para obtener los resultados.

## Ejemplos de funciones que puedes usar

- `x + y` para resolver `y' = x + y`
- `x**2 - sin(y)` para resolver `y' = x² - sin(y)`
- `exp(-x) * y` para resolver `y' = e^(-x) * y`

## Características

- Solución paso a paso mediante los métodos de Euler y Heun
- Visualización gráfica de las soluciones aproximadas
- Interfaz intuitiva para ingresar las ecuaciones diferenciales
- Soporte para diferentes funciones matemáticas a través de SymPy