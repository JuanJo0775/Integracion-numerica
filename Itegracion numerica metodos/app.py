# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sympy import symbols, sympify, lambdify
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)


def euler_method(f, x0, y0, h, n):
    """
    Implementación del método de Euler
    f: función a integrar
    x0, y0: punto inicial
    h: tamaño del paso
    n: número de pasos
    """
    x_values = [x0]
    y_values = [y0]
    steps = []

    x = x0
    y = y0

    for i in range(n):
        step_info = {
            "step": i,
            "x": x,
            "y": y,
            "f(x,y)": f(x, y),
            "h*f(x,y)": h * f(x, y)
        }

        y_next = y + h * f(x, y)
        x = x + h
        y = y_next

        step_info["x_next"] = x
        step_info["y_next"] = y
        steps.append(step_info)

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values, steps


def heun_method(f, x0, y0, h, n):
    """
    Implementación del método de Heun (predictor-corrector)
    f: función a integrar
    x0, y0: punto inicial
    h: tamaño del paso
    n: número de pasos
    """
    x_values = [x0]
    y_values = [y0]
    steps = []

    x = x0
    y = y0

    for i in range(n):
        # Predictor (Euler)
        k1 = f(x, y)
        y_pred = y + h * k1

        # Corrector (promedio de pendientes)
        k2 = f(x + h, y_pred)
        y_next = y + h * (k1 + k2) / 2

        step_info = {
            "step": i,
            "x": x,
            "y": y,
            "k1": k1,
            "y_pred": y_pred,
            "k2": k2,
            "y_next": y_next
        }
        steps.append(step_info)

        x = x + h
        y = y_next

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values, steps


def generate_plot(x_values, y_values, x0, y0, xf, method_name, function_expr):
    """Genera la gráfica de solución para el método seleccionado"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'b.-', label=f'Solución por {method_name}')
    plt.plot(x0, y0, 'ro', label='Punto inicial')

    plt.title(f'Solución de y\'={function_expr} usando {method_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Guardar la figura en un objeto BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Codificar la imagen para mostrarla en HTML
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64


def parse_function(function_str):
    """Convierte un string de función a una función evaluable"""
    x, y = symbols('x y')
    try:
        expr = sympify(function_str)
        return lambdify((x, y), expr, 'numpy')
    except Exception as e:
        raise ValueError(f"Error al analizar la función: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    # Obtener datos del formulario
    function_str = request.form.get('function')
    x0 = float(request.form.get('x0'))
    y0 = float(request.form.get('y0'))
    xf = float(request.form.get('xf'))
    n = int(request.form.get('n'))
    method = request.form.get('method')

    try:
        # Parsear la función
        f = parse_function(function_str)

        # Calcular el tamaño del paso
        h = (xf - x0) / n

        # Resolver según el método seleccionado
        if method == 'euler':
            x_values, y_values, steps = euler_method(f, x0, y0, h, n)
            method_name = 'Método de Euler'
        elif method == 'heun':
            x_values, y_values, steps = heun_method(f, x0, y0, h, n)
            method_name = 'Método de Heun'
        else:
            return jsonify({'error': 'Método no válido'})

        # Generar la gráfica
        image_base64 = generate_plot(x_values, y_values, x0, y0, xf, method_name, function_str)

        # Resultado para enviar a la plantilla
        result = {
            'method': method_name,
            'function': function_str,
            'x0': x0,
            'y0': y0,
            'xf': xf,
            'h': h,
            'n': n,
            'steps': steps,
            'image': image_base64
        }

        return render_template('result.html', result=result)

    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)