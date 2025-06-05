# app.py
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json
import time
from sympy import symbols, sympify, lambdify, dsolve, Function, Eq, diff, solve
import matplotlib
from datetime import datetime
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import traceback

matplotlib.use('Agg')

app = Flask(__name__)

# Configuración
app.config['SECRET_KEY'] = 'your-secret-key-here'


class NumericalSolver:
    """Clase para métodos numéricos de resolución de ecuaciones diferenciales"""

    @staticmethod
    def euler_method(f, x0, y0, h, n):
        """Método de Euler para ecuaciones de primer orden"""
        points = [{'x': x0, 'y': y0, 'step': 0}]
        steps = []
        x, y = x0, y0

        for i in range(n):
            slope = f(x, y)
            step_info = {
                "step": i,
                "x": x,
                "y": y,
                "f_xy": slope,
                "h_f_xy": h * slope
            }

            y_next = y + h * slope
            x_next = x + h

            step_info.update({"x_next": x_next, "y_next": y_next})
            steps.append(step_info)

            x, y = x_next, y_next
            points.append({'x': x, 'y': y, 'step': i + 1})

        return points, steps

    @staticmethod
    def heun_method(f, x0, y0, h, n):
        """Método de Heun (predictor-corrector)"""
        points = [{'x': x0, 'y': y0, 'step': 0}]
        steps = []
        x, y = x0, y0

        for i in range(n):
            k1 = f(x, y)
            y_pred = y + h * k1
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
            points.append({'x': x, 'y': y, 'step': i + 1})

        return points, steps

    @staticmethod
    def runge_kutta_4(f, x0, y0, h, n):
        """Método de Runge-Kutta de 4to orden"""
        points = [{'x': x0, 'y': y0, 'step': 0}]
        steps = []
        x, y = x0, y0

        for i in range(n):
            k1 = f(x, y)
            k2 = f(x + h / 2, y + h * k1 / 2)
            k3 = f(x + h / 2, y + h * k2 / 2)
            k4 = f(x + h, y + h * k3)

            y_next = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            step_info = {
                "step": i,
                "x": x,
                "y": y,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "k4": k4,
                "y_next": y_next
            }
            steps.append(step_info)

            x = x + h
            y = y_next
            points.append({'x': x, 'y': y, 'step': i + 1})

        return points, steps

    @staticmethod
    def euler_second_order(f, x0, y0, yp0, h, n):
        """Método de Euler para ecuaciones de segundo orden"""
        points = [{'x': x0, 'y': y0, 'yp': yp0, 'step': 0}]
        steps = []
        x, y, yp = x0, y0, yp0

        for i in range(n):
            ypp = f(x, y, yp)

            step_info = {
                "step": i,
                "x": x,
                "y": y,
                "yp": yp,
                "ypp": ypp
            }

            y_next = y + h * yp
            yp_next = yp + h * ypp
            x_next = x + h

            step_info.update({
                "x_next": x_next,
                "y_next": y_next,
                "yp_next": yp_next
            })
            steps.append(step_info)

            x, y, yp = x_next, y_next, yp_next
            points.append({'x': x, 'y': y, 'yp': yp, 'step': i + 1})

        return points, steps

    @staticmethod
    def rk4_second_order(f, x0, y0, yp0, h, n):
        """Método RK4 para ecuaciones de segundo orden"""
        points = [{'x': x0, 'y': y0, 'yp': yp0, 'step': 0}]
        steps = []
        x, y, yp = x0, y0, yp0

        for i in range(n):
            k1y = yp
            k1yp = f(x, y, yp)

            k2y = yp + h * k1yp / 2
            k2yp = f(x + h / 2, y + h * k1y / 2, yp + h * k1yp / 2)

            k3y = yp + h * k2yp / 2
            k3yp = f(x + h / 2, y + h * k2y / 2, yp + h * k2yp / 2)

            k4y = yp + h * k3yp
            k4yp = f(x + h, y + h * k3y, yp + h * k3yp)

            y_next = y + h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
            yp_next = yp + h * (k1yp + 2 * k2yp + 2 * k3yp + k4yp) / 6

            step_info = {
                "step": i,
                "x": x,
                "y": y,
                "yp": yp,
                "k1y": k1y,
                "k1yp": k1yp,
                "k2y": k2y,
                "k2yp": k2yp,
                "k3y": k3y,
                "k3yp": k3yp,
                "k4y": k4y,
                "k4yp": k4yp,
                "y_next": y_next,
                "yp_next": yp_next
            }
            steps.append(step_info)

            x = x + h
            y = y_next
            yp = yp_next
            points.append({'x': x, 'y': y, 'yp': yp, 'step': i + 1})

        return points, steps


class AnalyticalSolver:
    """Clase para soluciones analíticas conocidas"""

    @staticmethod
    def get_analytical_solution(formula, x0, y0, x_values):
        """Obtiene la solución analítica para ecuaciones conocidas"""
        try:
            clean_formula = formula.lower().replace(' ', '')

            # Biblioteca de soluciones analíticas
            solutions = {
                'x+y': lambda x, x0, y0: -x - 1 + (y0 + x0 + 1) * np.exp(x - x0),
                'y': lambda x, x0, y0: y0 * np.exp(x - x0),
                '-y': lambda x, x0, y0: y0 * np.exp(-(x - x0)),
                'x': lambda x, x0, y0: (x ** 2) / 2 - (x0 ** 2) / 2 + y0,
                'x*y': lambda x, x0, y0: y0 * np.exp((x ** 2 - x0 ** 2) / 2),
                'sin(x)-y': lambda x, x0, y0: 0.5 * (np.sin(x) - np.cos(x)) + (
                            y0 - 0.5 * (np.sin(x0) - np.cos(x0))) * np.exp(-(x - x0))
            }

            if clean_formula in solutions:
                return [solutions[clean_formula](x, x0, y0) for x in x_values]

            # Intentar con SymPy para casos más complejos (con manejo de errores)
            try:
                x, y = symbols('x y')
                Y = Function('Y')

                # Limpiar y preparar la fórmula para SymPy
                cleaned_formula = formula.replace('exp', 'exp').replace('sin', 'sin').replace('cos', 'cos')
                cleaned_formula = cleaned_formula.replace('log', 'log').replace('sqrt', 'sqrt')

                # Intentar parsear la expresión
                expr = sympify(cleaned_formula, locals={'x': x, 'y': y, 'pi': np.pi, 'e': np.e})
                eq = Eq(Y(x).diff(x), expr)
                sol = dsolve(eq, Y(x))

                # Extraer la constante de integración
                if sol:
                    expr = sol.rhs
                    C1 = symbols('C1')

                    # Aplicar condición inicial
                    from sympy import solve
                    initial_eq = expr.subs(x, x0) - y0
                    try:
                        C1_solutions = solve(initial_eq, C1)
                        if C1_solutions:
                            C1_value = C1_solutions[0]
                            final_expr = expr.subs(C1, C1_value)
                            f = lambdify(x, final_expr, 'numpy')
                            return [float(f(xi)) for xi in x_values]
                    except:
                        pass
            except Exception as sympy_error:
                print(f"Error en SymPy: {sympy_error}")

            # Si no se encuentra solución analítica, usar RK4 muy preciso como referencia
            try:
                # Usar math.js parser más robusto
                import re

                # Convertir funciones comunes a formato evaluable
                eval_formula = formula.replace('exp', 'np.exp').replace('sin', 'np.sin').replace('cos', 'np.cos')
                eval_formula = eval_formula.replace('tan', 'np.tan').replace('log', 'np.log')
                eval_formula = eval_formula.replace('sqrt', 'np.sqrt').replace('pi', 'np.pi')
                eval_formula = eval_formula.replace('^', '**')

                # Crear función evaluable
                def f(x_val, y_val):
                    try:
                        # Crear un entorno seguro para evaluar
                        safe_dict = {
                            'x': x_val, 'y': y_val,
                            'np': np, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                            'pi': np.pi, 'e': np.e
                        }
                        return eval(eval_formula, {"__builtins__": {}}, safe_dict)
                    except:
                        return 0

                # RK4 con pasos muy pequeños como referencia
                h_precise = (x_values[-1] - x0) / 10000
                n_precise = 10000
                points, _ = NumericalSolver.runge_kutta_4(f, x0, y0, h_precise, n_precise)

                # Interpolar para obtener valores en x_values
                x_precise = [p['x'] for p in points]
                y_precise = [p['y'] for p in points]

                return np.interp(x_values, x_precise, y_precise)

            except Exception as eval_error:
                print(f"Error en evaluación: {eval_error}")
                return None

        except Exception as e:
            print(f"Error general en solución analítica: {e}")
            return None


class ChartGenerator:
    """Clase para generar gráficas"""

    @staticmethod
    def generate_comparison_chart(results, formula, x0, y0, xf):
        """Genera gráfica de comparación de métodos"""
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        colors = {
            'euler': '#4a90e2',
            'heun': '#7b68ee',
            'rk4': '#9370db',
            'analytical': '#ff6384'
        }

        for method, data in results.items():
            if data['points']:
                x_vals = [p['x'] for p in data['points']]
                y_vals = [p['y'] for p in data['points']]

                if method == 'analytical':
                    plt.plot(x_vals, y_vals, color=colors[method], linewidth=3,
                             linestyle='--', label='Solución Analítica', alpha=0.8)
                else:
                    plt.plot(x_vals, y_vals, color=colors[method], linewidth=2,
                             marker='o', markersize=4, label=data['name'])

        plt.plot(x0, y0, 'ro', markersize=8, label='Punto inicial')
        plt.title(f"Comparación de Métodos Numéricos\ny' = {formula}", fontsize=14, pad=20)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Guardar en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                    facecolor='#0f0f23', edgecolor='none')
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()

        return image_base64


class PDFReportGenerator:
    """Clase para generar reportes en PDF"""

    @staticmethod
    def generate_report(results, parameters, chart_base64):
        """Genera reporte completo en PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Título
        title = Paragraph("Reporte de Integración Numérica", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Información del problema
        problem_info = f"""
        <b>Ecuación Diferencial:</b> y' = {parameters['formula']}<br/>
        <b>Condiciones Iniciales:</b> (x₀, y₀) = ({parameters['x0']}, {parameters['y0']})<br/>
        <b>Punto Final:</b> xf = {parameters['xf']}<br/>
        <b>Número de Pasos:</b> {parameters['n']}<br/>
        <b>Tamaño del Paso:</b> h = {parameters['h']:.6f}<br/>
        <b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        story.append(Paragraph(problem_info, styles['Normal']))
        story.append(Spacer(1, 12))

        # Tabla de resultados
        if 'analytical' in results:
            ref_value = results['analytical']['points'][-1]['y']
        else:
            ref_value = results.get('rk4', {}).get('points', [{}])[-1].get('y', 0)

        table_data = [['Método', 'Valor Final', 'Error Absoluto', 'Error Relativo (%)', 'Tiempo (ms)']]

        for method, data in results.items():
            if method != 'analytical' and data['points']:
                final_value = data['points'][-1]['y']
                abs_error = abs(final_value - ref_value) if ref_value else 0
                rel_error = (abs_error / abs(ref_value)) * 100 if ref_value else 0

                table_data.append([
                    data['name'],
                    f"{final_value:.6f}",
                    f"{abs_error:.8f}",
                    f"{rel_error:.4f}%",
                    f"{data.get('time', 0):.2f}"
                ])

        if 'analytical' in results:
            table_data.append([
                'Analítica (Exacta)',
                f"{ref_value:.6f}",
                '0.00000000',
                '0.0000%',
                '-'
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 12))

        # Agregar gráfica
        if chart_base64:
            # Decodificar y guardar imagen temporalmente
            chart_data = base64.b64decode(chart_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(chart_data)
                tmp_file_path = tmp_file.name

            try:
                img = Image(tmp_file_path, width=500, height=300)
                story.append(img)
            finally:
                os.unlink(tmp_file_path)

        # Generar PDF
        doc.build(story)
        buffer.seek(0)

        return buffer


def parse_function(function_str):
    """Convierte string de función a función evaluable"""
    try:
        # Limpiar y preparar la función
        cleaned_str = function_str.replace('^', '**')
        cleaned_str = cleaned_str.replace('exp', 'exp').replace('sin', 'sin').replace('cos', 'cos')
        cleaned_str = cleaned_str.replace('tan', 'tan').replace('log', 'log').replace('sqrt', 'sqrt')
        cleaned_str = cleaned_str.replace('pi', 'pi').replace('E', 'E')

        # Intentar con SymPy primero
        try:
            x, y = symbols('x y')
            expr = sympify(cleaned_str, locals={'x': x, 'y': y, 'pi': symbols('pi'), 'E': symbols('E')})
            return lambdify((x, y), expr, 'numpy')
        except:
            # Si SymPy falla, usar evaluación directa
            def eval_function(x_val, y_val):
                try:
                    # Crear entorno seguro para evaluación
                    safe_dict = {
                        'x': x_val, 'y': y_val,
                        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                        'pi': np.pi, 'e': np.e, 'E': np.e,
                        'abs': np.abs
                    }
                    return eval(cleaned_str, {"__builtins__": {}}, safe_dict)
                except Exception as e:
                    raise ValueError(f"Error al evaluar la función en x={x_val}, y={y_val}: {e}")

            return eval_function

    except Exception as e:
        raise ValueError(f"Error al analizar la función '{function_str}': {e}")


def parse_second_order_function(function_str):
    """Convierte string de función de segundo orden a función evaluable"""
    try:
        # Limpiar y preparar la función
        cleaned_str = function_str.replace('^', '**')
        cleaned_str = cleaned_str.replace('yp', 'yp')  # Mantener yp como está
        cleaned_str = cleaned_str.replace('exp', 'exp').replace('sin', 'sin').replace('cos', 'cos')
        cleaned_str = cleaned_str.replace('tan', 'tan').replace('log', 'log').replace('sqrt', 'sqrt')
        cleaned_str = cleaned_str.replace('pi', 'pi').replace('E', 'E')

        # Intentar con SymPy primero
        try:
            x, y, yp = symbols('x y yp')
            expr = sympify(cleaned_str, locals={'x': x, 'y': y, 'yp': yp, 'pi': symbols('pi'), 'E': symbols('E')})
            return lambdify((x, y, yp), expr, 'numpy')
        except:
            # Si SymPy falla, usar evaluación directa
            def eval_function(x_val, y_val, yp_val):
                try:
                    # Crear entorno seguro para evaluación
                    safe_dict = {
                        'x': x_val, 'y': y_val, 'yp': yp_val,
                        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                        'pi': np.pi, 'e': np.e, 'E': np.e,
                        'abs': np.abs
                    }
                    return eval(cleaned_str, {"__builtins__": {}}, safe_dict)
                except Exception as e:
                    raise ValueError(f"Error al evaluar la función en x={x_val}, y={y_val}, y'={yp_val}: {e}")

            return eval_function

    except Exception as e:
        raise ValueError(f"Error al analizar la función de segundo orden '{function_str}': {e}")


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    """Resolver ecuación diferencial"""
    try:
        # Obtener parámetros
        data = request.get_json() if request.is_json else request.form

        equation_type = data.get('equation_type', 'first')
        methods = data.getlist('methods') if hasattr(data, 'getlist') else data.get('methods', [])
        if isinstance(methods, str):
            methods = [methods]

        x0 = float(data.get('x0', 0))
        y0 = float(data.get('y0', 1))
        xf = float(data.get('xf', 1))
        n = int(data.get('n', 10))
        h = (xf - x0) / n

        results = {}

        if equation_type == 'first':
            # Ecuaciones de primer orden
            formula = data.get('formula', 'x + y')
            f = parse_function(formula)

            # Ejecutar métodos seleccionados
            method_functions = {
                'euler': NumericalSolver.euler_method,
                'heun': NumericalSolver.heun_method,
                'rk4': NumericalSolver.runge_kutta_4
            }

            method_names = {
                'euler': 'Método de Euler',
                'heun': 'Método de Heun',
                'rk4': 'Runge-Kutta 4'
            }

            for method in methods:
                if method in method_functions:
                    start_time = time.time()
                    points, steps = method_functions[method](f, x0, y0, h, n)
                    end_time = time.time()

                    results[method] = {
                        'name': method_names[method],
                        'points': points,
                        'steps': steps,
                        'time': (end_time - start_time) * 1000
                    }

            # Solución analítica
            x_values = [p['x'] for p in results[methods[0]]['points']]
            analytical_values = AnalyticalSolver.get_analytical_solution(formula, x0, y0, x_values)

            if analytical_values is not None:
                results['analytical'] = {
                    'name': 'Solución Analítica',
                    'points': [{'x': x, 'y': y} for x, y in zip(x_values, analytical_values)],
                    'steps': [],
                    'time': 0
                }

        else:
            # Ecuaciones de segundo orden
            formula = data.get('second_order_formula', '-y - 2*yp')
            yp0 = float(data.get('yp0', 0))
            f = parse_second_order_function(formula)

            method_functions = {
                'euler': NumericalSolver.euler_second_order,
                'rk4': NumericalSolver.rk4_second_order
            }

            method_names = {
                'euler': 'Euler (2° Orden)',
                'rk4': 'Runge-Kutta 4 (2° Orden)'
            }

            available_methods = [m for m in methods if m in method_functions]

            for method in available_methods:
                start_time = time.time()
                points, steps = method_functions[method](f, x0, y0, yp0, h, n)
                end_time = time.time()

                results[method] = {
                    'name': method_names[method],
                    'points': points,
                    'steps': steps,
                    'time': (end_time - start_time) * 1000
                }

        # Generar gráfica
        chart_base64 = ChartGenerator.generate_comparison_chart(
            results, data.get('formula', formula), x0, y0, xf
        )

        # Preparar datos para el template
        result_data = {
            'equation_type': equation_type,
            'formula': data.get('formula', formula) if equation_type == 'first' else formula,
            'x0': x0,
            'y0': y0,
            'xf': xf,
            'n': n,
            'h': h,
            'methods': methods,
            'results': results,
            'chart': chart_base64
        }

        if equation_type == 'second':
            result_data['yp0'] = yp0

        return render_template('results.html', result=result_data)

    except Exception as e:
        error_msg = f"Error al resolver la ecuación: {str(e)}"
        return render_template('index.html', error=error_msg)


@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    """Exportar resultados a PDF"""
    try:
        data = request.get_json()

        # Generar PDF
        pdf_buffer = PDFReportGenerator.generate_report(
            data['results'],
            data['parameters'],
            data['chart']
        )

        # Crear nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'reporte_integracion_{timestamp}.pdf'

        # Guardar temporalmente
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, 'wb') as f:
            f.write(pdf_buffer.getvalue())

        return send_file(temp_path, as_attachment=True, download_name=filename, mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/csv', methods=['POST'])
def export_csv():
    """Exportar resultados a CSV"""
    try:
        data = request.get_json()
        results = data['results']

        # Generar CSV
        csv_content = "x,"

        # Headers
        for method in results:
            if method != 'analytical':
                csv_content += f"{results[method]['name']},"

        if 'analytical' in results:
            csv_content += "Analítica,"

        csv_content += "\n"

        # Datos
        max_points = max(len(results[method]['points']) for method in results if results[method]['points'])

        for i in range(max_points):
            row = ""

            # Valor x
            x_val = results[list(results.keys())[0]]['points'][i]['x'] if i < len(
                results[list(results.keys())[0]]['points']) else ""
            row += f"{x_val},"

            # Valores y para cada método
            for method in results:
                if method != 'analytical':
                    if i < len(results[method]['points']):
                        row += f"{results[method]['points'][i]['y']:.8f},"
                    else:
                        row += ","

            # Valor analítico
            if 'analytical' in results:
                if i < len(results['analytical']['points']):
                    row += f"{results['analytical']['points'][i]['y']:.8f},"

            csv_content += row + "\n"

        # Crear archivo temporal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'datos_integracion_{timestamp}.csv'
        temp_path = os.path.join(tempfile.gettempdir(), filename)

        with open(temp_path, 'w') as f:
            f.write(csv_content)

        return send_file(temp_path, as_attachment=True, download_name=filename, mimetype='text/csv')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/presets')
def get_presets():
    """Obtener ecuaciones predefinidas"""
    presets = [
        {
            'name': 'Crecimiento Exponencial',
            'formula': 'x + y',
            'x0': 0,
            'y0': 1,
            'xf': 1,
            'description': 'y\' = x + y'
        },
        {
            'name': 'Exponencial Pura',
            'formula': 'y',
            'x0': 0,
            'y0': 1,
            'xf': 2,
            'description': 'y\' = y'
        },
        {
            'name': 'Decaimiento Exponencial',
            'formula': '-y',
            'x0': 0,
            'y0': 1,
            'xf': 2,
            'description': 'y\' = -y'
        },
        {
            'name': 'Separable',
            'formula': 'x*y',
            'x0': 0,
            'y0': 1,
            'xf': 1,
            'description': 'y\' = x·y'
        },
        {
            'name': 'Lineal con Seno',
            'formula': 'sin(x) - y',
            'x0': 0,
            'y0': 0,
            'xf': 3.14159,
            'description': 'y\' = sin(x) - y'
        },
        {
            'name': 'No Lineal',
            'formula': 'x*x - y*y',
            'x0': 0,
            'y0': 1,
            'xf': 1,
            'description': 'y\' = x² - y²'
        }
    ]

    return jsonify(presets)


if __name__ == '__main__':
    import time

    app.run(debug=True, host='0.0.0.0', port=5000)