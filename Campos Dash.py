import numpy as np
import sympy as sp

import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output

import webbrowser


def Campo_Escalar_R2(f, dom_x=[-5., 5.], dom_y=[-5., 5.],
                     res=20, colors='plasma'):
    '''Grafica las curvas de nivel de f(x,y) en R² usando la función Surface
    de Plotly

    Parametros
    ----------
        f : sympy expression
                Expresión simbólica del campo escalar en función de x e y.
                Ejemplo: f=1/sqrt(x**2 + y**2)

        dom_x, dom_y : list or tuple of 2 elements, default : [-5, 5]
                Extremos del dominio numerico que se generará para graficar
                el campo vectorial. El primer elemento debería ser el limite
                inferior mientras que el segundo elemento debería ser el limite
                superior del dominio para una dada variable.
                Ejemplo: dom_x=[-10, 10], dom_y=[0, 5]

        res : int, default : 20
                Raíz cuadrada de la cantidad de puntos en los que se calculará
                el campo. Es decir, se tomarán res valores de "x" y de "y" y se
                completará una grilla de "x" contra "y" (res×res elementos)
                como dominio.

        colors : str, default : "plasma"
                String con el nombre de un mapa de color de la librería que
                corresponda a la representación elegida.
                Para ver todas las opciones visitar:
                https://matplotlib.org/stable/tutorials/colors/colormaps.html
                https://plotly.com/python/builtin-colorscales/

    Devuelve
    --------
        fig : 

    '''
    # Se transforman las funciones simbolicas en funciones numericas
    x, y = sp.symbols('x y', real=True)  # Definición de variables simbolicas
    f_num = sp.lambdify([x, y], f, 'numpy')
    # Preparo las variables del dominio
    x = np.linspace(dom_x[0], dom_x[1], res)
    y = np.linspace(dom_y[0], dom_y[1], res)
    X, Y = np.meshgrid(x, y)
    # Calculo las variables de la imagen
    Z = f_num(X, Y)
    # Detalle en el título
    ltx_expr = sp.latex(f)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
                                     colorscale=colors)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                              highlightcolor="limegreen", project_z=True))
    fig.update_layout(title={'text': r'$f(x,y) = '+ltx_expr+'$',
                             'y':0.9, 'x':0.5,
                             'xanchor': 'center', 'yanchor': 'top',},
                      # hovermode=True,
                      autosize=True, margin=dict(l=65, r=50, b=65, t=150))
                    #   scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                    #   width=500, height=500,)
    return fig


def Gradiente_R2(f, dom_x=[-5., 5.], dom_y=[-5., 5.],
                 res=10, colors='plotly3'):
    '''Grafica el campo vectorial gradiente de f(x,y,z)
    utilizando la función Cone de Plotly.

    Parametros
    ----------
        f : sympy expression
                Expresión simbólica del campo escalar en función de x e y
                cuyo gradiente se quiere graficar.
                Ejemplo: f=1/sqrt(x**2 + y**2)

        dom_x, dom_y : list or tuple of 2 elements, default : [-5, 5]
                Extremos del dominio numerico que se generará para graficar
                el campo vectorial. El primer elemento debería ser el limite
                inferior mientras que el segundo elemento debería ser el limite
                superior del dominio para una dada variable.
                Ejemplo: dom_x=[-10, 10], dom_y=[0, 5]

        res : int, default : 20
                Raíz cuadrada de la cantidad de puntos en los que se calculará
                el campo. Es decir, se tomarán res valores de "x" y de "y",
                se completará una grilla de "x" contra "y" (res×res elementos)
                como dominio.

        colors : str, default : "plotly3"
                String con el nombre de un mapa de color de plotly.
                Para ver todas las opciones visitar:
                https://plotly.com/python/builtin-colorscales/

    Devuelve
    --------
        fig : plotly.graph_objs._figure.Figure
                La figura de plotly rellena con plotly.graph_objects.Cone,
                generados por la función.
                Para visualizar el plot, ejecutar luego fig.show().

    '''
    # Se transforman las funciones simbolicas en funciones numericas
    x, y = sp.symbols('x y', real=True)  # Definición de variables simbolicas
    f_num = sp.lambdify([x, y], f, 'numpy')
    u_num = sp.lambdify([x, y], f.diff('x'), 'numpy')
    v_num = sp.lambdify([x, y], f.diff('y'), 'numpy')
    # Preparo las variables del dominio numérico
    x = np.linspace(dom_x[0], dom_x[1], res)
    y = np.linspace(dom_y[0], dom_y[1], res)
    X, Y = np.meshgrid(x, y)
    x = X.flatten()
    y = Y.flatten()
    z = f_num(x, y)
    # Calculo las variables de la imagen
    U = u_num(x, y)
    V = v_num(x, y)
    W = (U**2 + V**2)
    # Corrijo en caso de constantes:
    if np.size(U)  != x.size:
        U = np.ones_like(x)*U
    if np.size(V)  != y.size:
        V = np.ones_like(y)*V
    if np.size(W)  != z.size:
        W = np.ones_like(z)*W
    # Normalizo para la representación gráfica
    norm = np.sqrt(U**2 + V**2 + W**2)
    # Me ahorro los vectores nulos
    mask = (norm != 0)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    U = U[mask]
    V = V[mask]
    W = W[mask]
    # Gráfico con Plotly
    fig = go.Figure(data=go.Cone(x=x, y=y, z=z, u=U, v=V, w=W,
                                 colorscale=colors, sizeref=1,
                                 anchor="center", showscale=False))
    return fig

#%%
app = dash.Dash(__name__,
                external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" ])

webbrowser.open_new_tab('http://127.0.0.1:8050/')


colors = px.colors.named_colorscales()

# App layout
app.layout = html.Div([

    html.H1("Campos Escalares de R² en R", style={'text-align': 'center'}),
    
    dcc.Markdown("f(x, y) ="),
    dcc.Input(id="expresion",
              type="text",
              placeholder="sin(x)*sin(y)/(x*y)",
              list="lista",
              debounce=True,
              style={'width': "40%"}),

    html.Datalist(id="lista",
                  children=[html.Option(value="(x**2 +y**2)*sin(x)*sin(y)/(x*y)"),
                            html.Option(value="(cos(x) - 1)*(cos(y) - 1)/(x*y)"),
                            html.Option(value="exp(-x**2 - y**2)"),
                            html.Option(value="Piecewise((sqrt(y*x), x*y>=0), (sqrt(-y*x), x*y<0))"),
                            html.Option(value="arg(sqrt(x+I*y))"),]),

    daq.BooleanSwitch(id='switch_grad', on=False,
                      label="Graficar Campo Gradiente", labelPosition="top"),
    
    dcc.Markdown("Dominio x:"),
    dcc.RangeSlider(id='dom_x', min=-30, max=30, value=[-10, 10], step=0.5,
                    allowCross=True, pushable=2,
                    marks=dict([(x, str(x)) for x in range(-30, 30+1, 5)])),
    
    dcc.Markdown("Dominio y:"),
    dcc.RangeSlider(id='dom_y', min=-30, max=30, value=[-10, 10], step=0.5,
                    allowCross=True, pushable=2,
                    marks=dict([(x, str(x)) for x in range(-30, 30+1, 5)])),
    
    html.Div(id='rango', children=[]),
        
    dcc.Markdown("Resolución:"),
    dcc.Slider(id='resolution', min=25, max=300, step=5, value=100,
               marks={25: '25', 50: '50', 75: '75', 100: '100', 125: '125', 150: '150',
                      175: '175', 200: '200', 225: '225', 250: '250', 275: '275', 300: '300',},),
    
    dcc.Markdown("Colores:"),
    dcc.Dropdown(id="cmaps", #La id es con lo que despues vamos a buscar a este Dropdown. Es una forma CSS-esca de guardar "objetos"
                 value="magma", #El valor que representa por defecto
                 options=[{'value': x, 'label': x} for x in colors],
                 style={'width': "40%"}),
    daq.BooleanSwitch(id='switch_color', on=False, label="Invertir Escala", labelPosition="top"),
    
    html.Br(),

    dcc.Graph(id='surface', figure={}, style=dict(height=1000, width=1000))

])

#%%
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='rango', component_property='children'),
     Output(component_id='surface', component_property='figure'),],
    [Input(component_id='expresion', component_property='value'),
     Input(component_id='switch_grad', component_property='on'),
     Input(component_id='dom_x', component_property='value'),
     Input(component_id='dom_y', component_property='value'),
     Input(component_id='resolution', component_property='value'),
     Input(component_id='cmaps', component_property='value'),
     Input(component_id='switch_color', component_property='on')]
)
def update_graph(expr, grad, dom_x, dom_y, res, cmap, invertir):
    if expr is None:
        expr = "sin(x)*sin(y)/(x*y)"
    if invertir:
        cmap += '_r'
    dominio = f'{dom_x}×{dom_y}'
    fig = Campo_Escalar_R2(sp.sympify(expr), dom_x, dom_y, res=int(res), colors=cmap)
    if grad:
        if invertir:
            cmap = cmap[:-2]
        fig2 = Gradiente_R2(sp.sympify(expr), dom_x, dom_y, res=int(res/5), colors=cmap)
        Conos = fig2.to_dict()['data'][0]
        fig.add_cone(**Conos)
    fig.update_layout(scene=dict(xaxis=dict(range=dom_x,),
                                 yaxis=dict(range=dom_y,)),)
    return dominio, fig, 


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)



