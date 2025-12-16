from .access_files import map_files
from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px

def plot_dashboard(mapped_files : dict[str, str], max_points=10000) -> None:

    # Load all files. Do this once to avoid reloading on every callback. 
    # Tried without but it was too slow.
    dataframes = {key: pd.read_csv(filepath) for key, filepath in mapped_files.items()}

    app = Dash()

    app.layout = html.Div([
        html.Div(children='RespFlow Breathing Signal Dashboard', style={'textAlign': 'center', 'fontSize': 24}),
        html.Hr(),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': key, 'value': key} for key in mapped_files.keys()
            ],
            value=list(mapped_files.keys())[0]
        ),
        dcc.Graph(figure={}, id='breathing_chart')
    ])


    @callback(
        Output(component_id='breathing_chart', component_property='figure'),
        Input(component_id='dropdown', component_property='value')
    )
    def update_graph(selected_file_key, max_points=max_points):
        df = dataframes[selected_file_key]

        # Downsample if dataset is large (keep every Nth point for faster rendering)
        max_points = max_points  # Adjust this value based on performance needs
        if len(df) > max_points:
            step = len(df) // max_points
            df_plot = df.iloc[::step]
        else:
            df_plot = df

        fig = px.line(df_plot, x=df_plot.columns[0], y=df_plot.columns[1], title=f'Breathing Signal - {selected_file_key}')
        # fig.update_layout(uirevision='constant')  # Prevent plotly from retaining data from previous files 
        return fig
    
    
    app.run(debug=True)