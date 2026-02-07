from .access_files import map_files, make_paths
from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def plot_dashboard(mapped_files : dict[str, str], max_points=10000) -> None:

    # Extract unique base filenames from the mapped files
    # Assumes files are structured as "stage/dir1/dir2/dir3/.../filename.csv"
    # Example Output: 
    # ['10/10-03-01.csv', '10/10-04-01.csv', '10/10-05-01.csv', 
    # '10/10-06-01.csv', '11/12/test-file.csv', '11/test-file.csv', 
    # '23/23-03-02.csv', '23/23-05-02.csv', '23/23-06-02.csv']
    base_files = sorted(set(
        key.split('/', 1)[-1] if '/' in key else key
        for key in mapped_files.keys()
    ))
    

    # Define all processing stages in order
    stages = ['raw', 'detrend', 'bandpass', 'fwr', 'screened', 'filled', 'smooth', 'feature']

    app = Dash()

    app.layout = html.Div([
        # Title
        html.Div(children='RespFlow Breathing Signal Dashboard', style={'textAlign': 'center', 'fontSize': 24}),
        # Just a line to split the title from the rest of the content
        html.Hr(),
        # Dropdown for selecting the file
        html.Div([
            html.Label('File:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='file-dropdown',
                options=[
                    {'label': file, 'value': file} for file in base_files
                ],
                value=base_files[0] if base_files else None,
                style={'width': '400px'}
            )
        ], style={'marginBottom': '20px'}),
        # Checklist for selecting which stages to display
        html.Div([
            html.Label('Signal Displayed:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Checklist(
                id='stage-checklist',
                options=[
                    {'label': f' {idx}: {stage}', 'value': stage}
                    for idx, stage in enumerate(stages, 1)
                ],
                value=['raw'],  # Default to showing raw
                inline=False,
                style={'columnCount': 2}
            )
        ], style={'marginBottom': '20px'}),
        # Checkbox for showing anomaly mask
        html.Div([
            dcc.Checklist(
                id='anomaly-checkbox',
                options=[{'label': ' Show anomaly mask', 'value': 'show'}],
                value=[],
                inline=True
            )
        ], style={'marginBottom': '20px'}),
        # Graph for displaying the breathing signals
        dcc.Graph(figure={}, id='breathing_chart')
    ])


    def get_file_path(mapped_files, filename, stage):
        """Finds the correct filepath for a specific stage and file."""
        for key, path in mapped_files.items():
            if filename in key and stage in key.lower():
                return path
        return None

    def load_and_downsample(filepath, max_points):
        """Loads the CSV and reduces number of points by plotting every nth (step)
            point up to `max_points`.
        """
        if not os.path.exists(filepath):
            return None

        try:
            df = pd.read_csv(filepath)
            if len(df) > max_points:
                step = len(df) // max_points
                return df.iloc[::step]
            return df
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    # --- THE MAIN CALLBACK (Updates every time a dropdown or checkbox is interacted with) ---

    @callback(
        Output('breathing_chart', 'figure'),
        [Input('file-dropdown', 'value'),
        Input('stage-checklist', 'value'),
        Input('anomaly-checkbox', 'value')]
    )
    def update_graph(selected_file, selected_stages, show_anomaly):
        if not selected_file or not selected_stages:
            return go.Figure()

        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis if anomaly mask is shown
        if show_anomaly and 'show' in show_anomaly:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        # Reverse viridis to have lighter colours for earlier stages
        viridis_colors = px.colors.sequential.Viridis[::-1]

        # Load anomaly data from screened stage if checkbox is checked
        anomaly_df = None
        if show_anomaly and 'show' in show_anomaly:
            screened_path = get_file_path(mapped_files, selected_file, 'screened')
            if screened_path:
                anomaly_df = load_and_downsample(screened_path, max_points)

        for idx, stage in enumerate(selected_stages):
            # 1. Find the file
            path = get_file_path(mapped_files, selected_file, stage)
            if not path:
                continue

            # 2. Load the data
            df = load_and_downsample(path, max_points)
            if df is None:
                continue

            # 3. Add to plot
            color = viridis_colors[idx % len(viridis_colors)]

            # Data trace - blank gaps where NaN
            fig.add_trace(go.Scattergl(
                x=df[df.columns[0]],
                y=df[df.columns[1]],
                mode='lines',
                name=stage,
                line=dict(color=color, width=1.5),
                connectgaps=False
            ), secondary_y=False if show_anomaly and 'show' in show_anomaly else None)

        # 4. Add anomaly mask overlay (once, from screened stage)
        if show_anomaly and 'show' in show_anomaly and anomaly_df is not None:
            # Look for anomaly column (e.g., Respiration_anomaly)
            signal_col = anomaly_df.columns[1]
            anomaly_col = f"{signal_col}_anomaly"
            if anomaly_col in anomaly_df.columns:
                fig.add_trace(go.Scattergl(
                    x=anomaly_df[anomaly_df.columns[0]],
                    y=anomaly_df[anomaly_col],
                    mode='lines',
                    name='anomaly mask',
                    line=dict(color='red', width=1.5),
                    opacity=0.7,
                    connectgaps=False
                ), secondary_y=True)

        # 5. Final Polish
        fig.update_layout(
            title=f'Breathing Signal - {selected_file}',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # Update y-axes labels if using secondary axis
        if show_anomaly and 'show' in show_anomaly:
            fig.update_yaxes(title_text="Signal", secondary_y=False)
            fig.update_yaxes(title_text="Anomaly (0/1)", secondary_y=True)

        return fig
    
    
    app.run(debug=False)