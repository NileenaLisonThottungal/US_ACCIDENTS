import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_daq as daq
import base64
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import aiohttp
import asyncio
# Load dataset
file_path = r"US_Accidents_March23_sampled_500k.csv"
df = pd.read_csv(file_path)
# Preprocess data for the bubble chart
df_filteredd = df[['State', 'Severity']].dropna()
top_states = df_filteredd['State'].value_counts().nlargest(10).index
df_filteredd = df_filteredd[df_filteredd['State'].isin(top_states)]
# Preprocess data for correlation matrix
df_filtered = df[['Temperature(F)', 'Wind_Speed(mph)', 'Humidity(%)', 'Severity']].dropna()
correlation_matrix = df_filtered.corr()
# Data for Sunburst Chart
df_sunburst = df[['State', 'City', 'Severity', 'ID']].dropna()
df_grouped_sunburst = df_sunburst.groupby(['State', 'City', 'Severity']).size().reset_index(name='Accident_Count')
# Data for Severity vs. Distance Scatter Plot
df_filtered_scatter = df[['Distance(mi)', 'Severity', 'Weather_Condition']].dropna()
# Check for missing values in 'Severity' and 'Street'
df['Severity'] = df['Severity'].fillna(0).astype(int)
df['Street'] = df['Street'].fillna('Unknown')
df_filtered = df[['Severity', 'Start_Time']].dropna()
df_grouped = df_filtered.groupby(['Severity']).size().reset_index(name='Counts')
df_treemap = df[['Weather_Condition', 'Severity']].dropna()
df_grouped_treemap = df_treemap.groupby(['Weather_Condition', 'Severity']).size().reset_index(name='Count')
# Data for Choropleth Map
df_state_counts = df['State'].value_counts().reset_index()
df_state_counts.columns = ['State', 'Accident_Count']
# Preprocess data for the polar bar chart
df['Severity'] = df['Severity'].astype(int)
df_filtered = df[['City', 'Severity']].dropna()
# 1. Parallel Coordinates Plot
df_parallel = df[['Severity', 'Temperature(F)', 'Wind_Speed(mph)', 'Humidity(%)', 'Visibility(mi)']].dropna()

# 2. Violin Plot
df_violin = df[['Severity', 'Temperature(F)', 'Humidity(%)']].dropna()


# 13. Word Cloud
text = ' '.join(df['Description'].dropna())
wc = WordCloud(width=800, height=400, background_color='white').generate(text)

# Save Word Cloud to a BytesIO object
img = io.BytesIO()
wc.to_image().save(img, format='png')
img.seek(0)
img_base64 = base64.b64encode(img.getvalue()).decode()
# Aggregate data by city and severity
city_severity_counts = df.groupby(['City', 'Severity']).size().reset_index(name='Count')
# Accident Hotspot Data
df_hotspots = df[['Start_Lat', 'Start_Lng', 'Severity']].dropna()
# Function to get top 10 cities by accident count for a given severity level
def get_top_cities_by_severity(df, severity_level):
    df_severity = df[df['Severity'] == severity_level]
    top_cities = df_severity['City'].value_counts().head(10).reset_index()
    top_cities.columns = ['City', 'Count']
    return top_cities
# Find the top cities for each severity level
def get_top_cities(df):
    top_cities = pd.DataFrame()
    for severity in df['Severity'].unique():
        subset = df[df['Severity'] == severity]
        top_cities = pd.concat([top_cities, subset.nlargest(10, 'Count')])
    return top_cities.reset_index(drop=True)

top_cities = get_top_cities(city_severity_counts)

# Sample DataFrame for Accident Scene Sketch
sample_df = df[['ID', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Description', 'Severity']].dropna()
# Define color mapping for different severity levels
color_map = {
    1: 'blue',    # Color for Severity 1
    2: 'orange',  # Color for Severity 2
    3: 'red',     # Color for Severity 3
    4: 'purple'   # Color for Severity 4
}

# Initialize the app with external stylesheets and suppress_callback_exceptions=True
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
], suppress_callback_exceptions=True)
server = app.server
# Define the summarized report
summarized_text = """
The US Accidents dataset (2016 - 2023) encompasses traffic accident records from 49 states across the United States, totaling approximately 1.5 million records. For the subset of 500,000 records analyzed, significant missing values are noted in key columns like End_Lat, End_Lng, and Weather_Timestamp. There are no duplicated rows. Numerical features reveal a mean accident severity of 2.21 on a scale from 1 (least impact) to 4 (most impact). The average latitude and longitude of incidents are 36.21 and -94.74 respectively, with an average distance affected being 0.56 miles. The dataset shows an average temperature of 61.65°F and an average humidity of 64.83%, with 'Fair' being the most common weather condition.
Categorically, Miami tops the list with 12,141 accidents, followed by states like California (113,274 accidents), Florida (56,710), and Texas (37,355). The severity distribution shows that most accidents are classified under Severity level 2 (398,142), with fewer incidents at Severity levels 3 (84,520) and 4 (13,064). Notably, accidents occur more frequently during the day (344,967) compared to the night (153,550). The dataset also indicates that 74,035 accidents occurred where a traffic signal was present, while 425,965 did not have a traffic signal. Overall, the data suggests that accidents are more common in clear or fair weather conditions and highlights specific states and cities with high accident frequencies, underscoring areas that may require targeted traffic safety interventions.
"""

# Define API endpoint for Phi-2
api_endpoint = "http://localhost:11434/api/generate"

async def ask_phi_with_summary(question):
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "phi",
            "prompt": (
                f"Below is a summary of US accident data:\n{summarized_text}\n\n"
                "Based on the information above, please provide a concise answer to the following question:\n\n"
                f"{question}"
            ),
            "raw": True,
            "stream": False
        }
        async with session.post(api_endpoint, json=payload) as response:
            if response.status == 200:
                response_text = await response.json()
                return response_text.get('response', '').split('.')[0] + '.'
            else:
                return f"Error: {response.status} - {await response.text()}"

# Wrapper function to run async code in sync callback
def run_async_chat(question):
    return asyncio.run(ask_phi_with_summary(question))

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),  # This component tracks the URL
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1("US Accidents Analysis Dashboard", className="text-center mb-4 moving-heading", style={"color": "#ffffff"}),
                html.P("Analyze and visualize US accident data with interactive charts.", className="text-center mb-4", style={"color": "#ffffff"})
            ]),
            width={"size": 8, "offset": 2}
        )
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(
            html.Div([
                dbc.Nav([
                    dbc.NavLink("Overview", id="overview-link", href="/", className="nav-link"),
                    html.Button("Chat with Us", id="open-chat", n_clicks=0, className="btn btn-primary mt-3"),
                    dbc.NavLink("Severity Analysis", id="severity-link", href="/severity", className="nav-link"),
                    dbc.NavLink("City-Wise Counts", id="city-link", href="/city", className="nav-link"),
                    dbc.NavLink("Location Analysis", id="location-link", href="/location", className="nav-link"),
                    dbc.NavLink("Search Accident", id="search-link", href="/search", className="nav-link"),
                    dbc.NavLink("Accident Scene", id="scene-link", href="/scene", className="nav-link") , # New tab for Accident Scene
                    dbc.NavLink("Accident Summary and Download", id="summary-link", href="/summary", className="nav-link"),
                    dbc.NavLink("Severity Funnel Chart", id="funnel-link", href="/funnel", className="nav-link"),
                    dbc.NavLink("Accident Visualizations", id="visualization-link", href="/visualizations", className="nav-link"),
                    dbc.NavLink("Hotspots and Severity Map", id="hotspots-link", href="/hotspots", className="nav-link"),
                    dbc.NavLink("Sunburst Chart", id="sunburst-link", href="/sunburst", className="nav-link"),
                    dbc.NavLink("Scatter Plot", id="scatter-link", href="/scatter", className="nav-link"),
                    dbc.NavLink("Treemap", id="treemap-link", href="/treemap", className="nav-link"),
                    dbc.NavLink("Accident Choropleth Map", id="choropleth-link", href="/choropleth", className="nav-link"),
                    dbc.NavLink("Top Cities by Severity", id="polar-bar-link", href="/polar-bar", className="nav-link"),
                    dbc.NavLink("Correlation Matrix", id="correlation-link", href="/correlation", className="nav-link"),
                    dbc.NavLink("Bubble Chart", id="bubble-chart-link", href="/bubble-chart", className="nav-link"),
                ], vertical=True, pills=True, className="sidebar"),
            ], className="sidebar-container"),
            width=2
        ),
        dbc.Col(
            html.Div(id='page-content'),
            width=10
        )
    ]),
    # Chat Modal
    dbc.Modal(
        [
            dbc.ModalHeader("Chat with Us"),
            dbc.ModalBody(
                html.Div([
                    dcc.Input(id='chat-input', type='text', placeholder='Type your message here...'),
                    html.Button('Send', id='chat-send', n_clicks=0),
                    html.Div(id='chat-response')
                ])
            ),
            dbc.ModalFooter(
                html.Button("Close", id="close-chat", className="btn btn-secondary")
            )
        ],
        id="chat-modal",
        size="lg",
        is_open=False
    )
], fluid=True)

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def update_page(pathname):
    if pathname == '/severity':
        return html.Div([
            dbc.Row([
                dbc.Col(
                    daq.Gauge(
                        id='severity-gauge',
                        label='Severity Level',
                        color={"gradient": True, "ranges": {"#ffcccc": [1, 2], "#ff9999": [2, 3], "#ff6666": [3, 4]}},
                        value=df['Severity'].mode()[0],  # Default to the most common severity
                        max=4,
                        min=1,
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id='severity-pie-chart'),
                    width=6
                ),
            ]),
            dcc.Slider(
                id='severity-slider',
                min=1,
                max=4,
                step=1,
                value=df['Severity'].mode()[0],  # Default to the most common severity
                marks={i: str(i) for i in range(1, 5)},
                updatemode='drag'
            )
        ])
    elif pathname == '/city':
        return html.Div([
            html.H3("Top Cities by Accident Counts", style={"color": "#ffffff"}),
            dcc.Graph(
                figure=px.bar(
                    top_cities,
                    x='City',
                    y='Count',
                    color='Severity',
                    title='Top Cities by Accident Counts and Severity',
                    labels={'Count': 'Number of Accidents'},
                    facet_col='Severity',
                    facet_col_wrap=2,
                    height=600
                ).update_layout(
                    plot_bgcolor='#333333',
                    paper_bgcolor='#1f1f1f',
                    font_color='#ffffff'
                )
            )
        ])
    elif pathname == '/location':  
        return html.Div([
            html.H3("US Accidents Map", style={"color": "#ffffff", "text-align": "center", "margin-bottom": "20px"}),
            dcc.Dropdown(
                id='severity-dropdown',
                options=[{'label': str(sev), 'value': sev} for sev in sorted(df['Severity'].unique())],
                value=df['Severity'].unique().tolist(),  # Default to all severities
                multi=True,
                style={"width": "50%", "margin": "auto"}  
            ),
            html.Div([
                dcc.Graph(id='map-graph', style={"height": "90vh"}) 
            ], style={"height": "90vh", "width": "100%", "margin-top": "20px"})
        ])
    elif pathname == '/search':  
        return html.Div([
            html.H3("Search for Accident by ID", style={"color": "#ffffff"}),
            dcc.Input(id='search-input', type='text', placeholder='Enter Accident ID', style={"margin-right": "10px"}),
            html.Button('Search', id='search-button', n_clicks=0),
            html.Div(id='search-results', style={"margin-top": "20px"})
        ])
    elif pathname == '/scene':  
        return html.Div([
            html.H3("Accident Scene Sketch", style={"color": "#ffffff"}),
            dcc.Dropdown(
                id='accident-dropdown',
                options=[{'label': f'Accident {i}', 'value': i} for i in sample_df['ID'].unique()],
                value=sample_df['ID'].iloc[0]  
            ),
            dcc.Graph(id='accident-sketch')
        ])
    elif pathname == '/summary':
        return html.Div([
            html.H3("Accident Summary and Download", style={"color": "#ffffff"}),
            dcc.Input(id='search-input', type='text', placeholder='Enter Accident ID', style={"margin-right": "10px"}),
            html.Button('Search', id='search-button', n_clicks=0),
            html.Div(id='accident-summary', style={"margin-top": "20px"}),
            # Download Button
            html.Button("Download Summary", id="download-button"),
            dcc.Download(id="download-summary")
    ])
    elif pathname == '/funnel':
        # Code for funnel chart page
        return html.Div([
            html.H3("Severity Funnel Chart", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Severity Levels", style={"color": "#ffffff"}),
                    dcc.Dropdown(
                        id='severity-dropdown',
                        options=[{'label': f'Severity {sev}', 'value': sev} for sev in df_grouped['Severity'].unique()],
                        value=df_grouped['Severity'].unique().tolist(),  # Default value
                        multi=True
                    )
                ], width=4),
                dbc.Col(dcc.Graph(id='funnel-chart'), width=8)
            ])
        ])
    elif pathname == '/visualizations':
        return html.Div([
            html.H3("US Accidents Visualization Dashboard", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(dcc.Graph(id="parallel-coordinates"), lg=6),
                dbc.Col(dcc.Graph(id="violin-plot"), lg=6),
            ]),
        ])
    elif pathname == '/hotspots':
        return html.Div([
            html.H3("Accident Hotspots and Severity Map", style={"color": "#ffffff"}),
            dcc.Graph(id='hotspots-map')
        ])
    elif pathname == '/sunburst':
        return html.Div([
            html.H3("US Accident Sunburst Chart", style={"color": "#ffffff"}),
            dcc.Graph(id='sunburst-chart'),
            dcc.Dropdown(
                id='severity-dropdown',
                options=[{'label': f'Severity {i}', 'value': i} for i in sorted(df_grouped_sunburst['Severity'].unique())],
                value=2,  # Default value
                clearable=False,
                style={'width': '50%'}
            )
        ])
    elif pathname == '/scatter':
        return html.Div([
            html.H3("Severity vs. Distance Scatter Plot", style={"color": "#ffffff"}),
            dcc.Graph(id='severity-distance-scatter')
        ])
    elif pathname == '/treemap':
        return html.Div([
            html.H3("Accident Severity and Weather Condition Treemap", style={"color": "#ffffff"}),
            dcc.Graph(id='treemap')
        ])
    elif pathname == '/choropleth':  # New route for Choropleth Map
        return html.Div([
            html.H3("US Accident Counts by State", style={"color": "#ffffff"}),
            dcc.Graph(id='map')
        ])
    elif pathname == '/polar-bar':  # New route for Polar Bar Chart
        return html.Div([
            html.H3("Top Cities by Accident Counts for Selected Severity", style={"color": "#ffffff"}),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Severity Level", style={"color": "#ffffff"}),
                            dcc.Slider(
                                min=df['Severity'].min(),
                                max=df['Severity'].max(),
                                step=1,
                                marks={i: str(i) for i in range(df['Severity'].min(), df['Severity'].max() + 1)},
                                value=df['Severity'].min(),
                                id='slider'
                            ),
                        ],
                        lg=4
                    ),
                    dbc.Col(dcc.Graph(id='graph'), lg=8),
                ]
            )
        ])
    elif pathname == '/correlation':  # New route for Correlation Matrix
        return html.Div([
            html.H3("Accidents Correlation Matrix", style={"color": "#ffffff"}),
            dcc.Graph(id='correlation-matrix-heatmap')
        ])
    elif pathname == '/bubble-chart':  # New route for Bubble Chart
        return html.Div([
            html.H3("Accident Counts and Severity by State (Bubble Chart)", style={"color": "#ffffff"}),
            dcc.RangeSlider(
                id='severity-slider',
                min=1,
                max=5,
                marks={i: f'Severity {i}' for i in range(1, 6)},
                value=[1, 5]
            ),
            dcc.Graph(id='bubble-chart')
        ])
    else:
        return html.Div([
            html.H3("Overview", style={"color": "#ffffff"}),
            html.P(
                "This is a countrywide traffic accident dataset, which covers 49 states of the United States. "
                "The data is continuously being collected from February 2016, using several data providers, including multiple APIs that provide streaming traffic event data. "
                "These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. "
                "Currently, there are about 1.5 million accident records in this dataset. Check the below descriptions for more detailed information.",
                style={"color": "#ffffff"}
            ),
            dbc.Row([
                dbc.Col(
                    html.Img(src='/assets/scene.jpg', style={'width': '100%', 'height': 'auto'}),
                    width=6
                ),
                dbc.Col(
                    html.Img(src='/assets/traffic.jpg', style={'width': '100%', 'height': 'auto'}),
                    width=6
                )
            ], className="mb-4"),
            html.H3("Summary of Dataset", style={"color": "#ffffff"}),
            html.P(f"Total Records: {df.shape[0]}", style={"color": "#ffffff"}),
            html.P(f"Total Columns: {df.shape[1]}", style={"color": "#ffffff"})
        ], style={"margin-top": "20px"})

@app.callback(
    Output('severity-gauge', 'value'),
    Input('severity-slider', 'value')
)
def update_gauge(value):
    return value

@app.callback(
    Output('severity-pie-chart', 'figure'),
    Input('severity-slider', 'value')
)
def update_pie_chart(severity_value):
    filtered_df = df[df['Severity'] == severity_value]
    street_counts = filtered_df['Street'].value_counts().head(10)
    fig = px.pie(
        values=street_counts.values,
        names=street_counts.index,
        title=f'Street Distribution for Severity {severity_value}'
    )
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#1f1f1f',
        font_color='#ffffff'
    )
    return fig

@app.callback(
    Output('map-graph', 'figure'),
    Input('severity-dropdown', 'value')
)
def update_map(selected_severity):
    filtered_df = df[df['Severity'].isin(selected_severity)]
    fig = px.scatter_mapbox(filtered_df, 
                            lat="Start_Lat", 
                            lon="Start_Lng", 
                            color="Severity", 
                            hover_name="ID", 
                            zoom=3, 
                            height=800)
    fig.update_layout(mapbox_style="carto-darkmatter")
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#1f1f1f',
        font_color='#ffffff'
    )
    return fig

@app.callback(
    Output('search-results', 'children'),
    Input('search-button', 'n_clicks'),
    State('search-input', 'value')
)
def search_accident(n_clicks, search_value):
    if n_clicks > 0 and search_value:
        # Filter DataFrame based on the search input (assuming 'ID' is the unique identifier)
        accident_details = df[df['ID'] == search_value]

        if not accident_details.empty:
            wind_speed = accident_details['Wind_Speed(mph)'].values[0]
            temperature = accident_details['Temperature(F)'].values[0]
            humidity = accident_details['Humidity(%)'].values[0]
            pressure = accident_details['Pressure(in)'].values[0]
            visibility = accident_details['Visibility(mi)'].values[0]
            precipitation = accident_details['Precipitation(in)'].values[0]
            wind_direction = accident_details['Wind_Direction'].values[0]
            weather_condition = accident_details['Weather_Condition'].values[0]

            return html.Div([
                html.H4(f"Details for Accident ID: {search_value}", style={"color": "#ffffff"}),

                # First row: Wind Speed, Humidity, Temperature
                dbc.Row([
                    dbc.Col(
                        daq.Gauge(
                            id='wind-speed-gauge',
                            label='Wind Speed (mph)',
                            value=wind_speed,
                            max=100,  # Adjust max value as per your dataset
                            min=0,
                            color={"gradient": True, "ranges": {"#33ccff": [0, 25], "#ffcc00": [25, 50], "#ff3300": [50, 100]}}
                        ),
                        width=4  # Adjust the width of the column
                    ),
                    dbc.Col(
                        daq.Gauge(
                            id='humidity-gauge',
                            label='Humidity (%)',
                            value=humidity,
                            max=100,  # Adjust max value as per your dataset
                            min=0,
                            color={"gradient": True, "ranges": {"#33ccff": [0, 50], "#ffcc00": [50, 75], "#ff3300": [75, 100]}}
                        ),
                        width=4
                    ),
                    dbc.Col(
                        daq.Gauge(
                            id='temperature-gauge',
                            label='Temperature (F)',
                            value=temperature,
                            max=120,  # Adjust max value as per your dataset
                            min=-20,
                            color={"gradient": True, "ranges": {"#33ccff": [-20, 40], "#ffcc00": [40, 80], "#ff3300": [80, 120]}}
                        ),
                        width=4
                    ),
                ]),

                # Second row: Pressure, Visibility, Precipitation
                dbc.Row([
                    dbc.Col(
                        daq.Gauge(
                            id='pressure-gauge',
                            label='Pressure (in)',
                            value=pressure,
                            max=32,  # Adjust max value as per your dataset
                            min=28,
                            color={"gradient": True, "ranges": {"#33ccff": [28, 29.5], "#ffcc00": [29.5, 31], "#ff3300": [31, 32]}}
                        ),
                        width=4
                    ),
                    dbc.Col(
                        daq.Gauge(
                            id='visibility-gauge',
                            label='Visibility (mi)',
                            value=visibility,
                            max=10,  # Adjust max value as per your dataset
                            min=0,
                            color={"gradient": True, "ranges": {"#ff3300": [0, 2], "#ffcc00": [2, 5], "#33ccff": [5, 10]}}
                        ),
                        width=4
                    ),
                    dbc.Col(
                        daq.Gauge(
                            id='precipitation-gauge',
                            label='Precipitation (in)',
                            value=precipitation,
                            max=2,  # Adjust max value as per your dataset
                            min=0,
                            color={"gradient": True, "ranges": {"#33ccff": [0, 0.5], "#ffcc00": [0.5, 1], "#ff3300": [1, 2]}}
                        ),
                        width=4
                    ),
                ]),

                # Additional Information (optional)
                html.Div([
                    html.H5(f"Wind Direction: {wind_direction}", style={"color": "#ffffff"}),
                    html.H5(f"Weather Condition: {weather_condition}", style={"color": "#ffffff"})
                ])
            ])

        else:
            return html.H5(f"No details found for Accident ID: {search_value}", style={"color": "#ff3300"})
    return None

@app.callback(
    Output('accident-sketch', 'figure'),
    Input('accident-dropdown', 'value')
)
def update_sketch(selected_id):
    accident = sample_df[sample_df['ID'] == selected_id].iloc[0]
    
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=[accident['Start_Lat'], accident['End_Lat']],
        lon=[accident['Start_Lng'], accident['End_Lng']],
        mode='markers+lines',
        marker=dict(size=14, color='red'),
        line=dict(width=2, color='blue'),
        name='Accident Path'
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[accident['Start_Lat']],
        lon=[accident['Start_Lng']],
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=[f'Accident {selected_id}: {accident["Description"]}'],
        textposition='top center',
        name='Accident Location'
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=(accident['Start_Lat'] + accident['End_Lat']) / 2,
                        lon=(accident['Start_Lng'] + accident['End_Lng']) / 2),
            zoom=14
        ),
        title=f'Accident Scene - {accident["Description"]}',
        showlegend=False
    )

    return fig
@app.callback(
    [Output('accident-summary', 'children'),
     Output('download-summary', 'data')],
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def search_accident(n_clicks, search_value):
    if n_clicks > 0 and search_value:
        accident_details = df[df['ID'] == search_value]

        if not accident_details.empty:
            details = accident_details.iloc[0]
            # Prepare the report text
            report_text = f"""
ACCIDENT REPORT

Incident ID: {details['ID']}  
Source: {details['Source']}  
Severity: {details['Severity']}  
Incident Duration: From {details['Start_Time']} to {details['End_Time']}  
Location:  
- Start Coordinates: Latitude {details['Start_Lat']}, Longitude {details['Start_Lng']}  
- End Coordinates: Latitude {details['End_Lat']}, Longitude {details['End_Lng']}  
- Distance Covered: {details['Distance(mi)']} miles  

DESCRIPTION:  
The accident occurred on {details['Street']}, in the city of {details['City']}, county of {details['County']}, state of {details['State']}, ZIP code {details['Zipcode']}, country {details['Country']}.  

WEATHER AND CONDITIONS AT THAT TIME:  
- Temperature: {details['Temperature(F)']}°F  
- Wind Chill: {details['Wind_Chill(F)']}°F  
- Humidity: {details['Humidity(%)']}%  
- Pressure: {details['Pressure(in)']} in  
- Visibility: {details['Visibility(mi)']} miles  
- Wind Speed: {details['Wind_Speed(mph)']} mph from the {details['Wind_Direction']}  
- Precipitation: {details['Precipitation(in)']} inches  
- Weather Condition: {details['Weather_Condition']}  
- Timezone: {details['Timezone']}  

NEARBY LANDMARK AND FEATURES:  
- Nearby airport: {details['Airport_Code']}  
- Amenity: {details['Amenity']}  
- Bump: {details['Bump']}  
- Crossing: {details['Crossing']}  
- Give Way: {details['Give_Way']}  
- Junction: {details['Junction']}  
- No Exit: {details['No_Exit']}  
- Railway: {details['Railway']}  
- Roundabout: {details['Roundabout']}  
- Station: {details['Station']}  
- Stop: {details['Stop']}  
- Traffic Calming Measures: {details['Traffic_Calming']}  
- Traffic Signals: {details['Traffic_Signal']}  
- Turning Loop: {details['Turning_Loop']}  

LIGHTING CONDITIONS:  
- Sunrise/Sunset: {details['Sunrise_Sunset']}  
- Civil Twilight: {details['Civil_Twilight']}  
- Nautical Twilight: {details['Nautical_Twilight']}  
- Astronomical Twilight: {details['Astronomical_Twilight']}  

---

SUMMARY:  
An accident with a severity rating of {details['Severity']} occurred on {details['Street']} in {details['City']}, {details['State']}, covering a distance of {details['Distance(mi)']} miles. Weather conditions at the time included {details['Weather_Condition']}, with visibility of {details['Visibility(mi)']} miles and wind speeds of {details['Wind_Speed(mph)']} mph. The accident took place between {details['Start_Time']} and {details['End_Time']} and was reported in the {details['Timezone']} timezone.

The exact causes and circumstances surrounding the accident will require further investigation.
"""

            # Create a downloadable text file with the report
            report_filename = f'accident_report_{search_value}.txt'
            with open(report_filename, 'w') as file:
                file.write(report_text)

            # Return the report and file download
            return [html.Pre(report_text, style={"color": "#ffffff"}), dcc.send_file(report_filename)]
        else:
            return [html.H5(f"No details found for Accident ID: {search_value}", style={"color": "#ff3300"}), None]

    return [None, None]

# Funnel chart callback
@app.callback(
    Output('funnel-chart', 'figure'),
    Input('severity-dropdown', 'value')
)
def update_funnel_chart(selected_severities):
    filtered_df = df_grouped[df_grouped['Severity'].isin(selected_severities)]
    
    # Create the funnel chart
    fig = px.funnel(
        filtered_df,
        x='Counts',
        y='Severity',
        color='Severity',
        color_discrete_map=color_map,
        title='Accident Severity Funnel'
    )
    
    return fig
# Callback for Parallel Coordinates Plot
@app.callback(
    Output("parallel-coordinates", "figure"),
    Input("parallel-coordinates", "id")
)
def update_parallel_coordinates(_):
    fig = px.parallel_coordinates(df_parallel, color='Severity',
                                  dimensions=['Temperature(F)', 'Wind_Speed(mph)', 'Humidity(%)', 'Visibility(mi)'])
    return fig

# Callback for Violin Plot
@app.callback(
    Output("violin-plot", "figure"),
    Input("violin-plot", "id")
)
def update_violin_plot(_):
    fig = px.violin(df_violin, y='Temperature(F)', color='Severity', box=True, points='all')
    return fig


# Callback for Hotspots Map
@app.callback(
    Output("hotspots-map", "figure"),
    Input("hotspots-map", "id")
)
def update_hotspots_map(_):
    # Create the geographical heatmap
    fig = px.density_mapbox(df_hotspots, lat='Start_Lat', lon='Start_Lng', z='Severity',
                            radius=10, center=dict(lat=37.0902, lon=-95.7129), 
                            mapbox_style="open-street-map",
                            title='Accident Hotspots and Severity')
    return fig
# Callback for the Sunburst Chart
@app.callback(
    Output('sunburst-chart', 'figure'),
    [Input('severity-dropdown', 'value')]
)
def update_sunburst_chart(selected_severity):
    # Filter dataset based on selected severity
    filtered_df = df_grouped_sunburst[df_grouped_sunburst['Severity'] == selected_severity]

    # Create sunburst chart
    fig = px.sunburst(
        filtered_df,
        path=['State', 'City'],
        values='Accident_Count',
        color='Accident_Count',
        color_continuous_scale='RdYlBu',
        title=f"Accident Counts by State and City (Severity {selected_severity})"
    )
    
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    return fig
# Callback for Scatter Plot (Severity vs. Distance)
@app.callback(
    Output('severity-distance-scatter', 'figure'),
    Input('severity-distance-scatter', 'id')
)
def update_severity_distance_scatter(_):
    fig = px.scatter(df_filtered_scatter, x='Distance(mi)', y='Severity', color='Weather_Condition',
                     color_continuous_scale='Rainbow',
                     title='Severity vs. Distance of Accidents',
                     labels={'Distance(mi)': 'Distance (miles)', 'Severity': 'Severity'})
    return fig
# Callback for Treemap
@app.callback(
    Output('treemap', 'figure'),
    Input('treemap', 'id')
)
def update_treemap(_):
    # Create the treemap figure
    fig = px.treemap(df_grouped_treemap, path=['Weather_Condition', 'Severity'], values='Count',
                     title='Accident Severity Distribution by Weather Condition')
    return fig
# Callback for Choropleth Map
@app.callback(
    Output("map", "figure"),
    Input("map", "id"),
)
def display_choropleth(_):
    # Use Plotly's built-in geographical data for US states
    fig = px.choropleth(
        df_state_counts,
        locations='State',
        locationmode='USA-states',
        color='Accident_Count',
        color_continuous_scale="Viridis",
        range_color=[0, df_state_counts['Accident_Count'].max()],
        title="Accident Counts by State",
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig
# Callback for Polar Bar Chart
@app.callback(
    Output("graph", "figure"),
    Input("slider", "value")
)
def update_figure(severity_level):
    # Get top 10 cities for the selected severity level
    df_top_cities = get_top_cities_by_severity(df_filtered, severity_level)
    
    # Create polar chart
    fig = px.bar_polar(
        df_top_cities,
        r="Count",
        theta="City",
        color="Count",
        template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Plasma_r
    )
    
    fig.update_layout(title=f"Top 10 Cities for Severity Level {severity_level}")
    return fig
# Callback for Correlation Matrix
@app.callback(
    Output('correlation-matrix-heatmap', 'figure'),
    Input('correlation-matrix-heatmap', 'id')
)
def update_correlation_matrix_heatmap(_):
    # Create the correlation matrix heatmap figure
    fig = px.imshow(correlation_matrix,
                    color_continuous_scale='RdBu',
                    title='Correlation Matrix of Accident Features')
    return fig
# Callback for Bubble Chart
@app.callback(
    Output('bubble-chart', 'figure'),
    Input('severity-slider', 'value')
)
def update_bubble_chart(severity_range):
    # Filter data based on selected severity range
    filtered_data = df_filteredd[df_filteredd['Severity'].between(severity_range[0], severity_range[1])]
    state_summary = filtered_data.groupby('State').agg(
        Accident_Count=('Severity', 'size'),
        Avg_Severity=('Severity', 'mean')
    ).reset_index()

    # Create the bubble chart
    fig = px.scatter(
        state_summary,
        x='State',
        y='Accident_Count',
        size='Accident_Count',
        color='Avg_Severity',
        color_continuous_scale='Viridis',
        labels={'Accident_Count': 'Number of Accidents', 'Avg_Severity': 'Average Severity'},
        title='Accident Counts and Severity by State',
        template='plotly_dark'
    )

    # Update layout for better visual appeal
    fig.update_layout(
        xaxis_title='State',
        yaxis_title='Number of Accidents',
        coloraxis_colorbar=dict(
            title='Average Severity',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1 - Low', '2', '3', '4', '5 - High']
        ),
        xaxis_tickangle=-45
    )

    return fig
@app.callback(
    Output('chat-modal', 'is_open'),
    [Input('open-chat', 'n_clicks'), Input('close-chat', 'n_clicks')],
    [State('chat-modal', 'is_open')]
)
def toggle_chat_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('chat-response', 'children'),
    Input('chat-send', 'n_clicks'),
    State('chat-input', 'value')
)
def handle_chat(n_clicks, chat_input):
    if n_clicks > 0 and chat_input:
        response = run_async_chat(chat_input)
        return html.Div([html.P("Response:"), html.Pre(response)])
    return html.Div([])
if __name__ == '__main__':
    app.run_server(debug=True)
