#%%
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, dash_table, Input, Output, State
from dash.dependencies import State
import base64
import warnings
warnings.filterwarnings('ignore')


fraud_data = pd.read_csv("/Users/JP/Desktop/new_fraudTrain.csv")
fraud_data2 = fraud_data.drop(['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1)

app = Dash(__name__)

app.layout = html.Div([
    html.Button("Help", id='help-button'),
    html.Div(id='help-modal', style={'display': 'none'}, children=[
        html.H2("Dashboard Help"),
        html.P("Welcome to the dashboard! Here's some information to get you started."),
        html.P("This dashboard consists of three scatterplots, each with the ability to narrow down a view of credit "
               "card fraud data via the drop down menu which allows the user to select specific transaction "
               "categories. Once a transaction category is selected, the scatterplot graphs update with data from that "
               "category. The user can then view the graph from a holistic view, or highlight a desired section of the "
               "graph to zoom into specific data points. Once zoomed in, the user has the ability to hover over a data "
               "point at which point information about that data point such as transaction amount, gender, and "
               "fraud/no_fraud will populate. To zoom out the user simply needs to double click on the graph to return "
               "to the original view."),
        html.Button("Close", id='close-help-button')
    ]),
    dcc.Dropdown(
        id="dropdown",
        options=[
            {'label': "Miscellaneous Online Transactions", 'value': "misc_net"},
            {'label': "In Store Grocery Shopping", 'value': "grocery_pos"},
            {'label': "Entertainment", 'value': "entertainment"},
            {'label': "Gas/Transportation Transactions", 'value': "gas_transport"},
            {'label': "In Store Miscellaneous Transactions", 'value': "misc_pos"},
            {'label': "Online Grocery Transactions", 'value': "grocery_net"},
            {'label': "Online Shopping", 'value': "shopping_net"},
            {'label': "In Store Shopping", 'value': "shopping_pos"},
            {'label': "Food/Dining Transactions", 'value': "food_dining"},
            {'label': "Personal Care Transactions", 'value': "personal_care"},
            {'label': "Health and Fitness Transactions", 'value': "health_fitness"},
            {'label': "Travel Transactions", 'value': "travel"},
            {'label': "Kids/Pets Transactions", 'value': "kids_pets"},
            {'label': "Home Related Transactions", 'value': "home"},
        ],
        value="misc_net",
        clearable=False,
    ),
    html.H1('Fraudulent Transactions by Transaction Category'),
    html.H3('Compared to Gender and Amount'),
    dcc.Graph(id="graph"),

    html.H1('Fraudulent Transactions by Transaction Category'),
    html.H3('Compared to Gender and City Population'),
    dcc.Graph(id="graph2"),

    html.H1('Fraudulent Transactions by Transaction Category'),
    dcc.Graph(id="graph3"),
])

# callback for graph 1
@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value"),
)
def update_bar_chart1(selected_category):
    mask = fraud_data2['category'] == selected_category
    fig = px.scatter(fraud_data2[mask], x='amt', y='gender', color='is_fraud', title=f'Fraudulent Transactions in {selected_category}')
    return fig

# app callback for graph 2
@app.callback(
    Output("graph2", "figure"),
    Input("dropdown", "value"),
)
def update_bar_chart2(selected_category):
    mask2 = fraud_data2['category'] == selected_category
    fig2 = px.scatter(fraud_data2[mask2], x='city_pop', y='gender', color='is_fraud', title=f'Fraudulent Transactions in {selected_category}')
    return fig2

# app callback for graph 3
@app.callback(
    Output("graph3", "figure"),
    Input("dropdown", "value"),
)
def update_bar_chart3(selected_category):
    mask3 = fraud_data2['category'] == selected_category
    fig3 = px.scatter(fraud_data2[mask3], x='amt', y='category', color='is_fraud', title=f'Fraudulent Transactions in {selected_category}')
    return fig3

# app callback for the help button
@app.callback(
    Output('help-modal', 'style'),
    Input('help-button', 'n_clicks'),
    Input('close-help-button', 'n_clicks')
)
def display_help_modal(n_clicks_help, n_clicks_close):
    if n_clicks_help is None and n_clicks_close is None:
        return {'display': 'none'}
    if n_clicks_help is not None:
        return {'display': 'block'}
    if n_clicks_close is not None:
        return {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)