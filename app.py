import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])


app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    
    html.Div(id='output-data-upload'),
])


def parse_contents(contents, filename, date):
    # contents = str(contents)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),header=None)
    # try:
    #     if 'csv' in filename:
    #         # Assume that the user uploaded a CSV file
    #         df = pd.read_csv(
    #             io.StringIO(decoded.decode('utf-8')))
    #     elif 'xls' in filename:
    #         # Assume that the user uploaded an excel file
    #         df = pd.read_excel(io.BytesIO(decoded),header= None)
    # except Exception as e:
    #     print(e)
    #     return html.Div([
    #         'There was an error processing this file.'
    #     ])
    if df.shape[1] ==1:
        df.columns=['capa']
    elif df.shape[1] ==2:
        df.columns=['index','capa']
    return filename,df
    


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         filename,df = parse_contents(list_of_contents, list_of_names, list_of_dates)
#         print(df.shape)
#         return html.Div([
#         html.H5(f'Name of file Uploaded: {filename}'),

#         dash_table.DataTable(
#             data=df.to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in df.columns]
#         )
#              ])
#     else: 
#         return html.H6(['No file upload Yet'])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        filename,df = parse_contents(list_of_contents, list_of_names, list_of_dates)
        print(df.shape)
        inputted_req =  df['capa'].tolist()
        dis = pd.DataFrame({'Requirement':inputted_req})
    
        return f'{list_of_names}',dbc.Col(dbc.Table.from_dataframe(dis, striped=True, bordered=True, hover=True),width=5)
        

if __name__ == '__main__':
    app.run_server(debug=True)
