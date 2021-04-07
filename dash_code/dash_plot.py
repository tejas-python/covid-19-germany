import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd
import numpy as np
t = np.linspace(0,200,200)
df_germany= pd.read_csv('./dash_code/Germnay.csv',index_col='Unnamed: 0')
df_grouped= pd.read_csv('./dash_code/grouped.csv')
df_grouped1= pd.read_csv('./dash_code/grouped1.csv')
df_grouped2= pd.read_csv('./dash_code/grouped2.csv')
df_germany1= pd.read_csv('./dash_code/g1.csv',index_col='Unnamed: 0')
result= pd.read_csv('./dash_code/result.csv')
result_reg= pd.read_csv('./dash_code/result_reg')
df_germany = df_germany.drop(columns=['stringency_index'])
def multi_plot_scatter(df, title, addAll = True):
    fig = go.Figure()
    
    df = df.drop(columns=[ 'year','month','date' ],axis=1)
    for column in df.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = df.new_cases,
                y = df[column],
                name = column,
                mode = 'markers',
                
            )
        )

    button_all = dict(label ='ALL',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
           
    )
    
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        xaxis_title="new cases",
        height=800
        
    )
   
    return fig
def multi_plot_bar(df, title, addAll = True):
    fig = go.Figure()
    df = df.set_index('month')
    df = df.drop(columns=[ 'year' ],axis=1)
    for column in df.columns.to_list():
        fig.add_trace(
            go.Bar(
                x = df.index,
                y = df[column],
                name = column
            )
        )

    button_all = dict(label ='ALL',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
           
    )
    
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        xaxis_title="months",
        height=800
        
    )
   
    return fig
def multi_plot(df, title, addAll = True):
    fig = go.Figure()
    df = df.set_index('date')
    df = df.drop(columns=['month', 'year'],axis=1)
    for column in df.columns.to_list():
        fig.add_trace(
            go.Line(
                x = df.index,
                y = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
           
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        xaxis_title="Time",
        height=800
        
    )
   
    return fig
def multi_distplot(df, title, addAll = True):
    fig = go.Figure()
    
    df = df.drop(columns=['month', 'year','date'],axis=1)
    for column in df.columns.to_list():
         fig.add_trace(
        go.Histogram(x=df[column],name=column, )
         )
    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
           
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        
        height=800
        
    )
   
    return fig
data_sir =[] 
data_sir.append(go.Line(x=t,y= result.actual,  name="actual"))
data_sir.append(go.Line(x=t,y= result.predicted,  name="predicted"))
layout_sir = go.Layout(title="Actual vs predicted ouput of SIR")
fig2 = go.Figure(data=data_sir,layout=layout_sir)

data =[] 
data.append(go.Line(x= df_grouped2.date,y=result_reg.predicted,  name="predicted"))
data.append(go.Line(x= df_grouped2.date,y=result_reg.actual,  name="actual"))
layout = go.Layout(title="Actual vs predicted ouput of the multiple regression")
fig1 = go.Figure(data=data,layout=layout)

app = dash.Dash()

app.layout = html.Div(children =[html.H1(children='Covid -19 data analysis for germnay'),
                            html.Div(children='EDA and Model predictions'),
                            dcc.Graph(id='dash_graph5',
                                     figure = multi_distplot(df_germany1, "Histogram plot For the variables")),
                            dcc.Graph(id='dash_graph',
                                     figure = multi_plot(df_germany1, title=" Variation of features with respect to time")),
                            dcc.Graph(id='dash_graph6',
                                     figure = multi_plot_scatter(df_germany1, title=" Variation of features with respect New cases")),
                            dcc.Graph(id='dash_graph4',
                                     figure = multi_plot_bar(df_grouped, title=" BAR plot for each features with respect each month")),
                            html.Div(children='Model development using SIR and Multiple regression'),
                            dcc.Graph(id='dash_graph2',
                                     figure = fig2),
                            dcc.Graph(id='dash_graph7',
                                     figure = multi_distplot(df_germany, "Histogram plot For the variables")),
                            dcc.Graph(id='dash_graph3',
                                     figure = fig1)
                                 
                        ])

if __name__ == '__main__':
    app.run_server()