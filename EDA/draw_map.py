import plotly.express as px
from urllib.request import urlopen
import json
import pandas as pd

def get_interactive_map(target_table):
    # Reading geodata
    with urlopen('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/russia.geojson') as response:
        russian_geodata = json.load(response)

    regions_republic = ['Бурятия', 'Тыва', 'Адыгея', 'Татарстан', 'Марий Эл',
                        'Северная Осетия – Алания', 'Алтай',
                        'Дагестан', 'Ингушетия', 'Башкортостан', 'Северная Осетия - Алания']

    cities = ['Москва', 'Санкт-Петербург']

    # Territory - ID mapping
    mapper = {}

    #For every region
    for regions in russian_geodata["features"]:
        # Insert additional 'id' key into json, which is required by Plotly
        regions['id'] = regions['properties']['cartodb_id']

        # Fix region's names
        if regions['properties']['name'] in regions_republic:
            regions['properties']['name'] = 'Республика ' + regions['properties']['name']

        if regions['properties']['name'] == 'Чувашия':
            regions['properties']['name'] = 'Чувашская республика'

        if regions['properties']['name'] in cities:
            regions['properties']['name'] = 'г.  ' + regions['properties']['name']

        mapper[regions['properties']['name']] = regions['properties']['cartodb_id']

    mapper_df = pd.Series(mapper).sort_index()
    target_to_plot = target_table.query("Территория not in ['г. Севастополь','Республика Крым', 'Российская Федерация']")
    mapper_df.index = target_to_plot['Территория'].sort_values().unique()

    # Add "id" column based on Territory
    target_to_plot['id'] = target_to_plot['Территория'].apply(lambda x: mapper_df[x])

    fig = px.choropleth(target_to_plot,
                        locations='id',
                        geojson=russian_geodata,
                        animation_frame='Год',
                        animation_group='Год',
                        color='НС%',
                        hover_name='Территория',
                        hover_data={'НС': True, 'НС%': True, 'id': False},
                        labels={"НС": "Новые случаи ",
                                "НС%": "Новые случаи (просантимилле) "},
                        color_continuous_scale="Oranges",
                        range_color=[0, 250]
                        )

    fig.update_layout(hoverlabel={'bgcolor': "white",
                                  'font_size': 12,
                                  'font_family': "Helvetica"},

                      margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      coloraxis_colorbar_title='НС%'
                      )

    fig.update_geos(projection_type='mercator',
                    projection_scale=1,
                    lataxis_range=[50, 80],
                    lonaxis_range=[19, 190],
                    visible=False)

    fig["layout"].pop("updatemenus")
    fig.update_traces(marker_line_width=1)

    return fig

