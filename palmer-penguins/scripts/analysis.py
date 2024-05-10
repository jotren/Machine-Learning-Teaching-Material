import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\projects\machine-learning-practise\palmer-penguins\data\penguin_data.csv')

# Define the two specified species you want to include
species_to_include_1 = ['Adelie', 'Gentoo']
species_to_include_2 = ['Adelie', 'Chinstrap']

# Filter the DataFrame to include only the specified species
filtered_df_1 = df[df['species'].isin(species_to_include_1)].dropna()
filtered_df_2 = df[df['species'].isin(species_to_include_2)].dropna()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(df, x='bill_length_mm', y='body_mass_g', z='bill_depth_mm', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')
# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(df, x='bill_length_mm', y='flipper_length_mm', z='bill_depth_mm', color='species', size="island", hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')
# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_1, x='bill_length_mm', y='flipper_length_mm', z='bill_depth_mm', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot~
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_2, x='bill_length_mm', y='flipper_length_mm', z='bill_depth_mm', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_2, x='bill_length_mm', y='flipper_length_mm', z='bill_depth_mm', animation_frame='sex', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_2, x='bill_length_mm', y='body_mass_g', z='bill_depth_mm', animation_frame='sex', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_2, x='bill_length_mm', y='body_mass_g', z='island', animation_frame='sex', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot
fig.show()

# Assuming your DataFrame is named 'df'
fig = px.scatter_3d(filtered_df_2, x='bill_depth_mm', y='body_mass_g', z='island', animation_frame='sex', color='species', hover_data=['island', 'sex'], title='Penguin Bill Length vs Bill Depth Over Time')

# Show the interactive plot
fig.show()
