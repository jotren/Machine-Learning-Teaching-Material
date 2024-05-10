import pandas as pd
import ast

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 300)

def attacksHome(attacks):    
    attacksHome = ast.literal_eval(attacks)
    return attacksHome[0]

def attacksAway(attacks):    
    attacksAway = ast.literal_eval(attacks)
    return attacksAway[1]

def shotsOnHome(shotsOn):    
    shotsOnHome = ast.literal_eval(shotsOn)
    return shotsOnHome[0]

def shotsOnAway(shotsOn):    
    shotsOnAway = ast.literal_eval(shotsOn)
    return shotsOnAway[1]

def shotsOffHome(shotsOn):    
    shotsOffHome = ast.literal_eval(shotsOn)
    return shotsOffHome[0]

def shotsOffAway(shotsOn):    
    shotsOffAway = ast.literal_eval(shotsOn)
    return shotsOffAway[1]
        
    
# Read the main DataFrame from the CSV file
df = pd.read_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\total-corner-data.csv')

# Read the league filter DataFrame from the CSV file
league_filter = pd.read_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\league-filter.csv')

# Read the column title DataFrame from the CSV file
column_title = pd.read_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\column-title.csv')

# Extract the 'l_id' values from the league filter DataFrame as a list
filtered_leagues = league_filter['l_id'].tolist()

# Filter the main DataFrame based on the 'l_id' values
filtered_df = df[df['l_id'].isin(filtered_leagues)]

# Get the new column names from the 'new_col_name' column of the column title DataFrame
new_column_names = column_title['new_col_name'].tolist()

# Rename the columns of the filtered DataFrame with the new names
filtered_df.columns = new_column_names

# Create a list of column names where 'usage' is 1
used_column_names = column_title[column_title['usage'] == 1]['new_col_name'].tolist()

# Filter the columns in the filtered DataFrame
filtered_df = filtered_df[used_column_names]

# Implement our functions
filtered_df['shotsOnHome'] = filtered_df['shotsOn'].apply(lambda x: shotsOnHome(x))
filtered_df['shotsOnAway'] = filtered_df['shotsOn'].apply(lambda x: shotsOnAway(x))
filtered_df['attacksHome'] = filtered_df['shotsOn'].apply(lambda x: attacksHome(x))
filtered_df['attacksAway'] = filtered_df['shotsOn'].apply(lambda x: attacksAway(x))
filtered_df['shotsOffHome'] = filtered_df['shotsOn'].apply(lambda x: shotsOffHome(x))
filtered_df['shotsOffAway'] = filtered_df['shotsOn'].apply(lambda x: shotsOffAway(x))

filtered_df.drop(columns=['attacks','shotsOff', 'shotsOn'], inplace=True)

# Assuming you want to sort by the 'column_to_sort_by' column in ascending order
sorted_df = filtered_df.sort_values(by=['status'], ascending=False)

# Print the filtered DataFrame with renamed columns
print(filtered_df)
