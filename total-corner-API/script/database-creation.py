import requests
from bs4 import BeautifulSoup
import ast
import pandas as pd
import time

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 300)

def dangerAttacksHome(attacks):
    return attacks[0]

def dangerAttacksAway(attacks):
    return attacks[1]

def attacksHome(attacks):
    return attacks[0]

def attacksAway(attacks):
    return attacks[1]

def shotsOnHome(shotsOn):
    return shotsOn[0]

def shotsOnAway(shotsOn):
    return shotsOn[1]

def shotsOffHome(shotsOn):
    return shotsOn[0]

def shotsOffAway(shotsOn):
    return shotsOn[1]

def fetch_and_append_data():
    API_key = '03fb3c855d094d6e'

    url = 'https://api.totalcorner.com/v1/match/today?token=03fb3c855d094d6e&type=inplay&columns=asian,goalLine,goalLineHalf,attacks,dangerousaAttacks,shotOn,shotOff'

    end_time = time.time() + 8 * 3600  # Run for the next 3 hours
    first_iteration = True  # Track the first iteration

    while time.time() < end_time:
        response = requests.get(url)

        if response.status_code == 200:
            print('Code: 200')
            data = response.json()
            data_data = data['data']
            filtered_df = pd.DataFrame(data_data)
            
            print(filtered_df.columns)

            # league_filter = pd.read_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\league-filter.csv')
            column_title = pd.read_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\column-title.csv')
            
            new_column_names = column_title['new_col_name'].tolist()
            filtered_df.columns = new_column_names
            used_column_names = column_title[column_title['usage'] == 1]['new_col_name'].tolist()
            filtered_df = filtered_df[used_column_names]
            filtered_df['dangerAttacksHome'] = filtered_df['shotsOn'].apply(lambda x: dangerAttacksHome(x))
            filtered_df['dangerAttacksAway'] = filtered_df['shotsOn'].apply(lambda x: dangerAttacksAway(x))
            filtered_df['shotsOnHome'] = filtered_df['shotsOn'].apply(lambda x: shotsOnHome(x))
            filtered_df['shotsOnAway'] = filtered_df['shotsOn'].apply(lambda x: shotsOnAway(x))
            filtered_df['attacksHome'] = filtered_df['shotsOn'].apply(lambda x: attacksHome(x))
            filtered_df['attacksAway'] = filtered_df['shotsOn'].apply(lambda x: attacksAway(x))
            filtered_df['shotsOffHome'] = filtered_df['shotsOn'].apply(lambda x: shotsOffHome(x))
            filtered_df['shotsOffAway'] = filtered_df['shotsOn'].apply(lambda x: shotsOffAway(x))

            filtered_df.drop(columns=['attacks', 'shotsOff', 'shotsOn'], inplace=True)

            print(filtered_df.head())

            # Append data to the CSV file with or without the header based on first_iteration
            sorted_df = filtered_df.sort_values(by=['status'], ascending=False)
            sorted_df.to_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\total-corner-database.csv', index=False, sep=',', mode='a+', header=first_iteration)

            if first_iteration:
                first_iteration = False  # Set to False after the first iteration

            print("Data appended to the CSV file.")
        else:
            print(f"Failed to retrieve data from the API. Status code: {response.status_code}")

        time.sleep(60)  # Sleep for 60 seconds before making the next request

if __name__ == "__main__":
    fetch_and_append_data()


