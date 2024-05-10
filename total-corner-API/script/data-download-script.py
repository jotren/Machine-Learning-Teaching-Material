import requests
from bs4 import BeautifulSoup

import pandas as pd

API_key = '03fb3c855d094d6e'

url = 'https://api.totalcorner.com/v1/match/today?token=03fb3c855d094d6e&type=inplay&columns=asian,goalLine,goalLineHalf,attacks,dangerousaAttacks,shotOn,shotOff'

# Send an HTTP GET request to the API
if 1==1:
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        data_data = data['data']
        
        print(data_data)
        
        df = pd.DataFrame(data_data)

        # Rename the columns as needed
        # df.columns = ['id', 'h', 'h_id', 'l', 'l_id', 'start', 'status', 'hc', 'ac', 'hg', 'ag', 'hrc', 'arc', 'hyc', 'ayc', 'hf_hc', 'hf_ac', 'hf_eg']
        
        # Print the DataFrame
        print(df)
        
        df.to_csv(r'C:\projects\machine-learning-practise\total-corner-API\data\total-corner-data.csv', index=False, sep=',')

    else:
        print(f"Failed to retrieve data from the API. Status code: {response.status_code}")

else:
    
    print('else')

