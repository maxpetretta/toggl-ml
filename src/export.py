# Export Toggl time entries from web API, storing in a CSV file for learning model use
import os
import csv
import math
import time
import requests

project_path = os.getcwd()
key_path = os.path.join(project_path, 'keys/')
headers = {'Content-Type': 'application/json'}

# Open key files for API access
with open(os.path.join(key_path, 'email.key')) as email, \
     open(os.path.join(key_path, 'api_token.key')) as api_token:
    email = email.read()
    api_token = api_token.read()

    print("Retrieving time entries for account: ", email)

    # Find workspace ID number
    url = 'https://toggl.com/api/v8/workspaces'
    req = requests.get(url, auth=(api_token, 'api_token'), headers=headers)
    workspace_id = req.json()[0]['id']

    # Determine number of pages in detail view
    url = 'https://toggl.com/reports/api/v2/details'
    payload = {'user_agent': email, 'workspace_id': workspace_id, 
               'since': '2018-01-01', 'until': '2018-12-31', 'page': 1}
    req = requests.get(url, auth=(api_token, 'api_token'), headers=headers,
                     params=payload)

    pages = math.ceil(req.json()['total_count'] / 50)

    print(f"Found {req.json()['total_count']} entries on {pages} pages")

    # Collect all records in .csv format
    with open(os.path.join(project_path, 'data/data.csv'), 'w') as file:
        writer = csv.writer(file)

        # Data header
        writer.writerow(['project', 'description', 'tags', 'start', 'end',
                         'updated', 'duration'])

        # Data body
        for i in range(1, pages + 1):

            # Wait 1 second to avoid rate-limiting
            time.sleep(1)

            # Retrieve next page of entry data
            payload = {'user_agent': email, 'workspace_id': workspace_id,
                       'since': '2018-01-01', 'until': '2018-12-31', 'page': i}
            req = requests.get(url, auth=(api_token, 'api_token'),
                             headers=headers, params=payload)
            data = req.json()['data']
            
            print(f'Writing page #{i} to data.csv')

            # Write current 50 entries to .csv file
            for entry in data:
                writer.writerow([entry['project'], entry['description'],
                    entry['tags'], entry['start'], entry['end'],
                    entry['updated'], entry['dur']])
        print('Finished exporting data')