# Export Toggl time entries from web API, storing in a CSV file for model use
import os
import csv
import math
import time
import requests


# Global variables
project_path = os.getcwd()
key_path = os.path.join(project_path, 'keys/')
data_path = os.path.join(project_path, 'data/')

api_token = ''
headers = {'Content-Type': 'application/json'}
workspace_url = 'https://toggl.com/api/v8/workspaces'
details_url = 'https://toggl.com/reports/api/v2/details'


# Make HTTP request against Toggl web endpoint
def api_request(url, api_token, payload):
    res = requests.get(url, auth=(api_token, 'api_token'), headers=headers,
                       params=payload)
    return res


# Save all time entries page by page to file
def write_csv(file, payload, pages):
    writer = csv.writer(file)

    # Data header
    writer.writerow(['project', 'description', 'tags', 'start', 'end',
                     'updated', 'duration'])

    # Data body
    for i in range(1, pages + 1):

        # Wait 1 second to avoid rate-limiting
        time.sleep(1)

        # Retrieve next page of entry data
        payload['page'] = i
        res = api_request(details_url, api_token, payload)
        data = res.json()['data']
        print(f'Writing page #{i} to data.csv')

        # Write current 50 entries to .csv file
        for entry in data:
            writer.writerow([entry['project'], entry['description'],
                entry['tags'][0] if entry['tags'] else 'None', entry['start'],
                entry['end'], entry['updated'], entry['dur']])


# Get all time entries between passed dates from web API
def export(since, until):
    global api_token

    # Open key files for API access
    with open(os.path.join(key_path, 'email.key')) as email, \
         open(os.path.join(key_path, 'api_token.key')) as api_token:
        email = email.read()
        api_token = api_token.read()
    
    print('Retrieving time entries for account: ', email)

    # Find workspace ID number
    res = api_request(workspace_url, api_token, {})
    workspace_id = res.json()[0]['id']

    # Determine number of pages in detail view
    payload = {'user_agent': email, 'workspace_id': workspace_id, 
               'since': since, 'until': until, 'page': 1}
    res = api_request(details_url, api_token, payload)
    pages = math.ceil(res.json()['total_count'] / 50)
    print(f"Found {res.json()['total_count']} entries on {pages} pages")

    # Collect all records in .csv format
    with open(os.path.join(data_path, 'data.csv'), 'w') as file:
        write_csv(file, payload, pages)
    
    print('Finished exporting data')


# DEBUG
if __name__ == '__main__':
    export('2018-01-01', '2018-12-31')
