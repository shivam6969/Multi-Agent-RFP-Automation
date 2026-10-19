import json
import os

catalog_path = os.path.join('extracted_data', 'rfps_catalog.json')
print(f'Catalog path: {catalog_path}')
print(f'File exists: {os.path.exists(catalog_path)}')

if os.path.exists(catalog_path):
    with open(catalog_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} RFPs')
    for i, rfp in enumerate(data):
        print(f'  {i+1}. {rfp["bid_title"]} (status: {rfp["status"]})')
else:
    print('Catalog file not found!')
