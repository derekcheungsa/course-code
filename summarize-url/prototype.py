# Python script to search for github repos

import requests
from bs4 import BeautifulSoup

url = 'https://github.com/trending'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

repos = soup.find_all('h1', class_='h3 lh-condensed')

for repo in repos:
    print(repo.text.strip())# Python script to search for github repos

import requests
from bs4 import BeautifulSoup

url = 'https://github.com/trending'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

repos = soup.find_all('h1', class_='h3 lh-condensed')

for repo in repos:
    print(repo.text.strip())