
import requests


def download_file(url: str, filename_to_save: str):
    r = requests.get(url, allow_redirects=True)
    open(filename_to_save, 'wb').write(r.content)
  
