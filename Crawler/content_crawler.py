import requests

from bs4 import BeautifulSoup

invalid_char = r'<>:"/\|?*'

class NoTextError(Exception):
    """what it says"""

def crawl_content(url: str, save_dir: str) -> None:
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    section = soup.find_all('article', class_='fck_detail')[0]

    article = ''
    for para in section.find_all('p', class_='Normal'):
        article += para.text + '\n'

    if len(article) < 1000: raise NoTextError("No article")

    with open(save_dir, 'w', encoding='utf-8') as txt_file:
        txt_file.write(url + '\n')
        txt_file.write(article)