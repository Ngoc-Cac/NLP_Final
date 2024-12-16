import os.path as osp
from pathlib import Path
from content_crawler import crawl_content, NoTextError

# Debug stuff
import logging as lg
from traceback import format_exc


if not osp.exists((path := osp.join('.', 'logs'))):
    Path(path).mkdir()


logger = lg.getLogger('Debug Info')
logger.setLevel(lg.DEBUG)

# create file handler that logs debug and higher level messages
fh = lg.FileHandler(osp.join('.', 'logs', 'debug.log'))
fh.setLevel(lg.DEBUG)

ch = lg.StreamHandler()
ch.setLevel(lg.ERROR)

formatter = lg.Formatter('%(levelname)s:\n\t%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

format_debug = lambda i, url, traceback: f'''Could not crawl
{url}
    at line {i}
{'=' * 20}\n{traceback}\n{'=' * 20}'''

# Crawling
error_path = osp.join('.', 'logs', 'error_links.txt')

topic = 'phap-luat.txt'
with open(osp.join('.', 'links', topic)) as file:
    text = [line.rstrip('\n') for line in file.readlines()]

if not osp.exists((path := osp.join('..', 'Corpus', topic))):
    Path(path).mkdir()

start = 0
end = len(text)
success = 0
for i, url in enumerate(text[start:end], start=start):
    try:
        crawl_content(url, osp.join('..', 'Corpus', topic, f'{success + 1:0>3}.txt'))
        success += 1
    except NoTextError: continue
    except Exception as e:
        traceback = format_exc()
        logger.debug(format_debug(i, url, traceback))
        with open(error_path, 'a') as file:
            file.write(f'{url}\n')
    finally:
        print(f'{success}/{end - start} success crawl')