"""
Target Website:
    [Linux Kernel Documentation](https://www.kernel.org/)
Source: [README](https://www.kernel.org/doc/readme/)
    various README files scattered around Linux kernel source.
robots.txt:
    [robots.txt under the root](https://www.kernel.org/robots.txt) - Gotten 404 Not Found.
"""

import random
from time import sleep
from os import getcwd, path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# from global_constant import HTML_DIR

TEXT_SOURCE_URL = "https://www.kernel.org/doc/readme/"

def __wait__(time=None):
    if time is None:
        sleep(random.uniform(1, 3))
    elif isinstance(time, int):
        if time >= 30:
            for _ in tqdm(range(time)):
                sleep(1)
        else:
            sleep(time)
    elif isinstance(time, list) or isinstance(time, tuple):
        assert isinstance(time[0], int) and isinstance(time[1], int)
        if len(time) != 2:
            raise ValueError("Only accept value like [0, 1] or (0, 1).")
        a, b = min(time), max(time)
        sleep(random.uniform(a, b))

response = requests.get(TEXT_SOURCE_URL, timeout=10)
soup = BeautifulSoup(response.text, features="html.parser")

a_tags = soup.find_all("a")[1:]

for a_tag in tqdm(a_tags):
    name = a_tag.get("href")
    link = urljoin(TEXT_SOURCE_URL, name)
    page_source = requests.get(link, timeout=10).text
    with open(path.join(getcwd(), "data_process", "raw_data", f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(page_source)
    __wait__([3,5])

print("Finished getting page texts.")
