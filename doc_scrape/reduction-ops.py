from glob import glob
import urllib.request
from datetime import datetime
from copy import deepcopy

import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

fp = urllib.request.urlopen("https://pytorch.org/docs/stable/torch.html")
html_doc = fp.read().decode("utf8")
fp.close()

soup = BeautifulSoup(html_doc, 'html.parser')

# scrape and validate pointwise ops
pwo_section = soup.find("section", {"id": "reduction-ops"})
pwo_rows = pwo_section.find_all("tr")
pwo_names = [r.find("td").text for r in pwo_rows]
