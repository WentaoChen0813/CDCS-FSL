import os
import shutil


domains = ['real', 'painting', 'sketch', 'clipart']
splits = ['base', 'val', 'novel']
categories = {}
for split in splits:
    categories[split] = []
    with open(split+'.txt') as f:
        for line in f.readlines():
            categories[split].append(line.strip())

for domain in domains:
    for split in splits:
        os.makedirs(os.path.join(domain, split), exist_ok=True)
        for category in categories[split]:
            shutil.move(os.path.join(domain, category), os.path.join(domain, split))