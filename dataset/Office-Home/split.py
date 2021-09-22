import os
import shutil


os.rename('Real World', 'real')
os.rename('Art', 'art')
os.rename('Clipart', 'clipart')
os.rename('Product', 'product')

domains = ['real', 'art', 'clipart', 'product']
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