import json
import matplotlib.pyplot as plt
import os

metric = 'f1_score_weighted'

plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'

for filename in os.listdir("./results"):
    idx = 0
    if filename.endswith('.json'):
        with open(f'results/{filename}') as file:
            content = json.load(file)
            x = filename.replace('automl_', '').replace('.json', '')
            y = content[metric]
            plt.bar(x, y)
            plt.text(x, y+0.01,f'{100*y:.2f}%')
    idx += 1

plt.title(f'Results ({metric})')
plt.savefig(f'./results/automl.png')

