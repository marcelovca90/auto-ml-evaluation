import json
import matplotlib.pyplot as plt
import os
import pandas as pd

metric = 'f1_score_weighted'

plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'

indices = []
df = pd.DataFrame()

for filename in os.listdir("./results"):
    idx = 0
    if filename.endswith('.json'):
        with open(f'results/{filename}') as file:
            framework = filename.replace('automl_', '').replace('.json', '')
            content = json.load(file)
            x = framework
            y = content[metric]
            # bar plot entries
            plt.bar(x, y)
            plt.text(x, y+0.01, f'{y:.03f}')
            # markdown table entries
            for k,v in content.items():
                if isinstance(v, float):
                    content.update({k: f'{v:.3f}'})
            df = df.append(pd.Series(data=content, name=framework))
    idx += 1

plt.title(f'Results ({metric})')
plt.savefig(f'./results/automl.png')

with open(f'results/automl.md', 'w') as file:
    file.write(f'{df.sort_index().to_markdown(tablefmt="grid")}\n')
