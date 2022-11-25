import collections
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from common import *

metric = 'f1_score_weighted'

plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'

plot_data = {}
markdown_data = pd.DataFrame()

for filename in os.listdir(f"./results/{DATASET_REF}"):
    if filename.endswith('.json'):
        with open(f'results/{DATASET_REF}/{filename}') as file:
            framework = filename.replace('automl_', '').replace('.json', '')
            content = json.load(file)
            # bar chart entries
            plot_data.update({framework: content[metric]})
            # markdown table entries
            for k,v in content.items():
                if isinstance(v, float):
                    content.update({k: f'{v:.3f}'})
            markdown_data = markdown_data.append(pd.Series(data=content, name=framework))

# plot bar chart
plot_data = collections.OrderedDict(sorted(plot_data.items()))
for k,v in plot_data.items():
    plt.bar(k, v)
    plt.text(k, v+0.01, f'{v:.03f}')
plt.title(f'Results ({metric})')
plt.savefig(f'./results/{DATASET_REF}/automl.png')

# write csv foçe
with open(f'results/{DATASET_REF}/automl.csv', 'w') as file:
    file.write(f'{markdown_data.sort_index().to_csv(index=False)}\n')

# write markdown table
with open(f'results/{DATASET_REF}/automl.md', 'w') as file:
    file.write(f'{markdown_data.sort_index().to_markdown(index=False, tablefmt="grid")}\n')
