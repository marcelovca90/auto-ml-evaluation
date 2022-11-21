import json
import matplotlib.pyplot as plt
import os


# plot the best result for each classifier
plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'

for filename in os.listdir("."):
    if filename.endswith('.json'):
        with open(filename) as file:
            content = json.load(file)
            x = filename.replace('automl_', '').replace('.json', '')
            y = content['roc_auc_score']
            plt.bar(x, y)

plt.title('Results')
plt.savefig(f'results.png')
