# A Practical Evaluation of AutoML Tools for Binary, Multiclass, and Multilabel Classification

Authors: *Marcelo Aragão, Augusto Afonso, Rafaela Ferraz, Rairon Ferreira, Sávio Leite, Felipe A. P. de Figueiredo, and Samuel B. Mafra.*

## Abstract:
Selecting the most suitable Automated Machine Learning (AutoML) tool is pivotal for achieving optimal performance in diverse classification tasks, including binary, multiclass, and multilabel scenarios. The wide range of frameworks with distinct features and capabilities complicates this decision, necessitating systematic evaluation. This study rigorously evaluates sixteen AutoML tools using twenty-one datasets through feature-based comparisons and time-constrained experiments, with weighted $F_1$ score and training time as primary metrics. Both native and label powerset representations were analyzed for multilabel classification to provide a comprehensive understanding of framework performance. The results demonstrate critical trade-offs between accuracy and speed: AutoGluon and AutoKeras performed strongly in binary and multiclass tasks, while AutoSklearn achieved superior accuracy in multilabel classification and AutoKeras excelled in training speed. This work emphasizes the importance of aligning tool selection with problem characteristics by addressing the interplay between task-specific requirements and computational constraints. The study’s open-source code and reproducible experimental protocols ensure its value as a resource for researchers and practitioners. This comprehensive analysis advances the understanding of AutoML capabilities and offers actionable insights to guide tool selection, fostering informed decision-making and future research in the field.

## Setup and Execution:
The tests require a need a Linux installation (bare-metal or virtualized).

```
git clone https://github.com/marcelovca90/auto-ml-evaluation.git
cd auto-ml-evaluation
conda create -n auto-ml-evaluation python=3.8
conda activate auto-ml-evaluation
chmod +x run.sh
./run.sh
```

Note: if you want to use Label Powerset, make sure to set `LABEL_POWERSET = True` in `common.py`.