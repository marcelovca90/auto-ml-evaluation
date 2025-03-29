# A Practical Evaluation of AutoML Tools for Binary, Multiclass, and Multilabel Classification

Authors: *Marcelo Aragão, Augusto Afonso, Rafaela Ferraz, Rairon Ferreira, Sávio Leite, Felipe A. P. de Figueiredo, and Samuel B. Mafra.*

## Abstract:
    Selecting the most suitable Automated Machine Learning (AutoML) tool is pivotal for
    achieving optimal performance in diverse classification tasks, including binary,
    multiclass, and multilabel scenarios. The wide range of frameworks with distinct
    features and capabilities complicates this decision, necessitating a systematic
    evaluation. This study benchmarks sixteen AutoML tools, including AutoGluon,
    AutoSklearn, TPOT, PyCaret, and Lightwood, across all three classification types
    using twenty-one real-world datasets. Unlike prior studies focusing on a subset of
    classification tasks or a limited number of tools, we provide a unified evaluation
    of sixteen frameworks, incorporating feature-based comparisons, time-constrained
    experiments, and multi-tier statistical validation. A key contribution of our study
    is the in-depth assessment of multilabel classification, exploring both native and
    label-powerset representations and revealing that several tools lack robust
    multilabel capabilities. Our findings demonstrate that AutoSklearn excels in
    predictive performance for binary and multiclass settings, albeit at longer training
    times, while Lightwood and AutoKeras offer faster training at the cost of predictive
    performance on complex datasets. AutoGluon emerges as the best overall solution,
    balancing predictive accuracy with computational efficiency. Our statistical
    analysis – at per-dataset, across-datasets, and all-datasets levels – confirms
    significant performance differences among tools, highlighting accuracy-speed
    trade-offs in AutoML. These insights underscore the importance of aligning tool
    selection with specific problem characteristics and resource constraints. The
    open-source code and reproducible experimental protocols further ensure the study’s
    value as a robust resource for researchers and practitioners.

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