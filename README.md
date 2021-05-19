# CREMI Alignment

## Introduction: 
This repo contains tools to reverse the predictions of Synaptic Clefts on [CREMI](https://cremi.org/) Benchmark datasets.

## Requirements:
- You can use one of the following two commands to install the required packages:
```
conda install --yes --file requirements.txt
pip install -r requirements.txt
```

## Usage:

```
python -a <volA-path> -b <volB-path> -c <volC-path> -thres <threshold_val>
```

The outputs can be converted to final submission volumes using [this tool](https://github.com/cremi/cremi_python).

