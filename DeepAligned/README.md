# Discovering New Intents with Deep Aligned Clustering

An alignment strategy for deep clustering to discover new intents. 

## Introduction
This repository provides the PyTorch implementation of the research paper [Discovering New Intents with Deep Aligned Clustering](https://xxx) (**Accepted by [AAAI2021](https://xxx)**).

If you are instrested in this work, please star this repository and cite by:
```
@inproceedings{xxx,
	title =	    {Discovering New Intents with DeepAligned Clustering},
	author =    {Zhang, Hanlei and Xu, Hua and Lin, Ting-En and Lv Rui},
	booktitle = {Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI)},
	year =      {2021},
	url =       xxx
}
```

## Usage
Install all required library
```
pip install -r requirements.txt
```
Get the pre-trained [BERT](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model and convert it into [Pytorch](https://huggingface.co/transformers/converting_tensorflow_models.html) 

Run the experiments by: 
```
sh run.sh
```
or
```
python DeepAligned.py --dataset clinc --fraction_of_clusters 1 --known_class_ratio 0.75
```
Selected Parameters
```
dataset: clinc | banking
factor_of_clusters: 1 (default) | 2 | 3 | 4 
known_class_ratio: 0.25 | 0.5 | 0.75 (default)
```



##  Results
### Main experiments
| Method   |       | CLINC |       |       |BANKING|       | 
|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Method   |  NMI  |  ARI  |  ACC  |  NMI  |  ARI  |  ACC  | 
| KM       | 71.42 | 67.62 | 84.36 | 67.26 | 49.93 | 61.00 | 
| AG       | 71.03 | 58.52 | 75.54 | 65.63 | 43.92 | 56.07 | 
| SAE-KM   | 78.24 | 74.66 | 87.88 | 59.70 | 31.72 | 50.29 | 
| DEC      | 84.62 | 82.32 | 91.59 | 53.36 | 29.43 | 39.60 | 
| DCN      | 58.64 | 42.81 | 57.45 | 54.54 | 32.31 | 47.48 | 
| DAC      | 79.97 | 69.17 | 76.29 | 75.37 | 56.30 | 63.96 | 
| BERT-KM  | 52.11 | 43.73 | 70.29 | 60.87 | 26.6  | 36.14 |
| PCK-means| 74.85 | 71.87 | 86.92 | 79.76 | 71.27 | 83.11 | 
| BERT-KCL | 75.16 | 61.90 | 63.88 | 83.16 | 61.03 | 60.62 | 
| BERT-Semi| 75.95 | 69.08 | 78.00 | 86.35 | 72.49 | 75.31 | 
| CDAC+    | __89.30__ | __86.82__ | __93.63__ | __94.74__ | __89.41__ | __91.66__ | 

### Ablation study
| Method   |       | SNIPS |       |       |DBPedia|       |   
|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Method   |  NMI  |  ARI  |  ACC  |  NMI  |  ARI  |  ACC  |
| DAC      | 79.97 | 69.17 | 76.29 | 75.37 | 56.30 | 63.96 | 
| DAC-KM   | 86.29 | 82.58 | 91.27 | 84.79 | 74.46 | 82.14 | 
| DAC+     | 86.90 | 83.15 | 91.41 | 86.03 | 75.99 | 82.88 | 
| CDAC     | 77.57 | 67.35 | 74.93 | 80.04 | 61.69 | 69.01 | 
| CDAC-KM  | 87.96 | 85.11 | 93.03 | 93.42 | 87.55 | 89.77 | 
| CDAC+    | __89.30__ | __86.82__ | __93.63__ | __94.74__ | __89.41__ | __91.66__ | 



### Acknowledgments
This work was supported by xxx
