# Time Series Contrastive Clustering (TSCC): An end-to-end unsupervised clustering of microseismic signals
-------------------------------------------------------

## Requirements

The recommended requirements for TS2Vec are specified as follows:
* Python 3.7
* torch==1.8.1
* scipy==1.7.3
* numpy==1.21.6
* pandas==1.3.5
* scikit_learn==0.24.2
* matplotlib==3.5.2
* Bottleneck==1.3.4
* seaborn==0.11.2

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data

Two types of datasets can be obtained from `datasets/` folder:

* [Micrseismic spectrograms] can be found in `datasets/Micrseismic_Spectrograms/` folder.
* [Micrseismic time series] can be found in `datasets/Micrseismic_Timeseries/` folder.

## Usage

To train and evaluate TSCC on a microseismic dataset, run the following command:

```run & evaluate
python run.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |


| Methods | --- | Time series |--- | --- | Feature R |--- |
| --- | ACC (%) | NMI (%) | AUPRC (%) | ACC (%) | NMI (%) | AUPRC (%) |
| Linear | 71.59 | 15.38 | 66.47 | 99.11 | 92.63 | 98.71 |
| KNN | 90.58 | 55.55 | 87.65 | 98.13 | 86.62 | 97.36 |
| SVM | 97.65 | 83.91 | 99.81 | 98.94 | 91.56 | 99.91 |
| TSCC | 98.07 | 86.26 | 97.15 | --- | --- | --- |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 

### You can get the paper from here:
Link:


------------------------------------------------------        
### You can get the training dataset from the



--------------------------------------------------------                        
Reference:              

                                                                                                    
BibTeX:              
                    
                                
------------------------------------------------------
## Abstract:


------------------------------------------------------
The architecture of TSCC used in our study. 

![network architecture](./results/Framework.jpg)

Sampel data. a) and b) are two examples of the seismograms with different polarity of first motion.
c) and d) are examples of local and teleseismic waveforms respectively while e) and f) are the associated Short-Time Fourier transforms. 
Clustering results. 


Visualization of embeded features. 
