# Domain Adaptation Based Pipeline for Character Classification and Handwritten Text Recognition

This repository contains the official implementation of our unsupervised domain adaptation pipeline for character classification and historical handwritten text recognition tasks.

### Key Features

- Three-stage training pipeline for robust domain adaptation
- State-of-the-art performance on Safran-MNIST-DLS dataset (76.17% Macro F1-score)
- Competitive results on READ 2018 datasets without target domain labels
- Increased computation time only during training, not inference
- Flexible framework applicable to both character-level and line-level recognition tasks

## Method

Our pipeline consists of three main steps:

1. **Pretraining**: Train an expert model on source data to learn strong general representations
2. **Unsupervised Domain Adaptation**: Leverage unlabeled target data to adapt the model to target domain characteristics
3. **Style Transfer Training**: Train the expert model on source data transformed to target style

By aligning feature distributions between source and target domains, the model generalizes effectively across diverse datasets without requiring labeled target data.

## Citation

If you find this work useful, please cite our paper:

    Reference article TBA
    ```
    @misc{TBA,
      title={Domain Adaptation Based Pipeline for Character Classification and Handwritten Text Recognition}, 
      author={TBA},
      year={TBA},
      eprint={TBA},
      archivePrefix={TBA},
      primaryClass={TBA},
      url={TBA}, 
    }
    ```


## Installation
CPU
```
pip install -r requirements.txt
```

For GPU install torch for GPU 

## Data
In order to reproduce the experiments described in the article, you can download the following data:  
- Safran-MNIST-DLS Dataset: https://zenodo.org/records/13321202
- Read2018 Dataset: https://zenodo.org/records/1442182

And format the databases with:
- EMNIST:
```
src\data\format_db\format_emnist.py
```
- DAGECC data:
```
src\data\format_db\format_dagecc_db.py
```
- Synthetic data DAGECC
```
src\data\image\synthesis\char_dagecc_style.py
```
- Read 2018
```
src\datautils\augmentation\generate_synthetic_data.py
```

Define paths in config file, cf. example in directory configuration

Can merge several training sets into one 

Multiple val and test sets

Alphabet defined in charset file, cf. src/data/text/charset_util.py to create one

Case multiple DB, all charsets are merged


### Character Classification
One file per db, with multiple img + label

### Handwriting Recognition Line level
One image per line

One label file for all

To gather labels into one json file: src\data\text\gather_labels_one_file.py

## Usage

### Character Classification

**Step 1:** Pretrain on source data

demo: /src/train/supervised/train_cnn.py

demo config: config/config_demo_character_classification_cpu.json

Note: 
- Update with EMNIST data (digits and upper letters) and synthetic data
- Specify full paths

Launch with parameters:
- Config file path  
- Directory to save logs


**Steps 2 & 3:** Domain Adaptation and Style Transfer

demo:  /src/train/domain_adaptation/train_DAGECC.py

demo config: config/config_demo_character_classification_da_cpu.json

Note: 
- Update with EMNIST data (digits and upper letters), synthetic data and DAGECC data
- Specify full paths

Launch with parameters:
- Config file path 
- Path model pretrained in step 1

### Handwriting Recognition Line level

**Step 1:** Pretrain on source data: /src/train/supervised/train_crnn.py

demo config: config/config_demo_htr_cpu.json

Note: update with all general data path

Launch with parameters:
- Config file path  
- Directory to save logs


**Steps 2 & 3:** Domain Adaptation and Style Transfer

demo:  /src/train/domain_adaptation/train_READ.py

demo config: config/config_demo_htr_da_cpu.json

Note: 
- update with READ data (source and target)
- specify full paths

Launch with parameters:
- config file path 
- path model pretrained in step 1
--dataset_source read_2018_general 
--dataset_target example: read_2018_30866 

## Models used in this work

- CRNN from: https://github.com/georgeretsi/HTR-best-practices/

    src/train/train_crnn.py: train, val and test CRNN

    Reference article CRNN
    ```
    @inproceedings{retsinas2022best,
      title={Best practices for a handwritten text recognition system},
      author={Retsinas, George and Sfikas, Giorgos and Gatos, Basilis and Nikou, Christophoros},
      booktitle={International Workshop on Document Analysis Systems},
      pages={247--259},
      year={2022},
      organization={Springer}
    }
    ```

- DRANet from: https://github.com/FIorentI/DRANet-pytorch

    Reference article DRANet
    ```
    @misc{lee2021dranetdisentanglingrepresentationadaptation,
      title={DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation}, 
      author={Seunghun Lee and Sunghyun Cho and Sunghoon Im},
      year={2021},
      eprint={2103.13447},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.13447}, 
    }
    ```
