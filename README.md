# CoPAS: learning Co-Plane Attention across MRI Sequences for diagnosing twelve types of knee abnormalities: A multi-center retrospective study

[![Python](https://img.shields.io/badge/Python-3.8.0-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://opensource.org/licenses/Apache-2.0)
<!-- [![DOI]()]()-->


## Introduction
This is the official repository of 

**Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities** 

by
*Zelin Qiu, Zhuoyao Xie, Yanwen Li, Huangjing Lin, Qiang Ye, Menghong Wang, Shisi Li, Yinghua Zhao, and Hao Chen*

### Dataset
We have prepared 50 sample data for test, click [here](https://drive.google.com/drive/folders/1ZG1Cnr6o5u35jumdbWrdw5LC2Je2MD0z?usp=sharing) to download.

#### Data Organization

N. B. Set ``train`` folder name as ``Internal``

```
/path/to/data/
    ├── train/
    │   ├── MR60236/
    │   │   ├── axi PDW/
    │   │   ├── cor PDW/
    │   │   ├── cor T1WI/
    │   │   ├── sag PDW/
    │   │   ├── sag T2WI/
    │   ├── MR87334/
    │   └── ... (35 samples)
    |   ├── validation/
    │   ├── MR53188/
    │   ├── MR65653/
    │   └── ... (8 samples)
    |   ├── test/
    │   ├── MR87764/
    │   ├── MR65651/
    │   └── ... (7 samples)
    ├── train_labels.csv
    ├── test_labels.csv
    ├── validation_labels.csv
```

#### General Structure

```
data_root/
├── train/
│   ├── patient1/
│   │   ├── sag PDW/         # Sagittal PD-weighted sequence
│   │   │   └── [DICOM files]
│   │   ├── cor PDW/         # Coronal PD-weighted sequence
│   │   │   └── [DICOM files]
│   │   ├── axi PDW/         # Axial PD-weighted sequence
│   │   │   └── [DICOM files]
│   │   ├── sag T2WI/        # Sagittal T2-weighted sequence
│   │   │   └── [DICOM files]
│   │   └── cor T1WI/        # Coronal T1-weighted sequence
│   │       └── [DICOM files]
│   └── patient2/
│       └── ...
├── val/
│   └── [same structure as train]
├── test/
│   └── [same structure as train]
├── train_labels.csv
├── val_labels.csv
└── test_labels.csv
```

#### The labels in CSV files

```
patient_id,MENI,ACL,CART,PCL,MCL,LCL,EFFU,CONT,PLICA,CYST,IFP,PR
patient1,1,0,1,0,0,0,1,0,0,1,0,0
patient2,0,1,0,0,1,0,0,1,0,0,0,1
...
```

## Installation Guide:
The code is based on Python 3.8.0

1. Download the repository
```bash
git clone https://github.com/zqiuak/CoPAS
```

2. Go to the `main` folder and install requested libarary.
```bash
cd main
pip install -r requirements.txt
```
Typically, it will take few minutes to complete the installation.


## Run
1. Fill the data path in ```PathDict.py```, the sample is given in the file.
2. Change parameters in ```Args.py``` to fit your data.
#### Run the following command for training:
```bash
python run.py
```

#### Optional:

``cd CoPAS-0812/main``

``python run.py --epochs 2 --batch_size 4 --lr 1e-3 --gpu 1 ``

```
python run.py --epochs 200 --batch_size 4 --lr 1e-3 --gpu 1 --augment True
```

#### Run the following command for testing:

```bash
# python run.py --test --weight_path PATH_TO_WEIGHT
python run.py --test --weight_path /home/C00579118/CoPAS-0812/experiment_folder/Exp-ResNet3D-0305-152920-/model_0.5146_train_auc_0.4874_test_auc_0.4675_epoch_1_bestval.pth
```
##### Output

```
03/05 03:41:48 PM loss 40.0547 | auc 0.4675 | acc 0.5312elapsed time 15.42179560661316 s
```

#### Sample

```
python run.py --test --weight_path /path/to/trained_model.pth
```

#### Results on Sample data

```
03/05 03:31:31 PM Epoch 1
 train loss 4.7876 | train auc 0.4937 | train acc 0.5238 |
 val loss 33.7739 | val auc 0.45 | val acc 0.4286 |
 test loss 37.8361 | test auc 0.3108 | test acc 0.5521 elapsed time 64.27903723716736 s
```

#### Other useful command line arguments:
```--epochs```: Maximum number of epoches in training.<br>
```--batch_size```: Batch size.<br>
```--lr```: Initial learning rate.<br>
```--gpu```: GPU card number.<br>
```--augment```: ```bool```, use augmentation or not.<br>

## To Run the project

Delete ``Cache`` when needed.

Make Executable files,

``chmod +x gpu_manager.py run_with_mem_management.py``

then run,

``python run_with_mem_management.py`` to run with memory management features
or 
If you get GPU memory errors, try:

1. Reduce INPUT_DIM to 160 or 128
2. Reduce SliceNum to 12 or 8
3. Use even smaller batch sizes
4. Disable some augmentations
5. Try using a different GPU (change gpu='1' or '2' in Args.py)



## Request to Authors

If you have any special requests, please send a email to Zelin Qiu (zqiuak@connect.ust.hk).

## License & Citation

This project is covered under the **Apache 2.0 License**.

If you find this work useful, please cite our paper:

```
@article{qiu2024learning,
  title={Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities},
  author={Qiu, Zelin and Xie, Zhuoyao and Lin, Huangjing and Li, Yanwen and Ye, Qiang and Wang, Menghong and Li, Shisi and Zhao, Yinghua and Chen, Hao},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={7637},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## To run on different dataset

``    /CoPAS-0812/main$ python run.py --epochs 2 --batch_size 4 --lr 1e-3 --gpu 1 ``

Bring update in following files and code
1. PathDict.py

```    (path and dataset directory, structure)```

2. dataloader.py or use new dataloader ``kneeDataSetMRNet(data.Dataset)``

```
train_ds = kneeDataSetSITK('train', dataset_name='Internal', transform=Kargs.Augmentor, aug_rate=Kargs.augrate, use_cache=Kargs.use_cache, args=Kargs)
val_ds = kneeDataSetSITK('val', dataset_name='Internal', transform=False, use_cache=Kargs.use_cache, args=Kargs)
test_ds_dict = {dsname:kneeDataSetSITK('test', dataset_name=dsname, transform=False, use_cache=Kargs.use_cache, args=Kargs) for dsname in Kargs.DatasetNameList}

# train_ds = kneeDataSetSITK('train', dataset_name='MRNet', transform=Kargs.Augmentor, aug_rate=Kargs.augrate, use_cache=Kargs.use_cache, args=Kargs)
# val_ds = kneeDataSetSITK('val', dataset_name='MRNet', transform=False, use_cache=Kargs.use_cache, args=Kargs)
# test_ds_dict = {dsname:kneeDataSetSITK('test', dataset_name=dsname, transform=False, use_cache=Kargs.use_cache, args=Kargs) for dsname in Kargs.DatasetNameList}
```

3. model.py

```
        elf.mining_conv = nn.Conv3d(1, 12, (12,12,12)) #(1, 3, (3,3,3))  # For 3 classes instead of 12 (1, 12, (12,12,12)) #line-201
```

4. train.py
   
   ```
        # test_loader = DataLoader(test_ds_dict['Internal'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_ds_dict['MRNet'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
   ```
   
5. Args.py

```
        # ['MENI', 'ACL', 'Abnormal'] 
        # ['MENI', 'ACL', 'CART', 'PCL', 'MCL', 'LCL', 'EFFU', 'CONT', 'PLICA', 'CYST', 'IFP', 'PR']
        self.DiseaseList = ['MENI', 'ACL', 'CART', 'PCL', 'MCL', 'LCL', 'EFFU', 'CONT', 'PLICA', 'CYST', 'IFP', 'PR']
        self.ViewList = ['Sag', 'Cor', 'Axi']
        self.SequenceList = ["sag PDW","cor PDW","axi PDW","sag T2WI","cor T1WI"]
        self.ClassNum = len(self.DiseaseList)
        args.SliceNum = 12 #24  # Set target slices
```

```
        # data_args        
        self.INPUT_DIM = 192 #224 # resolution of model input # Can reduce to 192
        self.MAX_PIXEL_VAL = 255
        # self.MEAN = 58.09
        # self.STDDEV = 49.73
        # Update mean and std for MRNet data
        self.MEAN = 0.5      # Update with actual MRNet statistics
        self.STDDEV = 0.5    # Update with actual MRNet statistics
        self.IMG_R = 576 # origin image resolution
        # self.Spacing = (0.3, 0.3, 3.8)
        self.Spacing = (1.0, 1.0, 1.0)  # MRNet is already preprocessed
        self.SliceNum = 12 #24 # Can Reduce this to 16
        self.Patch_R = 448# patch resolution
        self.Center_Crop = True
        self.ClassDistr = [771, 563, 278, 319, 114, 148, 114, 703, 287, 488, 146, 305, 80] # for Co-PAS, manually set [total, cls1, cls2...]
        # Update distribution for new classes
        # self.ClassDistr = self.calculate_class_distribution()  # For MRNet
        self.cal_class_weight()
        self.Keep_slice = False # for Co-PAS
        # self.Keep_slice = True  # For MRNet # Important for handling variable slices
```

```
# self.active_class = self.active_class = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # For 12 class Co-PAS
self.active_class = [1, 1, 1]  # Updated for 3 classes for MRNet
self.active_branch = [1, 1, 1]
```

```
def calculate_class_distribution(self):
        """Calculate class distribution from the training CSV file"""
        import pandas as pd
        
        try:
            # Get the path to the training labels file
            train_csv = None
            if 'MRNet' in self.DatasetNameList:
                train_csv = dataset_dict['MRNet']['train_label']
            else:
                # Try to find a valid training label file
                for dataset in self.DatasetNameList:
                    if dataset in dataset_dict and 'train_label' in dataset_dict[dataset]:
                        train_csv = dataset_dict[dataset]['train_label']
                        break
            
            # Check if file exists
            if not train_csv or not os.path.exists(train_csv):
                print(f"Warning: Training labels file not found. Using default distribution.")
                # Return a default distribution for 3 classes
                return [100, 42, 11, 81] #[100, 30, 25, 45]  # Default: [total, MENI, ACL, Abnormal]
            
            print(f"Reading training labels from: {train_csv}")
            # Read the CSV file
            df = pd.read_csv(train_csv)
            
            # Calculate distribution
            total_samples = len(df)
            
            # Check column names and adjust accordingly
            meni_positive = df['MENI'].sum() if 'MENI' in df.columns else 0
            acl_positive = df['ACL'].sum() if 'ACL' in df.columns else 0
            abnormal_positive = df['Abnormal'].sum() if 'Abnormal' in df.columns else 0
            
            class_distr = [total_samples, meni_positive, acl_positive, abnormal_positive]
            print(f"Calculated class distribution: {class_distr}")
            return class_distr
```

```
For example, if you have:

100 total samples in training set
35 samples with meniscus tear (MENI=1)
28 samples with ACL tear (ACL=1)
54 samples with abnormalities (Abnormal=1)
ClassDistr would be:
args.ClassDistr = [100, 35, 28, 54]```
