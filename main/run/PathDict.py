# ----------------------------------
# PathDict.py

# Root paths

RootPath = "/home/C00579118/SampleData_CoPAS"
dataroot = "/home/C00579118/SampleData_CoPAS"
CodePath = "/home/C00579118/CoPAS-0812"
DocEvlPath = "/home/C00579118/CoPAS-0812/doctor_eval/"
ExpFolder = "/home/C00579118/CoPAS-0812/experiment_folder/"
pretrain_folder = "/home/C00579118/SampleData_CoPAS/model.pth"

# Dataset configuration
dataset_dict = {
    'Internal': {  # Changed from 'internal' to 'Internal'
        "train_path": f"{dataroot}/Internal",  # Changed from 'train' to 'Internal'
        "train_label": f"{dataroot}/trainLabels.csv",
        
        "val_path": f"{dataroot}/validation",
        "val_label": f"{dataroot}/validationLabels.csv",
        
        "test_path": f"{dataroot}/test",
        "test_label": f"{dataroot}/testLabels.csv",
        
        "cache_path": f"{dataroot}/cache",
        "center_file":   f"{dataroot}/center_file.csv",  # Optional: File related to the center
        "doctor_file":   f"{dataroot}/doctor_file.csv",  # Optional: File related to the doctor           
        
        # Ensure sequence order matches your directories
        "modal": [
            "sag PDW",
            "cor PDW",
            "axi PDW",
            "sag T2WI",
            "cor T1WI"
        ]
    }
}
