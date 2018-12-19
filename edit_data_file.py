import numpy as np
import os 
import re

txt_file_dir = "test_configs_2fold_adni60/"
new_txt_file_dir = "data/com_"+txt_file_dir
prefix = "/mnt/HDD/febrian/"
split = "Storage/"


file_list = ['csf_1.txt']
new_file = ['iam_1.txt']
for i,txt in enumerate(file_list):
    with open(new_txt_file_dir+new_file[i],"w") as n:
        with open(new_txt_file_dir+txt,"r") as f:
            while True:
                line = f.readline()
                if not line: break
                new_line = re.sub('CSF_men_cleaned.nii.gz','IAM_GPU_result_def.nii.gz',line)
                n.write(new_line)

'''
if not os.path.exists(new_txt_file_dir):
    os.mkdir(new_txt_file_dir)
file_list = os.listdir(txt_file_dir)

for txt_file_name in file_list:
    print("Read File : "+txt_file_dir+txt_file_name)
    print("New File : "+new_txt_file_dir+txt_file_name)
    with open(new_txt_file_dir+txt_file_name, "w") as new_file:
        with open(txt_file_dir+txt_file_name, "r") as input_file:
            while True:
                line = input_file.readline()
                if not line: break
                if 'name' not in txt_file_name:
                    line = line.split(split)
                    if len(line)<2: break
                    new_line = prefix+line[1]

                    if 'icv' in txt_file_name:
                        new_line = re.sub('ADNI_20x3_2015','ADNI_20x3_2015_Semisupervised',new_line)
                        new_line = re.sub('ICV_cleaned.nii.gz','ICV_cer_cleaned.nii.gz',new_line)
                    if 'label' in txt_file_name:
                        new_line = re.sub('ADNI_20x3_2015','ADNI_20x3_2015_IAM',new_line)
                        new_line = re.sub('WMH_final_nifti_v2.nii.gz','WMH_label.nii.gz',new_line)
                    if 'iam' in txt_file_name:
                        new_line = re.sub('ADNI_20x3_2015/LOTS_IM_icvNotEroded_512s64m','ADNI_20x3_2015_IM_icvNotEroded_512s64m',new_line)
                        

                new_file.write(new_line)
'''            
        
        
        