from multiprocessing import Pool
import os
import sys 
import matplotlib.pyplot as plt 
import subprocess as sub 
import shutil
import numpy as np 
from tqdm import tqdm
import yaml 
import argparse

PATH_TO_RPG = "/home/cm2113/workspace/rpg_trajectory_evaluation"

def run_trajectory_evaluation(args):
    run, imu, path, _sub = args
    if imu:
        tmp_path = f'{path}/data{_sub}/withimu/{run}/'
        align = 'se3'
    else: 
        tmp_path = f'{path}/data{_sub}/withoutimu/{run}/'
        align = 'sim3'
        print(tmp_path)
    cmd =f'python {PATH_TO_RPG}/scripts/analyze_trajectory_single.py {tmp_path} --align_type {align}'
    sub.run(cmd.split())

def main(): 
    parser = argparse.ArgumentParser(description='Compute results for system')
    parser.add_argument('folder', type=str)
    parser.add_argument('sys', type=str)
    parser.add_argument('imu', type=int)
    parser.add_argument('result_path', type=str)
    parser.add_argument('--sub', type=str, default="")
    parser.add_argument('--gt', default='o', type=str)
    args = parser.parse_args()
    compute_results(args.result_path, args.folder, args.sys, args.imu, args.sub, args.gt)
    pass 

class compute_results: 
    def __init__(self, result_path, folder_name, system, imu:bool, _sub:str="", gt:str='o'): 
        """
            folder name is the name of the dataset that it is runing on - this code is sadly only running on DM-VIO so no need to initialize a system name 
        """
        self.sub = _sub
        self.path = f"{result_path}/{folder_name}/{system}"
        self.sys = system
        
        if imu: 
            self.data_folders = sorted(os.listdir(f'{self.path}/data{self.sub}/withimu'))
            #self.ground_truth_path = f'/home/cm2113/workspace/results/{folder_name}/groundtruth_estimate/withimu/stamped_groundtruth.txt'
        else: 
            self.data_folders = sorted(os.listdir(f'{self.path}/data{self.sub}/withoutimu'))
        
        assert gt == 'o' or gt == 't', "--gt has to be 'o' or 't'"

        if gt == 'o': 
            self.ground_truth_path = f'{result_path}/{folder_name}/groundtruth_estimate/withimu/stamped_groundtruth.txt'
        if gt == 't': 
            self.ground_truth_path = f'{result_path}/{folder_name}/groundtruth_estimate/withoutimu/stamped_groundtruth.txt'

        if self.sys == 'dm_vio': 
            file = f'tumvi_{folder_name}_0.txt' if not imu else 'resultScaled.txt'
        else: 
            file = f'f_{folder_name}_0.txt'
        self.imu = imu

        self.scale_error = []
        self.trans_error = []
        
        if len(self.data_folders)==0: 
            print("[EXIT] No data folders exists")
            sys.exit(1)
        
        self._reformat_files(file)
        #self._compute_results()
        
        num_processes = 4
    
        with Pool(num_processes) as p:
            # define the function arguments for each parallel process
            args = [(run, self.imu, self.path, _sub) for run in self.data_folders]
            # use tqdm to display the progress bar
            for _ in tqdm(p.imap_unordered(run_trajectory_evaluation, args), total=len(args)):
                pass
        #self._extract_results()

    def _reformat_files(self, file:str):
        """
            create correctly foramteed files
        """
        #file_g = file
        for i, run in enumerate(self.data_folders):
            if self.imu:
                tmp_path = f'{self.path}/data{self.sub}/withimu/{run}/'
            else: 
                tmp_path = f'{self.path}/data{self.sub}/withoutimu/{run}/'
            
            #files = os.listdir(tmp_path)
            nano=1 if self.sys == 'dm_vio' else int(1e9)
            #file = f'{file_g}_{i}.txt' if self.sys == 'orb_slam' else file_g
            cmd =f'python {PATH_TO_RPG}/scripts/dataset_tools/asl_groundtruth_to_pose.py {tmp_path}{file} --output=stamped_traj_estimate.txt --nano {nano}'
            sub.run(cmd.split())
            shutil.copy(self.ground_truth_path, os.path.join(tmp_path, 'stamped_groundtruth.txt'))

            
    def _compute_results(self): 
        print(f'IMU: {self.imu}')
        for run in tqdm(self.data_folders): 
            if self.imu:
                tmp_path = f'{self.path}/data{self.sub}/withimu/{run}/'
                align = 'se3'
                print(align)
            else: 
                tmp_path = f'{self.path}/data{self.sub}/withoutimu/{run}/'
                align = 'sim3'
                print(tmp_path)

            cmd =f'python {PATH_TO_RPG}/scripts/analyze_trajectory_single.py {tmp_path} --align_type={align}'
            sub.run(cmd.split())

    def _extract_results(self):
        for run in tqdm(sorted(self.data_folders)): 
            if self.imu: 
                folder = f'{self.path}/data{self.sub}withimu/{run}/saved_results/traj_est/absolute_err_statistics_sim3_-1.yaml'
            else:
                folder = f'{self.path}/data{self.sub}/withoutimu/{run}/saved_results/traj_est/absolute_err_statistics_sim3_-1.yaml'
            with open(folder) as f:
                data = yaml.load(f, Loader=yaml.loader.FullLoader)
                self.scale_error.append(data["scale"]['rmse'])
                self.trans_error.append(data['trans']['rmse'])
                
    def compute_result_statistics_scale(self): 
        return np.mean(self.scale_error), np.std(self.scale_error)
    
    def compute_results_statistics_trans(self):
        return np.mean(self.trans_error), np.std(self.trans_error)
            

if __name__=='__main__': 
    main() 
