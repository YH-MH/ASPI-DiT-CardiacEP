import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import re

class OrdDataset(Dataset):
    def __init__(self, data_root, phase = 'train1', sigma = 0.0, factor = None):
        print()
        print(f"OrdDataset {phase} phase")
        print(f"data_root = {data_root}")
        numbers = re.findall(r'\d+', phase)
        if numbers:
            data_version = int(numbers[0])
        else:
            data_version = 4
        print(f"dataset version = {data_version}")
        
        # load all factors to dict in memory
        self.factors = np.load(os.path.join(data_root, phase, "factors", "merged_factors.npy"))
        # factor 预处理
        if "train" not in phase :
            train_factors = np.load(os.path.join(data_root, f"train{data_version}", "factors", "merged_factors.npy"))
        else:
            train_factors = self.factors
        print(f"self.factors[0:2] = {self.factors[0:2]}")
        self.factors_scaler = StandardScaler().fit(train_factors)
        self.factors_min = self.factors.min()
        self.factors_max = self.factors.max()
        self.factors = self.factors_scaler.transform(self.factors)
        self.factors = torch.from_numpy(self.factors).to(torch.float)

        # load all ap file to dict in memory
        self.apdatas = np.load(os.path.join(data_root, phase, "apdatas", "merged_apdatas.npy"))
        # apdatas 预处理
        if "train" not in phase :
            self.unNorm_ap = self.apdatas
        else:
            self.unNorm_ap = None
        # 初次调换B 和 T维度
        self.apdatas = self.apdatas.transpose(1, 0)
        self.apdatas_scaler = StandardScaler().fit(self.apdatas)
        print(f"the shape of apdatas_scaler's mean_ = {self.apdatas_scaler.mean_.shape}")
        print(f"the shape of apdatas_scaler's var_ = {self.apdatas_scaler.var_.shape}")
        self.apdatas = self.apdatas_scaler.transform(self.apdatas)
        # 调换回B 和 T维度
        self.apdatas = self.apdatas.transpose(1, 0)
        
        # 添加不同比例的正态噪声
        if "test" in phase and sigma > 0.0:
            print(f"the shape of self.apdatas = {self.apdatas.shape} in normal")
            # index = 21
            # plt.figure(figsize=(10, 4))
            # plt.plot( self.apdatas[index], label="Original", alpha=0.8)
            self.apdatas = self.apdatas + (np.random.randn(*self.apdatas.shape) * sigma)
            # plt.plot( self.apdatas[index], label="Noisy", alpha=0.8)
            # plt.legend()
            # plt.title(f"Sample {index} Before and After Noise")
            # plt.xlabel("Feature Index")
            # plt.ylabel("Value")
            # plt.show()

        self.apdatas = np.expand_dims(self.apdatas, 2)
        self.apdatas = torch.from_numpy(self.apdatas).to(torch.float)

        # load all biomarks file to dict in memory
        self.biomarks = np.load(os.path.join(data_root, phase, "biomarks", "biomarks.npy"))
        _, _, C = self.biomarks.shape
        self.biomarks = self.biomarks.reshape(-1, C)
        # biomarks预处理
        if "train" not in phase :
            train_biomarks = np.load(os.path.join(data_root, f"train{data_version}", "biomarks", "biomarks.npy")).reshape(-1, C)
        else:
            train_biomarks = self.biomarks
        self.biomarks_scaler = StandardScaler().fit(train_biomarks)
        self.biomarks = self.biomarks_scaler.transform(self.biomarks)
        print(self.biomarks[0], self.biomarks[1])
        self.biomarks = self.biomarks.reshape(-1, 2*C)
        print(self.biomarks[0])
        self.biomarks = torch.from_numpy(self.biomarks).to(torch.float)
        

        print(f"the shape of self.factors = {self.factors.shape}")
        print(f"the shape of self.apdatas = {self.apdatas.shape}")
        if self.unNorm_ap is not None:
            print(f"the shape of self.unNorm_ap = {self.unNorm_ap.shape}")
        else:
            print(f"self.unNorm_ap = {self.unNorm_ap}")
        print(f"the shape of self.biomarks = {self.biomarks.shape}")

        self.num = self.apdatas.shape[0]

        print("OrdDataset 初始化结束")
        print()

    def inverse_transform_factors(self, scaled_data):
        assert len(scaled_data.shape) == 2
        return self.factors_scaler.inverse_transform(scaled_data)
    
    def inverse_transform_apdatas(self, scaled_data):
        assert len(scaled_data.shape) == 2
        return self.apdatas_scaler.inverse_transform(scaled_data)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, item):
        data = {
                'factors': self.factors[item % self.num],      # 9
                'ap_data': self.apdatas[item % self.num],      # 2000 * 1
                'unNorm_ap': self.unNorm_ap[item % self.num] if self.unNorm_ap is not None else -1,      # 2000
                'biomarks': self.biomarks[item % self.num],    # 16 + 16
        }
        return data
    

if __name__ == '__main__':
    pass