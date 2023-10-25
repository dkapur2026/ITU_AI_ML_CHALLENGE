# -*- coding: utf-8 -*-
# """
# Created on Fri Aug 25 11:19:53 2023

# @author: DhruvKapur
# """

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
ituData = pd.read_parquet("C:/Users/DhruvKapur/Downloads/cellular_dataframe.parquet")

ituData['PCell_SNR_max'] = ituData[['PCell_SNR_1', 'PCell_SNR_2']].max(axis=1)
ituData['SCell_SNR_max'] = ituData[['SCell_SNR_1', 'SCell_SNR_2']].max(axis=1)
ituData["uplink"] = ituData['datarate'].where(ituData['scenario'] == 'A3U')
ituData["downlink"] = ituData['datarate'].where(ituData['scenario'] == 'A3D')/1e6



ituData.dropna(subset=[
    #PCELL:
    'PCell_SNR_max', 'PCell_RSRP_max', 'PCell_RSRQ_max', 'PCell_RSSI_max',
    'PCell_Downlink_Average_MCS', "PCell_freq_MHz",
    'PCell_Downlink_bandwidth_MHz','PCell_Downlink_frequency',
    #SCELL:
    'SCell_SNR_max', 'SCell_RSRP_max', 'SCell_RSRQ_max', 'SCell_RSSI_max',
    'SCell_Downlink_Average_MCS', "SCell_freq_MHz", 
    'SCell_Downlink_bandwidth_MHz', 'SCell_Downlink_frequency',
    
    #MISC:
    'ping_ms', 'Altitude', 'jitter', 'humidity', 'cloudCover', 
    'Traffic Jam Factor', 'Traffic Distance', 'Pos in Ref Round' 

                       ,'downlink'], how='any', inplace=True)
    
ituData['PCell_Downlink_bandwidth_MHz'].astype(np.int32)
ituData['PCell_Downlink_frequency'].astype(np.int32)
ituData['SCell_Downlink_bandwidth_MHz'].astype(np.int32)
ituData['SCell_Downlink_frequency'].astype(np.int32)
X = ituData[[
    #PCELL:
    'PCell_SNR_max', 'PCell_RSRP_max', 'PCell_RSRQ_max', 'PCell_RSSI_max',
    'PCell_Downlink_Average_MCS', "PCell_freq_MHz",
    'PCell_Downlink_bandwidth_MHz',
    'PCell_Downlink_frequency',
    #SCELL:
    'SCell_SNR_max', 'SCell_RSRP_max', 'SCell_RSRQ_max', 'SCell_RSSI_max',
    'SCell_Downlink_Average_MCS', "SCell_freq_MHz", 
    'SCell_Downlink_bandwidth_MHz', 
    'SCell_Downlink_frequency',
    
    #MISC:
    'ping_ms', 'Altitude', 'jitter', 'humidity', 'cloudCover', 
    'Traffic Jam Factor', 'Traffic Distance', 'Pos in Ref Round' 

    ]].values
X = np.array(X, dtype=np.float32)
y = ituData['downlink'].values

variablesTested = "24var"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pd.DataFrame(X_train).to_csv('Train_Set_'+variablesTested+'.csv')
pd.DataFrame(X_test).to_csv('Test_Set_'+variablesTested+'.csv')
pd.DataFrame(y_train).to_csv('Train_Set_'+variablesTested+'_downlink.csv')
pd.DataFrame(y_test).to_csv('Test_Set_'+variablesTested+'_downlink.csv')

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 24  
hidden_size = 64  # Number of neurons in the hidden layer

model = NeuralNet(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
total_samples = X_train.shape[0]
num_batches = int(np.ceil(total_samples / batch_size))

for epoch in range(num_epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        # Reshape batch_y to have the shape [batch_size, 1]
        batch_y = batch_y.view(-1, 1)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    
torch.save(model, "NeuralNetwork_"+variablesTested+".pkl")
y_pred = y_pred.numpy()
y_test = y_test.numpy()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-Squared Value: {r2}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
