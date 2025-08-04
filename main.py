import os
import argparse
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from model import SimplePointNet
from torch.utils.data import TensorDataset, DataLoader

#コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, default=100)
args = parser.parse_args()

#.plyファイルの読み込み関数
def load_points(folder_paths):
    x_data = []
    y_data = []
    
    for label, folder_path in enumerate(folder_paths):
        for filename in os.listdir(folder_path):
            pcd_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            np_pcd = np.asarray(pcd.points) #ply→numpy

            np_pcd = np_pcd - np.mean(np_pcd, axis=0)
            np_pcd = np_pcd/np.max(np.linalg.norm(np_pcd, axis=1))

            x_data.append(np_pcd.astype(np.float32))
            y_data.append(label)

    x_data = np.array(x_data)
    x_tensor = torch.tensor(x_data).permute(0, 2, 1)
    y_tensor = torch.tensor(y_data, dtype=torch.long)

    return x_tensor, y_tensor

#データセットの読み込み
data1 = "data1/"
data2 = "data2/"
data = [data1, data2]

x_data_tensor, y_data_tensor = load_points(data)

#trainデータとtestデータへの分割
x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
    x_data_tensor, y_data_tensor,
    test_size=0.3,
    random_state=1,
    stratify=y_data_tensor
)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=4)

#学習モデルの読み込み
model = SimplePointNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()
iteration = args.iteration

#学習
loss_list=[]

for i in range(iteration):
    model.train() #訓練モードに切り替え
    for xb, yb in train_loader:     
        optimizer.zero_grad() #勾配の初期化
        out = model(xb) #予測実行
        loss = loss_fun(out, yb) #損失の計算
        loss.backward() #誤差逆伝播
        optimizer.step() #パラメータ更新
        
    print(f"iteration {i+1} : Loss = {loss.item():.4f}")
    loss_list.append(loss.item())

#学習の記録
x = np.arange(1, iteration+1)
plt.plot(x, loss_list, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")
plt.close()

#テスト
model.eval()
pred = model(x_test_tensor)
pred_class = torch.argmax(pred, dim=1)

#精度評価
correct = (pred_class==y_test_tensor).sum().item()
total = y_test_tensor.size(0)
accuracy = correct/total
print(f"Accuracy : {accuracy*100:.2f} %")

#混合行列の表示
print(confusion_matrix(y_test_tensor.numpy(), pred_class.numpy()))
