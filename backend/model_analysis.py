import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

class KELM:
    """Kernel Extreme Learning Machine"""
    def __init__(self, kernel='rbf', gamma=1.0):
        self.kernel = kernel
        self.gamma = gamma
        
    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        K = self._kernel_matrix(X, X)
        self.beta = np.linalg.solve(K + np.eye(n_samples), y)
        
    def predict(self, X):
        K = self._kernel_matrix(X, self.X_train)
        return K.dot(self.beta)
        
    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'rbf':
            dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1.dot(X2.T)
            return np.exp(-self.gamma * dist)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback:idx+self.lookback+self.horizon]
        return x, y

class RNNCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 3, padding=1)
        )
        self.fc = nn.Linear(16 * 5, 1)
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        cnn_out = self.cnn(rnn_out.transpose(1, 2))
        return self.fc(cnn_out.flatten(1))

class DeepCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 3, padding=1)
        )
        self.fc = nn.Linear(16 * 5, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        return self.fc(cnn_out.flatten(1))

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, nhead),
            num_layers=3
        )
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        transformer_out = self.transformer(x)
        return self.fc(transformer_out.mean(1))

class MultiStepPredictor:
    def __init__(self, lookback=10, horizons=[1,3,5,10]):
        self.lookback = lookback
        self.horizons = horizons
        self.scaler = StandardScaler()
        
        # 基础模型
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=100),
            'kelm': KELM(),
            'xgb': XGBRegressor(),
            'rnn_cnn': None,
            'deep_cnn': None,
            'transformer': None
        }
        
        # 元模型
        self.meta_models = {
            'logistic': LogisticRegression(),
            'gdboost': GradientBoostingClassifier(),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50))
        }
        
        self.best_meta_model = None
        
    def prepare_data(self, df):
        features = df.drop(['Date'], axis=1).values
        features = self.scaler.fit_transform(features)
        
        X, y = {}, {}
        for horizon in self.horizons:
            X[horizon] = []
            y[horizon] = []
            for i in range(len(features) - self.lookback - horizon + 1):
                X[horizon].append(features[i:i+self.lookback])
                y[horizon].append(features[i+self.lookback:i+self.lookback+horizon])
                
        return {h: (np.array(X[h]), np.array(y[h])) for h in self.horizons}
        
    def train_deep_models(self, X, y, device='cuda'):
        input_dim = X.shape[2]
        
        self.base_models['rnn_cnn'] = RNNCNN(input_dim).to(device)
        self.base_models['deep_cnn'] = DeepCNN(input_dim).to(device)
        self.base_models['transformer'] = TransformerModel(input_dim).to(device)
        
        dataset = TimeSeriesDataset(X, self.lookback, max(self.horizons))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for name in ['rnn_cnn', 'deep_cnn', 'transformer']:
            model = self.base_models[name]
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            
            for epoch in range(10):
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    
    def generate_signals(self, predictions, volatility_threshold=1.0):
        signals = np.zeros(len(predictions))
        volatility = np.std(predictions, axis=1)
        
        for i in range(len(predictions)):
            pred_mean = np.mean(predictions[i])
            if pred_mean > volatility[i] * volatility_threshold:
                signals[i] = 1  # 买入
            elif pred_mean < -volatility[i] * volatility_threshold:
                signals[i] = -1  # 卖出
                
        return signals
        
    def select_best_meta_model(self, meta_predictions, true_returns):
        best_score = float('-inf')
        best_model = None
        
        for name, model in self.meta_models.items():
            model.fit(meta_predictions, np.sign(true_returns))
            score = model.score(meta_predictions, np.sign(true_returns))
            
            if score > best_score:
                best_score = score
                best_model = name
                
        return best_model
        
    def train_and_predict(self, df):
        data_dict = self.prepare_data(df)
        all_predictions = []
        
        # 对每个预测步长训练和预测
        for horizon in self.horizons:
            X, y = data_dict[horizon]
            
            # 训练机器学习模型
            ml_predictions = []
            for name in ['rf', 'kelm', 'xgb']:
                self.base_models[name].fit(X.reshape(X.shape[0], -1), y.reshape(y.shape[0], -1))
                pred = self.base_models[name].predict(X.reshape(X.shape[0], -1))
                ml_predictions.append(pred)
                
            # 训练深度学习模型
            self.train_deep_models(X, y)
            dl_predictions = []
            for name in ['rnn_cnn', 'deep_cnn', 'transformer']:
                with torch.no_grad():
                    pred = self.base_models[name](torch.FloatTensor(X).cuda())
                    dl_predictions.append(pred.cpu().numpy())
                    
            # 合并所有预测
            horizon_predictions = np.column_stack([*ml_predictions, *dl_predictions])
            all_predictions.append(horizon_predictions)
            
        # 选择最佳元模型
        meta_features = np.mean(all_predictions, axis=0)
        true_returns = np.diff(df['Close'].values)
        self.best_meta_model = self.select_best_meta_model(meta_features, true_returns)
        
        # 使用最佳元模型生成最终信号
        final_predictions = self.meta_models[self.best_meta_model].predict(meta_features)
        signals = self.generate_signals(final_predictions)
        
        return signals, self.best_meta_model

def main():
    # 创建signals目录
    os.makedirs('signals', exist_ok=True)
    
    # 读取所有货币对的数据
    pairs = ['CNYAUD', 'CNYEUR', 'CNYGBP', 'CNYJPY', 'CNYUSD']
    results = []
    
    for pair in pairs:
        print(f"Processing {pair}...")
        
        # 读取PCA处理后的数据
        pca_path = os.path.join('FE', f'{pair}_PCA.csv')  # 修改文件路径
        
        if not os.path.exists(pca_path):
            print(f"Warning: File not found: {pca_path}")
            continue
            
        try:
            df = pd.read_csv(pca_path)
            
            # 初始化模型
            model = MultiStepPredictor()
            
            # 训练模型并生成信号
            signals, best_model = model.train_and_predict(df)
            
            # 保存信号
            signal_df = pd.DataFrame({
                'Date': df['Date'],
                'Signal': signals
            })
            
            # 创建signals目录（如果不存在）
            signals_dir = os.path.join('signals')
            os.makedirs(signals_dir, exist_ok=True)
            
            # 保存信号文件
            signal_path = os.path.join(signals_dir, f'{pair}_signals.csv')
            signal_df.to_csv(signal_path, index=False)
            
            # 记录结果
            results.append({
                'Pair': pair,
                'Best_Meta_Model': best_model,
                'Signal_Count': len(signals),
                'Buy_Signals': np.sum(signals == 1),
                'Sell_Signals': np.sum(signals == -1),
                'Hold_Signals': np.sum(signals == 0),
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            continue
    
    if results:
        # 保存分析结果
        summary_path = os.path.join('signals', 'analysis_summary.csv')
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print("Analysis completed!")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
