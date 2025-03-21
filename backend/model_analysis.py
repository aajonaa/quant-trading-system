import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from pyswarm import pso
import logging
import matplotlib.pyplot as plt
import glob

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 定义KELM模型
class KELM:
    def __init__(self, gamma=1.0, C=1.0):
        self.gamma = gamma
        self.C = C
        self.beta = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]

        # 计算核矩阵
        K = self._rbf_kernel(X, X)

        # 添加正则化
        I = np.eye(n_samples)
        self.beta = np.linalg.solve(K + I / self.C, y)

        return self

    def predict(self, X):
        K = self._rbf_kernel(X, self.X_train)
        return K.dot(self.beta)

    def _rbf_kernel(self, X1, X2):
        # 计算RBF核
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(-self.gamma * np.sum((X1[i] - X2[j]) ** 2))

        return K


# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # CNN层
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM层
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 转换为CNN输入形状: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # CNN层
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # 转换为LSTM输入形状: (batch_size, seq_len/4, 64)
        x = x.permute(0, 2, 1)

        # LSTM层
        _, (h_n, _) = self.lstm(x)

        # 全连接层
        x = self.fc(h_n.squeeze(0))

        return x


# 定义TCN模型的基本块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        # 当输入和输出维度不同时，使用1x1卷积进行维度匹配
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        
    def init_weights(self):
        # 初始化权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        # 保存输入用于残差连接
        res = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            res = self.downsample(res)
            
        # 确保维度匹配
        if out.size(2) != res.size(2):
            # 如果序列长度不匹配，进行裁剪或填充
            if out.size(2) > res.size(2):
                # 裁剪out以匹配res
                out = out[:, :, :res.size(2)]
            else:
                # 裁剪res以匹配out
                res = res[:, :, :out.size(2)]
                
        return self.relu(out + res)


# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, input_dim, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.input_dim = input_dim
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 转换为TCN输入形状: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 通过TCN网络
        x = self.network(x)
        
        # 使用最后一个时间步的特征进行预测
        x = x[:, :, -1]  # 形状: (batch_size, num_channels[-1])
        
        # 全连接层
        x = self.fc(x)  # 形状: (batch_size, 1)
        
        return x


# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU层
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向GRU

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)

        # GRU层
        output, h_n = self.gru(x)

        # 连接最后一个时间步的正向和反向隐藏状态
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)

        # 全连接层
        x = self.fc(h_n)

        return x


# 定义时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 多步预测器类
class MultiStepPredictor:
    def __init__(self, lookback=10, horizons=[1, 3, 5, 10], particles=1, iterations=1):
        """初始化多步预测器"""
        self.lookback = lookback
        self.horizons = horizons
        self.particles = particles
        self.iterations = iterations
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.meta_model = None
        self.original_prices = None
        self.dates = None
        self.horizon_weights = {h: (1 / h) for h in horizons}

    def prepare_data(self, df):
        """准备训练数据，返回不同预测步长的数据字典"""
        # 存储原始价格数据用于回测
        if 'Close' in df.columns:
            self.original_prices = df['Close'].values
            if 'Date' in df.columns:
                self.dates = pd.to_datetime(df['Date']).values

        # 提取特征列（排除Date和Close）
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        features = df[feature_cols].values

        # 标准化特征
        features = self.scaler.fit_transform(features)

        # 如果有Close列，计算收益率作为目标变量
        if 'Close' in df.columns:
            prices = df['Close'].values
            # 计算收益率: (price_t - price_{t-1}) / price_{t-1}
            returns = np.diff(prices) / prices[:-1]
            # 添加一个0作为第一个收益率，保持长度一致
            returns = np.insert(returns, 0, 0)
            targets = returns
        else:
            # 如果没有Close列，使用最后一个特征作为目标
            targets = features[:, -1]
            features = features[:, :-1]

        # 为每个预测步长创建数据
        data_dict = {}

        for horizon in self.horizons:
            X, y = self._create_sequences(features, targets, self.lookback, horizon)
            data_dict[horizon] = (X, y)

        return data_dict

    def _create_sequences(self, features, targets, lookback, horizon):
        """创建时间序列数据的输入序列和目标序列"""
        X, y = [], []

        for i in range(len(features) - lookback - horizon + 1):
            # 输入序列
            X.append(features[i:i + lookback])

            # 目标序列 - 使用未来horizon步的平均收益率
            y.append(np.mean(targets[i + lookback:i + lookback + horizon]))

        return np.array(X), np.array(y)

    def optimize_rf(self, X, y):
        """使用PSO优化随机森林参数"""
        self.logger.info("优化随机森林参数...")

        def objective_function(params):
            n_estimators = int(params[0])
            max_depth = int(params[1])
            min_samples_split = int(params[2])

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

            # 使用交叉验证评估模型
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 计算均方误差
            mse = np.mean((y_val - y_pred) ** 2)

            return mse

        # 参数范围
        lb = [50, 3, 2]
        ub = [200, 20, 20]

        # 运行PSO
        best_params, _ = pso(objective_function, lb, ub, swarmsize=self.particles, maxiter=self.iterations)

        # 返回最佳参数
        return {
            'n_estimators': int(best_params[0]),
            'max_depth': int(best_params[1]),
            'min_samples_split': int(best_params[2])
        }

    def optimize_kelm(self, X, y):
        """使用PSO优化KELM参数"""
        self.logger.info("优化KELM参数...")

        def objective_function(params):
            gamma = params[0]
            C = params[1]

            model = KELM(gamma=gamma, C=C)

            # 使用交叉验证评估模型
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 计算均方误差
            mse = np.mean((y_val - y_pred) ** 2)

            return mse

        # 参数范围
        lb = [0.01, 0.1]
        ub = [10.0, 100.0]

        # 运行PSO
        best_params, _ = pso(objective_function, lb, ub, swarmsize=self.particles, maxiter=self.iterations)

        # 返回最佳参数
        return {
            'gamma': best_params[0],
            'C': best_params[1]
        }

    def optimize_xgb(self, X, y):
        """使用PSO优化XGBoost参数"""
        self.logger.info("优化XGBoost参数...")

        def objective_function(params):
            n_estimators = int(params[0])
            max_depth = int(params[1])
            learning_rate = params[2]
            subsample = params[3]

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42
            )

            # 使用交叉验证评估模型
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 计算均方误差
            mse = np.mean((y_val - y_pred) ** 2)

            return mse

        # 参数范围
        lb = [50, 3, 0.01, 0.5]
        ub = [200, 10, 0.3, 1.0]

        # 运行PSO
        best_params, _ = pso(objective_function, lb, ub, swarmsize=self.particles, maxiter=self.iterations)

        # 返回最佳参数
        return {
            'n_estimators': int(best_params[0]),
            'max_depth': int(best_params[1]),
            'learning_rate': best_params[2],
            'subsample': best_params[3]
        }

    def train_ml_models(self, X, y):
        """训练机器学习模型"""
        self.logger.info("训练机器学习模型...")

        ml_models = {}

        # 优化并训练随机森林
        self.logger.info("优化和训练随机森林...")
        rf_params = self.optimize_rf(X, y)
        self.logger.info(f"随机森林最佳参数: {rf_params}")
        rf = RandomForestRegressor(**rf_params, random_state=42)
        rf.fit(X, y)
        ml_models['rf'] = rf

        # 优化并训练KELM
        self.logger.info("优化和训练KELM...")
        kelm_params = self.optimize_kelm(X, y)
        self.logger.info(f"KELM最佳参数: {kelm_params}")
        kelm = KELM(**kelm_params)
        kelm.fit(X, y)
        ml_models['kelm'] = kelm

        # 优化并训练XGBoost
        self.logger.info("优化和训练XGBoost...")
        xgb_params = self.optimize_xgb(X, y)
        self.logger.info(f"XGBoost最佳参数: {xgb_params}")
        xgb = XGBRegressor(**xgb_params, random_state=42)
        xgb.fit(X, y)
        ml_models['xgb'] = xgb

        return ml_models

    def train_dl_models(self, X, y, input_dim):
        """训练深度学习模型"""
        self.logger.info("训练深度学习模型...")
        dl_models = {}
        
        # 准备数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # 训练CNN-LSTM模型
        try:
            self.logger.info("训练CNN-LSTM模型...")
            cnn_lstm = CNNLSTM(input_dim=input_dim, hidden_dim=64)
            
            # 训练模型
            optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            batch_size = 32
            for epoch in range(5):
                for i in range(0, len(X_train_tensor), batch_size):
                    # 获取批次数据
                    batch_x = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    # 确保批次大小足够
                    if len(batch_x) < 2:
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    output = cnn_lstm(batch_x)
                    
                    # 确保输出和目标维度匹配
                    batch_y = batch_y.reshape(batch_y.size(0), -1)
                    output = output.reshape(output.size(0), -1)
                    
                    # 计算损失
                    loss = criterion(output, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
            
            dl_models['cnn_lstm'] = cnn_lstm
            self.logger.info("CNN-LSTM模型训练完成")
        except Exception as e:
            self.logger.error(f"训练CNN-LSTM模型时出错: {str(e)}")
        
        # 训练TCN模型
        try:
            self.logger.info("训练TCN模型...")
            tcn = TCN(input_dim=input_dim)
            
            # 训练模型
            optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            batch_size = 32
            for epoch in range(5):
                for i in range(0, len(X_train_tensor), batch_size):
                    # 获取批次数据
                    batch_x = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    # 确保批次大小足够
                    if len(batch_x) < 2:
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    output = tcn(batch_x)
                    
                    # 确保输出和目标维度匹配
                    batch_y = batch_y.view(batch_y.size(0), -1)
                    
                    # 计算损失
                    loss = criterion(output, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
            
            dl_models['tcn'] = tcn
            self.logger.info("TCN模型训练完成")
        except Exception as e:
            self.logger.error(f"训练TCN模型时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())  # 打印完整的错误堆栈
        
        # 训练GRU模型
        try:
            self.logger.info("训练GRU模型...")
            gru = GRUModel(input_dim=input_dim)
            
            # 训练模型
            optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            batch_size = 32
            for epoch in range(5):
                for i in range(0, len(X_train_tensor), batch_size):
                    # 获取批次数据
                    batch_x = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    # 确保批次大小足够
                    if len(batch_x) < 2:
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    output = gru(batch_x)
                    
                    # 确保输出和目标维度匹配
                    batch_y = batch_y.reshape(batch_y.size(0), -1)
                    output = output.reshape(output.size(0), -1)
                    
                    # 计算损失
                    loss = criterion(output, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
            
            dl_models['gru'] = gru
            self.logger.info("GRU模型训练完成")
        except Exception as e:
            self.logger.error(f"训练GRU模型时出错: {str(e)}")
        
        return dl_models

    def predict_ml_model(self, model, X):
        """使用机器学习模型进行预测"""
        try:
            return model.predict(X)
        except Exception as e:
            self.logger.error(f"机器学习模型预测出错: {str(e)}")
            return np.array([])

    def predict_dl_model(self, model, X):
        """使用深度学习模型进行预测"""
        try:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).cpu().numpy().flatten()
            return predictions
        except Exception as e:
            self.logger.error(f"深度学习模型预测出错: {str(e)}")
            return np.array([])

    def predict_for_horizon(self, horizon, X, ml_models, dl_models):
        """为特定预测步长生成预测"""
        predictions = []

        # 机器学习模型预测
        for name, model in ml_models.items():
            try:
                pred = self.predict_ml_model(model, X)
                predictions.append(pred)
                self.logger.info(f"{name} 在步长 {horizon} 的预测完成")
            except Exception as e:
                self.logger.error(f"{name} 在步长 {horizon} 预测出错: {str(e)}")

        # 深度学习模型预测
        for name, model in dl_models.items():
            try:
                # 重塑数据以适应深度学习模型
                X_reshaped = X.reshape(-1, self.lookback, X.shape[1] // self.lookback)
                pred = self.predict_dl_model(model, X_reshaped)
                predictions.append(pred)
                self.logger.info(f"{name} 在步长 {horizon} 的预测完成")
            except Exception as e:
                self.logger.error(f"{name} 在步长 {horizon} 预测出错: {str(e)}")

        # 确保所有预测长度一致
        min_length = min(len(p) for p in predictions) if predictions else 0
        aligned_preds = [p[:min_length] for p in predictions] if min_length > 0 else []

        # 计算平均预测
        if aligned_preds:
            ensemble_pred = np.mean(aligned_preds, axis=0)
            return ensemble_pred
        else:
            self.logger.warning(f"步长 {horizon} 没有有效预测")
            return np.array([])

    def predict_multi_horizon(self, data_dict, ml_models, dl_models):
        """生成多步预测"""
        self.logger.info(f"预测步长权重: {self.horizon_weights}")
        
        all_predictions = []
        prediction_lengths = []
        
        for horizon in self.horizons:
            self.logger.info(f"处理预测步长 {horizon}...")
            X_test, _ = data_dict[horizon]
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # 存储当前步长的所有模型预测
            horizon_predictions = {}
            
            # 机器学习模型预测
            for model_name, model in ml_models.items():
                try:
                    preds = model.predict(X_test_flat)
                    # 检查并处理NaN值
                    if np.isnan(preds).any():
                        self.logger.warning(f"{model_name} 在步长 {horizon} 的预测包含NaN值，使用0填充")
                        preds = np.nan_to_num(preds, nan=0.0)
                    horizon_predictions[model_name] = preds * self.horizon_weights[horizon]
                    self.logger.info(f"{model_name} 在步长 {horizon} 的预测完成")
                except Exception as e:
                    self.logger.error(f"{model_name} 在步长 {horizon} 的预测出错: {str(e)}")
            
            # 深度学习模型预测
            for model_name, model in dl_models.items():
                try:
                    X_tensor = torch.FloatTensor(X_test)
                    with torch.no_grad():
                        preds = model(X_tensor).numpy().flatten()
                    # 检查并处理NaN值
                    if np.isnan(preds).any():
                        self.logger.warning(f"{model_name} 在步长 {horizon} 的预测包含NaN值，使用0填充")
                        preds = np.nan_to_num(preds, nan=0.0)
                    horizon_predictions[model_name] = preds * self.horizon_weights[horizon]
                    self.logger.info(f"{model_name} 在步长 {horizon} 的预测完成")
                except Exception as e:
                    self.logger.error(f"{model_name} 在步长 {horizon} 的预测出错: {str(e)}")
            
            # 将当前步长的预测添加到总预测中
            if horizon_predictions:
                all_predictions.append(horizon_predictions)
                # 记录每个步长的预测长度
                for model_name, preds in horizon_predictions.items():
                    prediction_lengths.append(len(preds))
        
        # 合并所有步长的预测
        if not all_predictions:
            return []
        
        # 找出所有预测中的最小长度，以确保所有预测可以对齐
        min_length = min(prediction_lengths) if prediction_lengths else 0
        self.logger.info(f"调整所有预测到最小长度: {min_length}")
        
        # 创建一个字典来存储所有模型的对齐预测
        combined_predictions = {}
        
        # 首先初始化字典
        for horizon_pred in all_predictions:
            for model_name in horizon_pred.keys():
                if model_name not in combined_predictions:
                    combined_predictions[model_name] = np.zeros(min_length)
        
        # 然后累加预测，确保长度一致
        for horizon_pred in all_predictions:
            for model_name, preds in horizon_pred.items():
                # 截断或填充预测以匹配最小长度
                if len(preds) > min_length:
                    preds = preds[:min_length]
                elif len(preds) < min_length:
                    # 这种情况不应该发生，但为了安全起见
                    self.logger.warning(f"{model_name} 预测长度 {len(preds)} 小于最小长度 {min_length}，填充0")
                    padding = np.zeros(min_length - len(preds))
                    preds = np.concatenate([preds, padding])
                
                combined_predictions[model_name] += preds
        
        # 将所有模型的预测合并为一个特征矩阵
        X_combined = np.column_stack([preds for model_name, preds in combined_predictions.items()])
        
        # 检查并处理最终预测中的NaN值
        if np.isnan(X_combined).any():
            self.logger.warning("合并后的预测包含NaN值，使用0填充")
            X_combined = np.nan_to_num(X_combined, nan=0.0)
        
        return X_combined

    def train_meta_model(self, X, y):
        """训练元模型"""
        try:
            self.logger.info("训练元模型...")
            
            # 检查输入数据是否包含NaN
            if np.isnan(X).any():
                self.logger.warning("输入特征包含NaN值，使用0填充")
                X = np.nan_to_num(X, nan=0.0)
            
            if np.isnan(y).any():
                self.logger.warning("目标变量包含NaN值，使用0填充")
                y = np.nan_to_num(y, nan=0.0)
            
            # 创建分类标签：上涨(1)、下跌(-1)、持平(0)
            threshold = 0.001  # 1‰的变动阈值
            y_class = np.zeros_like(y)
            y_class[y > threshold] = 1  # 上涨
            y_class[y < -threshold] = -1  # 下跌
            
            # 训练梯度提升分类器
            self.meta_model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42
            )
            
            self.meta_model.fit(X, y_class)
            
            # 计算训练集准确率
            train_preds = self.meta_model.predict(X)
            accuracy = np.mean(train_preds == y_class)
            self.logger.info(f"元模型训练集准确率: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"训练元模型时出错: {str(e)}")
            # 创建一个简单的备用模型
            self.meta_model = None

    def generate_signals(self, X):
        """生成交易信号"""
        try:
            self.logger.info("生成交易信号...")
            
            # 检查输入数据是否包含NaN
            if np.isnan(X).any():
                self.logger.warning("输入特征包含NaN值，使用0填充")
                X = np.nan_to_num(X, nan=0.0)
            
            # 如果元模型训练失败，使用简单规则生成信号
            if self.meta_model is None:
                self.logger.warning("元模型不可用，使用简单规则生成信号")
                # 使用第一个模型的预测作为信号
                signals = np.zeros(X.shape[0])
                signals[X[:, 0] > 0] = 1  # 上涨预测
                signals[X[:, 0] < 0] = -1  # 下跌预测
                return signals
            
            # 使用元模型预测
            signals = self.meta_model.predict(X)
            
            # 应用信号平滑：避免频繁交易
            smoothed_signals = self.smooth_signals(signals)
            
            return smoothed_signals
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            # 返回空信号
            return np.zeros(X.shape[0])

    def smooth_signals(self, signals, min_hold_period=5):
        """优化信号，避免频繁交易"""
        optimized = signals.copy()

        # 实现一个简单的持有期策略
        last_signal = 0
        hold_counter = 0

        for i in range(len(signals)):
            if hold_counter > 0:
                # 在持有期内保持上一个信号
                optimized[i] = last_signal
                hold_counter -= 1
            elif signals[i] != 0 and signals[i] != last_signal:
                # 新的非零信号，开始新的持有期
                last_signal = signals[i]
                optimized[i] = last_signal
                hold_counter = min_hold_period
            else:
                # 保持原信号
                last_signal = signals[i]

        return optimized

    def backtest(self, signals, prices, dates):
        """回测交易信号"""
        try:
            self.logger.info("进行回测...")
            
            # 确保信号和价格长度匹配
            if len(signals) + 1 != len(prices):
                self.logger.warning(f"信号长度 ({len(signals)}) 和价格长度 ({len(prices)}) 不匹配，调整价格长度")
                if len(signals) + 1 < len(prices):
                    prices = prices[-(len(signals)+1):]
                else:
                    # 如果信号比价格多，截断信号
                    signals = signals[:len(prices)-1]
            
            # 初始资金
            initial_capital = 10000
            capital = initial_capital
            position = 0  # 0表示空仓，1表示多头，-1表示空头
            
            # 记录每次交易
            trades = []
            equity_curve = [capital]
            
            for i in range(len(signals)):
                current_price = prices[i]
                next_price = prices[i+1]
                signal = signals[i]
                
                # 检查信号是否有效
                if np.isnan(signal):
                    self.logger.warning(f"第 {i} 个信号为NaN，跳过")
                    continue
                
                # 执行交易
                if signal == 1 and position <= 0:  # 买入信号
                    position = 1
                    trades.append({
                        'date': dates[i],
                        'action': 'BUY',
                        'price': current_price,
                        'capital': capital
                    })
                elif signal == -1 and position >= 0:  # 卖出信号
                    position = -1
                    trades.append({
                        'date': dates[i],
                        'action': 'SELL',
                        'price': current_price,
                        'capital': capital
                    })
                
                # 更新资金
                if position == 1:
                    capital = capital * (next_price / current_price)
                elif position == -1:
                    capital = capital * (current_price / next_price)
                
                equity_curve.append(capital)
            
            # 计算回测指标
            returns = np.diff(equity_curve) / np.array(equity_curve)[:-1]
            total_return = (capital - initial_capital) / initial_capital
            annual_return = total_return / (len(signals) / 252)  # 假设一年252个交易日
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(equity_curve)
            
            self.logger.info(f"总收益率: {total_return:.4f}")
            self.logger.info(f"年化收益率: {annual_return:.4f}")
            self.logger.info(f"夏普比率: {sharpe_ratio:.4f}")
            self.logger.info(f"最大回撤: {max_drawdown:.4f}")
            
            # 返回回测结果
            backtest_data = {
                'signals': signals,
                'prices': prices[:-1],  # 去掉最后一个价格，使长度与信号匹配
                'dates': dates,
                'equity_curve': equity_curve,
                'trades': trades,
                'metrics': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
            }
            
            self.logger.info("回测数据准备完成")
            return backtest_data
            
        except Exception as e:
            self.logger.error(f"回测过程中出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 返回空回测数据
            return {
                'signals': signals,
                'prices': prices[:-1] if len(prices) > len(signals) else prices,
                'dates': dates,
                'equity_curve': [10000] * (len(signals) + 1),
                'trades': [],
                'metrics': {
                    'total_return': 0,
                    'annual_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            }

    def train_and_predict(self, df):
        """训练模型并生成预测信号"""
        self.logger.info("准备训练数据...")
        data_dict = self.prepare_data(df)
        
        # 使用第一个预测步长训练所有模型
        X, y = data_dict[self.horizons[0]]
        X_flat = X.reshape(X.shape[0], -1)
        
        # 训练机器学习模型
        ml_models = self.train_ml_models(X_flat, y)
        
        # 训练深度学习模型
        input_dim = X.shape[2]
        dl_models = self.train_dl_models(X, y, input_dim)  # 确保传递正确的参数
        
        # 使用多步预测生成最终预测
        self.logger.info("生成多步预测...")
        predictions = self.predict_multi_horizon(data_dict, ml_models, dl_models)
        
        # 如果没有有效预测，返回空结果
        if len(predictions) == 0:
            self.logger.error("没有生成有效的预测结果")
            return None, None, None
        
        # 计算未来收益率，用于训练元模型
        if hasattr(self, 'original_prices'):
            future_returns = np.diff(self.original_prices) / self.original_prices[:-1]
        else:
            # 如果没有original_prices属性，使用df中的Close列
            future_returns = df['returns'].values if 'returns' in df.columns else df['Close'].pct_change().values
        
        # 确保future_returns长度与predictions匹配
        future_returns = future_returns[-len(predictions):] if len(future_returns) >= len(predictions) else np.zeros(len(predictions))
        
        # 训练元模型
        self.train_meta_model(predictions, future_returns)
        
        # 生成交易信号
        signals = self.generate_signals(predictions)
        
        # 进行回测
        if hasattr(self, 'original_prices') and hasattr(self, 'dates'):
            prices = self.original_prices[-len(signals)-1:]
            dates = self.dates[-len(signals):]
        else:
            # 如果没有original_prices或dates属性，使用df中的数据
            prices = df['Close'].values[-len(signals)-1:]
            dates = df['Date'].values[-len(signals):] if 'Date' in df.columns else np.arange(len(signals))
        
        backtest_data = self.backtest(signals, prices, dates)
        
        return signals, "gdboost", backtest_data

    def save_results(self, symbol, signals, best_model, backtest_data):
        """保存预测结果和回测数据"""
        self.logger.info(f"保存 {symbol} 的结果...")

        # 创建结果目录
        os.makedirs("signals", exist_ok=True)

        # 保存信号
        signal_df = pd.DataFrame({
            'Date': backtest_data['dates'],
            'Price': backtest_data['prices'],
            'Signal': backtest_data['signals']
        })

        signal_df.to_csv(f"signals/{symbol}_signals.csv", index=False)
        self.logger.info(f"{symbol} 的信号已保存")

        # 绘制信号图
        self.plot_signals(symbol, backtest_data)

        return signal_df

    def plot_signals(self, symbol, backtest_data):
        """绘制信号图"""
        plt.figure(figsize=(12, 6))

        # 绘制价格
        plt.plot(backtest_data['dates'], backtest_data['prices'], label='Price')

        # 标记买入信号
        buy_dates = [date for i, date in enumerate(backtest_data['dates']) if backtest_data['signals'][i] == 1]
        buy_prices = [price for i, price in enumerate(backtest_data['prices']) if backtest_data['signals'][i] == 1]
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')

        # 标记卖出信号
        sell_dates = [date for i, date in enumerate(backtest_data['dates']) if backtest_data['signals'][i] == -1]
        sell_prices = [price for i, price in enumerate(backtest_data['prices']) if backtest_data['signals'][i] == -1]
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')

        plt.title(f'{symbol} Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # 保存图表
        plt.savefig(f"signals/{symbol}_signals.png")
        plt.close()

        self.logger.info(f"{symbol} 的信号图已保存")


def main():
    # 创建signals目录
    os.makedirs("signals", exist_ok=True)
    
    # 获取所有特征工程后的数据文件
    pca_files = glob.glob("FE/*_PCA.csv")
    processed_files = glob.glob("FE/*_processed.csv")
    
    # 创建文件映射，将PCA文件和processed文件关联起来
    file_mapping = {}
    for pca_file in pca_files:
        symbol = os.path.basename(pca_file).split('_')[0]  # 提取货币对名称
        for proc_file in processed_files:
            if symbol in proc_file and "_processed" in proc_file:
                file_mapping[pca_file] = proc_file
                break
    
    # 处理每个PCA文件
    for pca_file, proc_file in file_mapping.items():
        # 提取货币对名称
        symbol = os.path.basename(pca_file).split('_')[0]
        logger.info(f"处理 {symbol}...")
        
        try:
            # 读取PCA数据（用于训练模型）
            df_pca = pd.read_csv(pca_file)
            
            # 读取processed数据（用于获取Close和return）
            df_processed = pd.read_csv(proc_file)
            
            # 如果PCA数据中没有Date列，添加一个日期列
            if 'Date' not in df_pca.columns:
                if 'Date' in df_processed.columns:
                    df_pca['Date'] = df_processed['Date'].values[:len(df_pca)]
                else:
                    df_pca['Date'] = pd.date_range(start='2015-01-01', periods=len(df_pca), freq='D')
            
            # 如果PCA数据中没有Close列，从processed数据中添加
            if 'Close' not in df_pca.columns:
                logger.info(f"从processed数据中添加Close列到PCA数据")
                if 'Close' in df_processed.columns:
                    df_pca['Close'] = df_processed['Close'].values[:len(df_pca)]
                else:
                    # 如果processed数据中也没有Close列，尝试从原始数据获取
                    try:
                        raw_file = f"data/{symbol}.csv"
                        if os.path.exists(raw_file):
                            raw_df = pd.read_csv(raw_file)
                            if len(raw_df) >= len(df_pca):
                                df_pca['Close'] = raw_df['Close'].values[:len(df_pca)]
                            else:
                                # 如果原始数据长度不足，使用最后一列作为收盘价
                                df_pca['Close'] = df_pca.iloc[:, -1].values
                        else:
                            # 如果没有原始数据，使用最后一列作为收盘价
                            df_pca['Close'] = df_pca.iloc[:, -1].values
                    except Exception as e:
                        logger.error(f"添加Close列时出错: {str(e)}")
                        # 使用最后一列作为收盘价
                        df_pca['Close'] = df_pca.iloc[:, -1].values
            
            # 如果PCA数据中没有returns列，从processed数据中添加或计算
            if 'returns' not in df_pca.columns:
                logger.info(f"添加returns列到PCA数据")
                if 'returns' in df_processed.columns:
                    df_pca['returns'] = df_processed['returns'].values[:len(df_pca)]
                else:
                    # 计算收益率
                    df_pca['returns'] = df_pca['Close'].pct_change()
                    df_pca['returns'] = df_pca['returns'].fillna(0)  # 填充第一行的NaN
            
            # 创建多步预测器
            model = MultiStepPredictor(lookback=10, horizons=[1, 3, 5, 10], particles=1, iterations=1)
            
            # 训练模型并生成预测
            signals, best_model, backtest_data = model.train_and_predict(df_pca)
            
            if signals is not None and backtest_data is not None:
                # 保存结果
                model.save_results(symbol, signals, best_model, backtest_data)
            else:
                logger.warning("未生成结果!")
                
        except Exception as e:
            logger.error(f"处理 {symbol} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()