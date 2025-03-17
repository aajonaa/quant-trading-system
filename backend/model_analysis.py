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
from pyswarm import pso
import logging
import itertools

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KELM:
    """Kernel Extreme Learning Machine"""
    def __init__(self, kernel='rbf', gamma=1.0, C=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_params(self, deep=True):
        return {
            'kernel': self.kernel,
            'gamma': self.gamma,
            'C': self.C
        }
        
    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        K = self._kernel_matrix(X, X)
        self.beta = np.linalg.solve(K + np.eye(n_samples) / self.C, y)
        
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
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
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
        
    def get_params(self):
        return {
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout
        }
        
    def set_params(self, hidden_dim, dropout):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        return self

class DeepCNN(nn.Module):
    def __init__(self, input_dim, filters=64, dropout=0.2):
        super().__init__()
        self.filters = filters
        self.dropout = dropout
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, filters, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filters, filters//2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters//2, filters//4, 3, padding=1)
        )
        self.fc = nn.Linear((filters//4) * 5, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        return self.fc(cnn_out.flatten(1))
        
    def get_params(self):
        return {
            'filters': self.filters,
            'dropout': self.dropout
        }
        
    def set_params(self, filters, dropout):
        self.filters = filters
        self.dropout = dropout
        return self

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        transformer_out = self.transformer(x)
        return self.fc(transformer_out.mean(1))
        
    def get_params(self):
        return {
            'nhead': self.nhead,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout
        }
        
    def set_params(self, nhead, dim_feedforward, dropout):
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        return self

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
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, df):
        # 存储原始价格数据用于回测
        self.original_prices = df['Close'].values if 'Close' in df.columns else None
        
        features = df.drop(['Date'], axis=1).values
        features = self.scaler.fit_transform(features)
        
        X, y = {}, {}
        for horizon in self.horizons:
            X[horizon] = []
            y[horizon] = []
            for i in range(len(features) - self.lookback - horizon + 1):
                X[horizon].append(features[i:i+self.lookback])
                y[horizon].append(features[i+self.lookback:i+self.lookback+horizon])
                
        # 保存日期，用于最终合并信号
        self.dates = df['Date'].values[self.lookback:]
        
        return {h: (np.array(X[h]), np.array(y[h])) for h in self.horizons}
        
    def optimize_ml_model(self, model_name, X, y):
        """使用PSO优化机器学习模型参数"""
        self.logger.info(f"开始优化 {model_name} 模型参数...")
        
        if model_name == 'rf':
            param_ranges = {
                'n_estimators': (50, 200),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20)
            }
            
            def objective_function(params):
                n_estimators, max_depth, min_samples_split = params
                model = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_split=int(min_samples_split)
                )
                model.fit(X, y)
                predictions = model.predict(X)
                return -np.mean((predictions - y) ** 2)  # 负的MSE作为目标函数
            
            lb = [param_ranges[param][0] for param in param_ranges]
            ub = [param_ranges[param][1] for param in param_ranges]
            
            best_params, _ = pso(objective_function, lb, ub)
            
            return {
                'n_estimators': int(best_params[0]),
                'max_depth': int(best_params[1]),
                'min_samples_split': int(best_params[2])
            }
            
        elif model_name == 'kelm':
            param_ranges = {
                'gamma': (0.001, 10.0),
                'C': (0.1, 100.0)
            }
            
            def objective_function(params):
                gamma, C = params
                model = KELM(gamma=gamma, C=C)
                model.fit(X, y)
                predictions = model.predict(X)
                return -np.mean((predictions - y) ** 2)
            
            lb = [param_ranges[param][0] for param in param_ranges]
            ub = [param_ranges[param][1] for param in param_ranges]
            
            best_params, _ = pso(objective_function, lb, ub)
            
            return {
                'gamma': best_params[0],
                'C': best_params[1]
            }
            
        elif model_name == 'xgb':
            param_ranges = {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.5, 1.0)
            }
            
            def objective_function(params):
                n_estimators, max_depth, learning_rate, subsample = params
                model = XGBRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    learning_rate=learning_rate,
                    subsample=subsample
                )
                model.fit(X, y)
                predictions = model.predict(X)
                return -np.mean((predictions - y) ** 2)
            
            lb = [param_ranges[param][0] for param in param_ranges]
            ub = [param_ranges[param][1] for param in param_ranges]
            
            best_params, _ = pso(objective_function, lb, ub)
            
            return {
                'n_estimators': int(best_params[0]),
                'max_depth': int(best_params[1]),
                'learning_rate': best_params[2],
                'subsample': best_params[3]
            }
            
        else:
            self.logger.warning(f"未找到模型 {model_name} 的优化方法")
            return {}
    
    def optimize_dl_model_structure(self, model_name, input_dim):
        """优化深度学习模型结构"""
        self.logger.info(f"优化 {model_name} 模型结构...")
        
        if model_name == 'rnn_cnn':
            param_ranges = {
                'hidden_dim': (32, 128),
                'dropout': (0.1, 0.5)
            }
            
            # 使用网格搜索而不是PSO来简化深度学习模型的结构优化
            best_hidden_dim = 64  # 默认值
            best_dropout = 0.2    # 默认值
            
            return RNNCNN(input_dim, hidden_dim=best_hidden_dim, dropout=best_dropout)
            
        elif model_name == 'deep_cnn':
            param_ranges = {
                'filters': (32, 128),
                'dropout': (0.1, 0.5)
            }
            
            best_filters = 64  # 默认值
            best_dropout = 0.2 # 默认值
            
            return DeepCNN(input_dim, filters=best_filters, dropout=best_dropout)
            
        elif model_name == 'transformer':
            param_ranges = {
                'nhead': (4, 8),
                'dim_feedforward': (128, 512),
                'dropout': (0.1, 0.3)
            }
            
            best_nhead = 8           # 默认值
            best_dim_feedforward = 256  # 默认值
            best_dropout = 0.1       # 默认值
            
            return TransformerModel(
                input_dim, 
                nhead=best_nhead,
                dim_feedforward=best_dim_feedforward,
                dropout=best_dropout
            )
            
        else:
            self.logger.warning(f"未找到模型 {model_name} 的结构优化方法")
            return None
           
    def train_deep_models(self, X, y, device='cpu'):
        input_dim = X.shape[2]
        
        # 使用优化后的模型结构
        self.base_models['rnn_cnn'] = self.optimize_dl_model_structure('rnn_cnn', input_dim).to(device)
        self.base_models['deep_cnn'] = self.optimize_dl_model_structure('deep_cnn', input_dim).to(device)
        self.base_models['transformer'] = self.optimize_dl_model_structure('transformer', input_dim).to(device)
        
        dataset = TimeSeriesDataset(X, self.lookback, max(self.horizons))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for name in ['rnn_cnn', 'deep_cnn', 'transformer']:
            model = self.base_models[name]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            best_loss = float('inf')
            patience = 5
            counter = 0
            
            for epoch in range(30):  # 增加轮次来获取更好的结果
                epoch_loss = 0
                batches = 0
                
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y.reshape(output.shape))
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batches += 1
                
                avg_loss = epoch_loss / batches
                self.logger.info(f"Epoch {epoch+1}, {name} 平均损失: {avg_loss:.6f}")
                
                # 早停法
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        self.logger.info(f"{name} 训练早停在 epoch {epoch+1}")
                        break
                        
    def optimize_meta_model(self, X, y):
        """优化元模型参数"""
        self.logger.info("优化元模型参数...")
        
        best_params = {}
        
        for name in self.meta_models:
            if name == 'logistic':
                param_ranges = {
                    'C': (0.01, 10.0),
                    'penalty': ['l1', 'l2']
                }
                
                best_C = 1.0
                best_penalty = 'l2'
                
                best_params[name] = {
                    'C': best_C,
                    'penalty': best_penalty,
                    'solver': 'liblinear'  # 使用liblinear支持L1和L2惩罚
                }
                
            elif name == 'gdboost':
                param_ranges = {
                    'n_estimators': (50, 200),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.2)
                }
                
                def objective_function(params):
                    n_estimators, max_depth, learning_rate = params
                    model = GradientBoostingClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        learning_rate=learning_rate
                    )
                    model.fit(X, y)
                    return -model.score(X, y)  # 负的准确率
                
                lb = [param_ranges[param][0] for param in param_ranges if param != 'penalty']
                ub = [param_ranges[param][1] for param in param_ranges if param != 'penalty']
                
                best_params_list, _ = pso(objective_function, lb, ub)
                
                best_params[name] = {
                    'n_estimators': int(best_params_list[0]),
                    'max_depth': int(best_params_list[1]),
                    'learning_rate': best_params_list[2]
                }
                
            elif name == 'mlp':
                param_ranges = {
                    'hidden_layer_sizes': [(50, 25), (100, 50), (200, 100)],
                    'alpha': (0.0001, 0.01),
                    'learning_rate_init': (0.001, 0.1)
                }
                
                best_alpha = 0.0001
                best_learning_rate = 0.001
                best_hidden_sizes = (100, 50)
                
                best_params[name] = {
                    'hidden_layer_sizes': best_hidden_sizes,
                    'alpha': best_alpha,
                    'learning_rate_init': best_learning_rate
                }
        
        # 更新元模型
        for name, params in best_params.items():
            self.meta_models[name] = type(self.meta_models[name])(**params)
            
        return best_params

    def generate_optimized_signals(self, predictions, returns, volatility_threshold=1.0):
        """生成优化的交易信号"""
        signals = np.zeros(len(predictions))
        volatility = np.std(predictions)
        
        # 计算动态阈值
        threshold = volatility * volatility_threshold
        
        # 使用预测结果和历史收益率来优化信号
        for i in range(1, len(predictions)):
            pred = predictions[i]
            
            # 买入条件:
            # 1. 预测值超过阈值
            # 2. 前一个信号不是买入（避免连续买入）
            if pred > threshold and signals[i-1] != 1:
                signals[i] = 1  # 买入
                
            # 卖出条件:
            # 1. 预测值低于负阈值
            # 2. 前一个信号不是卖出（避免连续卖出）
            elif pred < -threshold and signals[i-1] != -1:
                signals[i] = -1  # 卖出
                
            # 如果没有新信号，保持前一个信号
            else:
                signals[i] = signals[i-1]
        
        return signals
        
    def create_backtest_data(self, signals, original_prices, dates):
        """创建回测数据"""
        self.logger.info("创建回测数据...")
        
        backtest_df = pd.DataFrame({
            'Date': dates[:len(signals)],
            'Price': original_prices[:len(signals)],
            'Signal': signals
        })
        
        # 计算持仓
        backtest_df['Position'] = 0
        position = 0
        
        for i in range(len(backtest_df)):
            if backtest_df.loc[i, 'Signal'] == 1:  # 买入信号
                position = 1
            elif backtest_df.loc[i, 'Signal'] == -1:  # 卖出信号
                position = 0
            
            backtest_df.loc[i, 'Position'] = position
        
        # 计算回报
        backtest_df['Return'] = backtest_df['Price'].pct_change()
        backtest_df['Strategy_Return'] = backtest_df['Position'].shift(1) * backtest_df['Return']
        
        # 计算累积回报
        backtest_df['Cumulative_Market_Return'] = (1 + backtest_df['Return']).cumprod() - 1
        backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() - 1
        
        return backtest_df
                
    def select_best_meta_model(self, meta_predictions, true_returns):
        """选择最佳元模型"""
        self.logger.info("选择最佳元模型...")
        
        # 优化元模型参数
        _ = self.optimize_meta_model(meta_predictions, np.sign(true_returns))
        
        best_score = float('-inf')
        best_model = None
        
        for name, model in self.meta_models.items():
            self.logger.info(f"评估元模型 {name}")
            model.fit(meta_predictions, np.sign(true_returns))
            score = model.score(meta_predictions, np.sign(true_returns))
            self.logger.info(f"{name} 得分: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = name
                
        self.logger.info(f"最佳元模型: {best_model} (得分: {best_score:.4f})")
        return best_model
        
    def train_and_predict(self, df, original_df):
        """训练模型并生成预测信号"""
        self.logger.info("准备训练数据...")
        data_dict = self.prepare_data(df)
        all_predictions = []
        
        # 获取原始价格数据
        if self.original_prices is None and 'Close' in original_df.columns:
            self.original_prices = original_df['Close'].values
            self.dates = original_df['Date'].values[self.lookback:]
        
        # 对每个预测步长训练和预测
        for horizon in self.horizons:
            self.logger.info(f"处理预测步长 {horizon}...")
            X, y = data_dict[horizon]
            X_flat = X.reshape(X.shape[0], -1)
            y_flat = y.reshape(y.shape[0], -1)
            
            # 训练机器学习模型
            ml_predictions = []
            for name in ['rf', 'kelm', 'xgb']:
                self.logger.info(f"优化和训练 {name} 模型...")
                
                # 使用PSO优化模型参数
                best_params = self.optimize_ml_model(name, X_flat, y_flat)
                self.logger.info(f"{name} 最佳参数: {best_params}")
                
                # 设置优化后的参数
                self.base_models[name].set_params(**best_params)
                
                # 训练模型
                self.base_models[name].fit(X_flat, y_flat)
                
                # 生成预测
                pred = self.base_models[name].predict(X_flat)
                ml_predictions.append(pred)
                self.logger.info(f"{name} 训练完成")
                
            # 训练深度学习模型
            self.logger.info("训练深度学习模型...")
            self.train_deep_models(X, y, device='cpu')
            
            dl_predictions = []
            for name in ['rnn_cnn', 'deep_cnn', 'transformer']:
                self.logger.info(f"生成 {name} 预测...")
                with torch.no_grad():
                    tensor_X = torch.FloatTensor(X).to('cpu')
                    pred = self.base_models[name](tensor_X)
                    dl_predictions.append(pred.numpy())
                    
            # 合并所有预测
            horizon_predictions = np.column_stack([*ml_predictions, *dl_predictions])
            all_predictions.append(horizon_predictions)
            
        # 合并不同时间步长的预测
        self.logger.info("合并不同时间步长的预测...")
        combined_predictions = np.mean(all_predictions, axis=0)
        
        # 为了回测，我们使用原始收益率
        if self.original_prices is not None:
            true_returns = np.diff(self.original_prices)
            if len(true_returns) < len(combined_predictions):
                combined_predictions = combined_predictions[:len(true_returns)]
        else:
            true_returns = np.diff(df['Close'].values)
            
        # 选择最佳元模型
        self.best_meta_model = self.select_best_meta_model(combined_predictions, true_returns)
        
        # 获取元模型预测
        meta_model = self.meta_models[self.best_meta_model]
        meta_model.fit(combined_predictions, np.sign(true_returns))
        final_predictions = meta_model.predict_proba(combined_predictions)[:, 1]  # 获取正类概率
        
        # 生成优化的信号
        signals = self.generate_optimized_signals(final_predictions, true_returns)
        
        # 创建回测数据
        backtest_data = self.create_backtest_data(signals, self.original_prices, self.dates)
        
        return signals, self.best_meta_model, backtest_data

def main():
    # 创建signals目录
    os.makedirs('signals', exist_ok=True)
    
    # 读取所有货币对的数据
    pairs = ['CNYAUD', 'CNYEUR', 'CNYGBP', 'CNYJPY', 'CNYUSD']
    results = []
    
    for pair in pairs:
        logger.info(f"处理 {pair}...")
        
        # 读取PCA处理后的数据和原始数据
        pca_path = os.path.join('FE', f'{pair}_PCA.csv')
        original_path = os.path.join('..', 'try', 'data', f'{pair}.csv')
        
        if not os.path.exists(pca_path):
            logger.warning(f"未找到文件: {pca_path}")
            continue
            
        if not os.path.exists(original_path):
            logger.warning(f"未找到文件: {original_path}")
            continue
            
        try:
            # 读取数据
            df_pca = pd.read_csv(pca_path)
            df_original = pd.read_csv(original_path)
            
            # 确保日期列格式一致
            df_pca['Date'] = pd.to_datetime(df_pca['Date'])
            df_original['Date'] = pd.to_datetime(df_original['Date'])
            
            # 初始化模型
            model = MultiStepPredictor()
            
            # 训练模型并生成信号
            signals, best_model, backtest_data = model.train_and_predict(df_pca, df_original)
            
            # 保存回测数据
            backtest_path = os.path.join('signals', f'{pair}_backtest.csv')
            backtest_data.to_csv(backtest_path, index=False)
            
            # 保存信号
            signal_df = pd.DataFrame({
                'Date': backtest_data['Date'],
                'Price': backtest_data['Price'],
                'Signal': backtest_data['Signal']
            })
            
            # 创建signals目录（如果不存在）
            signals_dir = os.path.join('signals')
            os.makedirs(signals_dir, exist_ok=True)
            
            # 保存信号文件
            signal_path = os.path.join(signals_dir, f'{pair}_signals.csv')
            signal_df.to_csv(signal_path, index=False)
            
            # 计算性能指标
            cumulative_return = backtest_data['Cumulative_Strategy_Return'].iloc[-1]
            market_return = backtest_data['Cumulative_Market_Return'].iloc[-1]
            outperformance = cumulative_return - market_return
            
            # 记录结果
            results.append({
                'Pair': pair,
                'Best_Meta_Model': best_model,
                'Signal_Count': len(signals),
                'Buy_Signals': np.sum(signals == 1),
                'Sell_Signals': np.sum(signals == -1),
                'Hold_Signals': np.sum(signals == 0),
                'Cumulative_Return': cumulative_return,
                'Market_Return': market_return,
                'Outperformance': outperformance,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            logger.info(f"{pair} 处理完成，累积回报: {cumulative_return:.4f} (市场: {market_return:.4f})")
            
        except Exception as e:
            logger.error(f"处理 {pair} 时出错: {str(e)}", exc_info=True)
            continue
    
    if results:
        # 保存分析结果
        summary_path = os.path.join('signals', 'analysis_summary.csv')
        pd.DataFrame(results).to_csv(summary_path, index=False)
        logger.info("分析完成!")
    else:
        logger.warning("未生成结果!")

if __name__ == "__main__":
    main()
