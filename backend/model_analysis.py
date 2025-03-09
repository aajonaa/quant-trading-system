import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.kernel_ridge import KernelRidge
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import json
import shap
from pylab import mpl
import traceback
from statsmodels.tsa.seasonal import STL
import tensorflow.keras.backend as K
from PyEMD import CEEMDAN
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.saving import register_keras_serializable
from sklearn.utils.class_weight import compute_class_weight


mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
class HybridForexModel:
    """混合外汇预测模型：结合深度学习和机器学习"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dl_models = {}  # 深度学习模型
        self.ml_models = {}  # 机器学习模型
        self.meta_models = {}  # 元模型（用于堆叠集成）
        self.scalers = {}  # 特征标准化器
        self.is_trained = False
        self.currency_pairs = ["CNYAUD", "CNYEUR"]

    def _create_model(self, sequence_length=10, n_features=6, static_features=5, pair=None,
                     lstm_units=32, dropout_rate=0.8, learning_rate=0.005, l2_reg=1e-4):
        """创建改进的混合模型架构，加入CNN处理时间序列"""
        try:
            # 序列输入
            sequence_input = Input(shape=(sequence_length, n_features), name='sequence_input')

            # 增强的正则化参数
            regularizer = tf.keras.regularizers.L1L2(l1=3e-3, l2=l2_reg)

            # === CNN分支 ===
            # 添加多尺度CNN层以捕获不同时间跨度的模式
            conv1 = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizer
            )(sequence_input)
            conv1 = tf.keras.layers.BatchNormalization()(conv1)

            conv2 = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=4,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizer
            )(sequence_input)
            conv2 = tf.keras.layers.BatchNormalization()(conv2)

            conv3 = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=4,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizer
            )(sequence_input)
            conv3 = tf.keras.layers.BatchNormalization()(conv3)

            # 合并多尺度卷积结果
            cnn_features = tf.keras.layers.concatenate([conv1, conv2, conv3])
            cnn_features = tf.keras.layers.Dropout(0.4)(cnn_features)

            # === RNN分支 ===
            # 双向LSTM处理
            lstm1 = tf.keras.layers.Bidirectional(
                LSTM(lstm_units, return_sequences=True,
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer,
                     recurrent_dropout=dropout_rate)
            )(sequence_input)
            lstm1 = tf.keras.layers.LayerNormalization()(lstm1)
            lstm1 = tf.keras.layers.Dropout(dropout_rate)(lstm1)

            # 修改注意力机制的维度
            attention = tf.keras.layers.Dense(1, activation='tanh')(lstm1)
            attention = tf.keras.layers.Reshape((sequence_length,))(attention)
            attention = tf.keras.layers.Activation('softmax')(attention)
            attention = tf.keras.layers.RepeatVector(lstm_units * 2)(attention)  # 修改这里，使用lstm_units * 2
            attention = tf.keras.layers.Permute([2, 1])(attention)

            weighted_lstm = tf.keras.layers.Multiply()([lstm1, attention])

            # 第二层LSTM
            lstm2 = tf.keras.layers.Bidirectional(
                LSTM(lstm_units // 2, kernel_regularizer=regularizer)  # 使用lstm_units的一半
            )(weighted_lstm)
            lstm2 = tf.keras.layers.LayerNormalization()(lstm2)
            lstm2 = tf.keras.layers.Dropout(dropout_rate)(lstm2)

            # === CNN和RNN特征融合 ===
            # 聚合CNN特征
            cnn_pool = tf.keras.layers.GlobalAveragePooling1D()(cnn_features)

            # 组合CNN和RNN特征
            sequence_features = tf.keras.layers.concatenate([cnn_pool, lstm2])
            sequence_features = tf.keras.layers.Dense(128, activation='relu')(sequence_features)
            sequence_features = tf.keras.layers.BatchNormalization()(sequence_features)
            sequence_features = tf.keras.layers.Dropout(dropout_rate)(sequence_features)

            # === 静态特征处理分支 ===
            static_input = Input(shape=(static_features,), name='static_input')

            dense1 = Dense(128, activation='selu', kernel_regularizer=regularizer)(static_input)
            dense1 = tf.keras.layers.BatchNormalization()(dense1)
            dense1 = tf.keras.layers.Dropout(0.3)(dense1)

            dense2 = Dense(128, activation='selu', kernel_regularizer=regularizer)(dense1)
            dense2 = tf.keras.layers.BatchNormalization()(dense2)
            dense2 = tf.keras.layers.Dropout(0.5)(dense2)

            # === 特征融合层 ===
            merged = concatenate([sequence_features, dense2])
            merged = tf.keras.layers.Dropout(0.4)(merged)

            # 深度全连接层
            merged = tf.keras.layers.Dense(128, activation='relu')(merged)
            merged = tf.keras.layers.BatchNormalization()(merged)
            merged = tf.keras.layers.Dropout(0.3)(merged)

            merged = tf.keras.layers.Dense(128, activation='relu')(merged)
            merged = tf.keras.layers.BatchNormalization()(merged)
            merged = tf.keras.layers.Dropout(0.4)(merged)

            # 输出层
            output = Dense(3, activation='softmax', name='output')(merged)

            # 创建模型
            model = Model(inputs=[sequence_input, static_input], outputs=output)

            # 使用AdamW优化器
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=l2_reg
            )

            # 编译模型
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            return model
            
        except Exception as e:
            self.logger.error(f"创建模型时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def _prepare_features(self, data_dict, sequence_length=15):
        """增强的特征工程与选择"""
        features = {}

        for pair, df in data_dict.items():
            try:
                # 复制DataFrame以避免修改原始数据
                data = df.copy()

                # ==== 1. 基础时间特征 ====
                # 删除周期性编码相关代码

                # ==== 2. 添加lag特征 ====
                for days in [1, 3, 5, 7, 14, 21]:
                    data[f'price_change_{days}d'] = data['Close'].pct_change(periods=days)

                # ==== 3. 添加季节性特征 ====
                stl = STL(data['Close'], period=365)
                res = stl.fit()
                data['seasonal'] = res.seasonal
                data['trend'] = res.trend
                data['residual'] = res.resid

                # ==== 4. 添加时间窗口统计特征 ====
                for window in [5, 10, 20, 30]:
                    data[f'mean_{window}d'] = data['Close'].rolling(window=window).mean()
                    data[f'std_{window}d'] = data['Close'].rolling(window=window).std()
                    data[f'max_{window}d'] = data['Close'].rolling(window=window).max()
                    data[f'min_{window}d'] = data['Close'].rolling(window=window).min()

                # ==== 2. 价格特征增强 ====
                # 归一化价格
                data['norm_open'] = data['Open'] / data['Close'].shift(1)
                data['norm_high'] = data['High'] / data['Close'].shift(1)
                data['norm_low'] = data['Low'] / data['Close'].shift(1)
                data['norm_close'] = data['Close'] / data['Close'].shift(1)

                # 跳空缺口和高低价差
                data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
                data['hl_ratio'] = (data['High'] - data['Low']) / data['Close']
                data['ho_ratio'] = (data['High'] - data['Open']) / data['Open']
                data['lo_ratio'] = (data['Low'] - data['Open']) / data['Open']
                data['co_ratio'] = (data['Close'] - data['Open']) / data['Open']

                # ==== 3. 技术指标扩展 ====
                # RSI多周期
                for window in [6, 14, 21]:
                    delta = data['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=window).mean()
                    avg_loss = loss.rolling(window=window).mean()
                    rs = avg_gain / avg_loss
                    data[f'rsi_{window}'] = 100 - (100 / (1 + rs))

                # 移动平均多周期
                for window in [5, 10, 20, 50, 100]:
                    data[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
                    if window < 50:  # 中短期MA距离
                        data[f'ma_dist_{window}'] = (data['Close'] - data[f'ma_{window}']) / data[f'ma_{window}']

                # 移动平均交叉
                data['ma5_10'] = (data['ma_5'] - data['ma_10']) / data['Close']
                data['ma10_20'] = (data['ma_10'] - data['ma_20']) / data['Close']
                data['ma20_50'] = (data['ma_20'] - data['ma_50']) / data['Close']

                # 趋势指标
                for window in [5, 10, 20]:
                    data[f'trend_{window}'] = data['Close'].diff(window) / data['Close']

                # 波动率指标
                for window in [5, 10, 20, 50]:
                    data[f'volatility_{window}'] = data['Close'].rolling(window=window).std() / data['Close']

                # 布林带指标
                bb_window = 20
                data['bb_middle'] = data['Close'].rolling(window=bb_window).mean()
                data['bb_std'] = data['Close'].rolling(window=bb_window).std()
                data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
                data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                data['bb_pos'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

                # MACD指标
                data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['macd'] = data['ema12'] - data['ema26']
                data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
                data['macd_hist'] = data['macd'] - data['signal']
                data['macd_norm'] = data['macd_hist'] / data['Close']

                # ATR指标
                high_low = data['High'] - data['Low']
                high_close = (data['High'] - data['Close'].shift()).abs()
                low_close = (data['Low'] - data['Close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                data['atr'] = true_range.rolling(window=14).mean()
                data['atr_ratio'] = data['atr'] / data['Close']

                # ==== 4. 填充缺失值 ====
                data = data.ffill().bfill()  # 使用前向后向填充替代deprecated方法

                # ==== 5. 特征筛选 ====
                # 计算特征重要性(如果已有训练数据)
                feature_imp = {}
                if pair in self.ml_models and 'rf' in self.ml_models[pair]:
                    try:
                        # 获取已有的RF模型的特征重要性
                        rf_model = self.ml_models[pair]['rf']
                        feature_cols = list(data.columns)
                        for i, col in enumerate(feature_cols):
                            if col not in ['Open', 'High', 'Low', 'Close']:
                                try:
                                    feature_imp[col] = rf_model.feature_importances_[i]
                                except:
                                    feature_imp[col] = 0
                    except:
                        # 如果没有现有模型，则使用预定义的特征集
                        pass

                # 定义基础特征集
                base_sequence_cols = [
                    'norm_close', 'hl_ratio', 'gap', 'rsi_14', 'ma5_10', 'ma10_20',
                    'trend_10', 'volatility_20', 'bb_pos', 'macd_norm'
                ]

                base_static_cols = [
                    'rsi_14', 'ma5_10', 'ma10_20', 'ma_dist_20', 'volatility_20',
                    'bb_width', 'macd_norm', 'atr_ratio', 'gap'  # 移除 month_sin, month_cos
                ]

                # 动态特征选择(基于历史训练的重要性)
                if feature_imp:
                    # 选择最重要的特征
                    important_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                    # 取前20个最重要特征
                    additional_cols = [feat for feat, imp in important_features[:20]
                                      if feat not in base_sequence_cols + base_static_cols]

                    sequence_cols = base_sequence_cols + additional_cols[:5]  # 添加5个重要特征
                    static_cols = base_static_cols + additional_cols[5:10]    # 添加5个重要特征
                else:
                    # 没有历史数据时使用扩展特征集
                    extended_sequence = [
                        'ho_ratio', 'co_ratio', 'rsi_6', 'rsi_21', 'ma20_50',
                        'trend_5', 'trend_20', 'volatility_10'  # 移除 year_sin, year_cos
                    ]

                    extended_static = [
                        'rsi_6', 'rsi_21', 'ma20_50', 'trend_5', 'trend_20',
                        'volatility_10', 'bb_pos'  # 移除 day_sin, day_cos
                    ]

                    sequence_cols = base_sequence_cols + extended_sequence
                    static_cols = base_static_cols + extended_static

                # ==== 6. 创建特征序列 ====
                # 序列特征
                sequence_features = []
                for i in range(len(data) - sequence_length - 1):
                    sequence_features.append(data[sequence_cols].iloc[i:i+sequence_length].values)
                sequence_features = np.array(sequence_features)
                self.logger.info(f"序列特征: {sequence_features.shape}")

                # 静态特征
                static_features = data[static_cols].iloc[sequence_length:-1].reset_index(drop=True)
                self.logger.info(f"静态特征: {static_features.shape}")

                # 存储特征
                features[pair] = {
                    'sequence': sequence_features,
                    'static': static_features,
                    'feature_cols': {'sequence': sequence_cols, 'static': static_cols}
                }

            except Exception as e:
                self.logger.error(f"准备 {pair} 的特征时出错: {str(e)}")
                self.logger.error(traceback.format_exc())

        return features

    def _prepare_labels(self, df):
        """准备标签数据

        参数:
        - df: 原始数据DataFrame

        返回:
        - 标签数组
        """
        try:
            # 计算价格变化
            price_changes = df['Close'].pct_change()

            # 定义阈值
            threshold = 0.001  # 0.1%的变化阈值

            # 创建标签
            labels = np.zeros(len(price_changes))
            labels[price_changes > threshold] = 2  # 上涨
            labels[price_changes < -threshold] = 0  # 下跌
            labels[(price_changes >= -threshold) & (price_changes <= threshold)] = 1  # 持平

            # 移除第一个NaN值
            return labels[1:]
            
        except Exception as e:
            self.logger.error(f"准备标签时出错: {str(e)}")
            return None
            
    def _train_enhanced_models(self, pair, X_train_seq, X_train_static, y_train,
                              X_val_seq, X_val_static, y_val,
                              sequence_length, n_features, static_features,
                              models_dir, epochs=89, class_weight=None):
        """训练多个增强模型"""
        try:
            # 初始化模型字典
            if pair not in self.dl_models:
                self.dl_models[pair] = {
                    'base_models': [],  # 深度学习基础模型
                    'meta_model': None,  # 深度学习元模型
                    'final_model': None  # 最终集成模型
                }

            histories = []
            model_names = []

            # 基础回调函数
            base_callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    min_delta=0.0001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.000001,
                    min_delta=0.0005,
                    verbose=1
                )
            ]

            # ==== 训练基础模型 ====
            # 1. CNN-RNN 模型
            model1 = self._create_model(sequence_length, n_features, static_features, pair)
            history1 = model1.fit(
                [X_train_seq, X_train_static],
                y_train,
                validation_data=([X_val_seq, X_val_static], y_val),
                epochs=epochs,
                batch_size=256,
                callbacks=base_callbacks + [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=models_dir / f"{pair}_model1.keras",
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=1
                    ),
                    tf.keras.callbacks.LearningRateScheduler(
                        lambda epoch, lr: float(min(0.01, lr * np.exp(0.1))) if epoch < 3 else
                                        float(0.01 * (1 + np.cos(
                                            np.pi * (epoch - 3) / (epochs - 3)
                                        )))
                    )
                ],
                verbose=1
            )
            self.dl_models[pair]['base_models'].append(model1)
            histories.append(history1.history)
            model_names.append("CNN-RNN")

            # 2. DeepCNN 模型
            model2 = self._create_deep_cnn(
                sequence_shape=X_train_seq.shape[1:],  # 传入(sequence_length, n_features)
                static_features=X_train_static.shape[1]
            )
            history2 = model2.fit(
                [X_train_seq, X_train_static],
                y_train,
                validation_data=([X_val_seq, X_val_static], y_val),
                epochs=epochs,
                batch_size=256,
                callbacks=base_callbacks + [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=models_dir / f"{pair}_model2.keras",
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=1
                    )
                ],
                verbose=1
            )
            self.dl_models[pair]['base_models'].append(model2)
            histories.append(history2.history)
            model_names.append("DeepCNN")

            # 3. Transformer 模型
            model3 = self._create_transformer(sequence_length, n_features, static_features)
            history3 = model3.fit(
                [X_train_seq, X_train_static],
                y_train,
                validation_data=([X_val_seq, X_val_static], y_val),
                epochs=epochs,
                batch_size=258,
                callbacks=base_callbacks + [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=models_dir / f"{pair}_model3.keras",
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=1
                    )
                ],
                verbose=1
            )
            self.dl_models[pair]['base_models'].append(model3)
            histories.append(history3.history)
            model_names.append("Transformer")

            # ==== 第一层：深度学习模型集成 ====
            dl_train_predictions = []
            dl_val_predictions = []

            for model in self.dl_models[pair]['base_models']:
                train_pred = model.predict([X_train_seq, X_train_static])
                val_pred = model.predict([X_val_seq, X_val_static])
                dl_train_predictions.append(train_pred)
                dl_val_predictions.append(val_pred)

            # 创建深度学习集成的元模型
            dl_meta_model = GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.0001,
                max_depth=3,
                random_state=42
            )
            
            # 训练深度学习元模型
            dl_meta_features_train = np.concatenate(dl_train_predictions, axis=1)
            dl_meta_features_val = np.concatenate(dl_val_predictions, axis=1)

            dl_meta_model.fit(dl_meta_features_train, y_train)
            dl_meta_pred = dl_meta_model.predict_proba(dl_meta_features_train)

            # 保存元模型
            self.dl_models[pair]['meta_model'] = dl_meta_model

            # ==== 第二层：综合集成 ====
            # 准备基础特征
            base_features = np.concatenate([
                X_train_static,  # 静态特征
                X_train_seq.reshape(X_train_seq.shape[0], -1),  # 展平的序列特征
                dl_meta_pred  # 深度学习集成预测
            ], axis=1)

            val_base_features = np.concatenate([
                X_val_static,
                X_val_seq.reshape(X_val_seq.shape[0], -1),
                dl_meta_model.predict_proba(dl_meta_features_val)
            ], axis=1)

            # 定义机器学习基础分类器
            base_classifiers = [
                ('rf', RandomForestClassifier(
                    n_estimators=400,      #决策树数量
                    max_depth=2,
                    random_state=42
                )),
                ('kelm', KernelRidge(
                    kernel='rbf',
                    alpha=6,      #越大正则化越强
                    gamma=1      #越大越关注全局特征
                )),
                ('svm', SVC(
                kernel='rbf',
                probability=True,
                random_state=42
                ))
            ]

            # 创建最终的堆叠集成分类器
            final_stacking = StackingClassifier(
                estimators=base_classifiers,
                final_estimator=GradientBoostingClassifier(  # 使用GradientBoosting作为最终分类器
                    n_estimators=500,
                    learning_rate=0.001,
                    max_depth=2,
                    random_state=42

                ),
                cv=5,
                n_jobs=-1
            )

            # 训练最终集成模型
            final_stacking.fit(base_features, y_train)

            # 评估模型
            train_acc = accuracy_score(y_train, final_stacking.predict(base_features))
            val_acc = accuracy_score(y_val, final_stacking.predict(val_base_features))

            self.logger.info(f"{pair} 最终集成模型训练准确率: {train_acc:.4f}")
            self.logger.info(f"{pair} 最终集成模型验证准确率: {val_acc:.4f}")

            # 保存最终模型
            self.dl_models[pair]['final_model'] = final_stacking

            # 创建历史记录
            ensemble_history = {
                'accuracy': [train_acc],
                'val_accuracy': [val_acc]
            }
            histories.append(ensemble_history)

            return self.dl_models[pair]['base_models'], histories, model_names
            
        except Exception as e:
            self.logger.error(f"训练增强模型时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [], [], []

    def train(self, data_dict):
        """训练混合模型"""
        try:
            self.logger.info("开始训练混合模型...")

            for pair in self.currency_pairs:
                self.logger.info(f"处理 {pair} 数据...")

                # 准备训练数据
                X_seq, X_static, y = self._prepare_training_data(data_dict[pair], pair)

                if X_seq is None or X_static is None or y is None:
                    continue

                # 划分训练集和验证集
                split_idx = int(len(X_seq) * 0.8)
                X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
                X_static_train, X_static_val = X_static[:split_idx], X_static[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                self.logger.info("开始超参数优化...")
                best_params = self.optimize_hyperparameters(
                    X_seq_train, X_static_train, y_train,
                    X_seq_val, X_static_val, y_val
                )

                model = self._create_model(
                    sequence_length=X_seq.shape[1],
                    n_features=X_seq.shape[2],
                    static_features=X_static.shape[1],
                    lstm_units=int(best_params[0]),
                    dropout_rate=best_params[1],
                    learning_rate=best_params[2],
                    l2_reg=best_params[3]
                )

                # 创建必要的目录
                models_dir = Path("models")
                plots_dir = Path("plots")
                for directory in [models_dir, plots_dir]:
                    directory.mkdir(exist_ok=True, parents=True)

                # 训练增强模型
                base_models, histories, model_names = self._train_enhanced_models(
                    pair=pair,
                    X_train_seq=X_seq_train,
                    X_train_static=X_static_train,
                    y_train=y_train,
                    X_val_seq=X_seq_val,
                    X_val_static=X_static_val,
                    y_val=y_val,
                    sequence_length=X_seq.shape[1],
                    n_features=X_seq.shape[2],
                    static_features=X_static.shape[1],
                    models_dir=models_dir,
                    epochs=68
                )

                # 绘制每个模型的训练历史
                for history, model_name in zip(histories, model_names):
                    plt.figure(figsize=(12, 4))

                    # 绘制损失
                    plt.subplot(1, 2, 1)
                    plt.plot(history['loss'], label='Training Loss')
                    if 'val_loss' in history:
                        plt.plot(history['val_loss'], label='Validation Loss')
                    plt.title(f'{pair} {model_name} Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()

                    # 绘制准确率
                    plt.subplot(1, 2, 2)
                    plt.plot(history['accuracy'], label='Training Accuracy')
                    if 'val_accuracy' in history:
                        plt.plot(history['val_accuracy'], label='Validation Accuracy')
                    plt.title(f'{pair} {model_name} Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(plots_dir / f'{pair}_{model_name}_history.png')
                    plt.close()

                # 评估模型性能
                for model, model_name in zip(base_models, model_names):
                    # 在验证集上评估
                    val_loss, val_acc = model.evaluate(
                        [X_seq_val, X_static_val],
                        y_val,
                        verbose=0
                    )
                    self.logger.info(f"{pair} {model_name} - 验证集准确率: {val_acc:.4f}")

                    # 生成预测
                    y_pred = model.predict([X_seq_val, X_static_val])
                    y_pred_classes = np.argmax(y_pred, axis=1)

                    # 绘制混淆矩阵
                    cm = confusion_matrix(y_val, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'{pair} {model_name} Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.savefig(plots_dir / f'{pair}_{model_name}_confusion_matrix.png')
                    plt.close()

                self.logger.info(f"{pair} 训练完成")

            self.logger.info("所有模型训练完成")
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"训练过程中出错: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _prepare_stacking_features(self, pair, X_seq, X_static, y, train_idx, val_idx):
        """准备堆叠集成的特征"""
        try:
            if pair not in self.dl_models:
                raise ValueError(f"模型 {pair} 未训练")

            # 获取深度学习模型的预测概率
            dl_probas = []
            for model_name, model in self.dl_models[pair].items():
                proba = model.predict([X_seq, X_static], verbose=0)
                dl_probas.append(proba)

            # 准备机器学习特征
            flattened_features = self._flatten_features(X_seq, X_static)

            # 组合所有特征
            all_features = [flattened_features] + dl_probas

            return np.hstack(all_features)

        except Exception as e:
            self.logger.error(f"准备堆叠特征时出错: {str(e)}")
            return None

    def _train_ml_models(self, features, labels, pair):
        """训练多个机器学习模型"""
        try:
            self.logger.info(f"训练 {pair} 机器学习模型...")

            # 标准化特征
            scaler = StandardScaler()
            X = scaler.fit_transform(features)
            self.scalers[pair] = scaler

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)

            # 建立多种分类器
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=500,
                    max_depth=2,
                    min_samples_split=5,
                    min_samples_leaf=4,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=0.0001,
                    max_depth=2,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    subsample=0.1,     #每棵树选取样本的样本百分比，越大则越可能过拟合
                    random_state=42
                ),
                'svm': SVC(
                    C=0.01,     #较小的值增加正则化强度
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
            }

            # 训练每个模型并评估
            for name, model in models.items():
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    cv_scores.append(score)

                self.logger.info(f"{pair} {name} 交叉验证平均得分: {np.mean(cv_scores):.4f}")

                # 重新在全部数据上训练
                model.fit(X, labels)

            # 创建投票分类器，使用软投票模式
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft',
                weights=[0.6, 0.2, 0.2]  # 给RF更高的权重
            )

            # 训练集成模型
            ensemble.fit(X, labels)

            # 保存所有模型
            self.ml_models[pair] = models
            self.ml_models[pair]['ensemble'] = ensemble

            self.logger.info(f"{pair} 机器学习模型训练完成")
            
        except Exception as e:
            self.logger.error(f"训练机器学习模型时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def _calculate_correlation_matrix(self, data):
        """计算货币对之间的相关性矩阵"""
        try:
            # 创建收盘价DataFrame
            prices = pd.DataFrame()

            # 获取每个货币对的收盘价
            for pair, df in data.items():
                prices[pair] = df['Close']

            # 计算每日收益率
            returns = prices.pct_change().dropna()
                
            # 计算相关性矩阵
            correlation_matrix = returns.corr()

            # 绘制相关性热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('货币对相关性矩阵')

            # 保存图片
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(plots_dir / 'currency_correlation.png')
            plt.close()

            self.logger.info("货币对相关性矩阵:")
            self.logger.info(f"\n{correlation_matrix}")

            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"计算相关性矩阵时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            # 返回空的相关性矩阵
            pairs = list(data.keys())
            return pd.DataFrame(np.eye(len(pairs)), index=pairs, columns=pairs)

    def _generate_multi_currency_signal(self, signals, corr_matrix):
        """
        根据单一货币对信号和相关性矩阵生成多货币对风险信号

        策略:
        1. 高度正相关的货币对同向信号意味着较高风险敞口
        2. 负相关的货币对反向信号提供对冲效果
        3. 加权组合信号形成最终风险指标
        """
        try:
            # 结果字典
            multi_signals = {}

            # 货币对列表
            pairs = list(signals.keys())

            # 确保所有信号长度一致
            min_signal_length = min(len(signal) for signal in signals.values())
            aligned_signals = {pair: signals[pair][:min_signal_length] for pair in pairs}

            # 为每对货币组合生成风险信号
            for i, pair1 in enumerate(pairs):
                for j, pair2 in enumerate(pairs):
                    if j <= i:  # 避免重复组合
                        continue

                    # 获取相关性
                    correlation = corr_matrix.loc[pair1, pair2]
                    self.logger.info(f"{pair1}-{pair2} 相关性: {correlation:.4f}")

                    # 创建组合键名
                    combo_key = f"{pair1}_{pair2}"

                    # 风险敞口计算
                    # 1. 如果高度正相关(>0.7)且信号同向，风险增加
                    # 2. 如果高度负相关(<0.7)且信号反向，风险减少(对冲效果)
                    # 3. 相关性较低时(-0.3~0.3)，风险相对中性

                    signal1 = aligned_signals[pair1]
                    signal2 = aligned_signals[pair2]

                    # 计算多货币风险敞口指标
                    exposure = np.zeros(min_signal_length)

                    for k in range(min_signal_length):
                        # 信号是否同向
                        same_direction = (signal1[k] * signal2[k] > 0)
                        # 信号是否反向
                        opposite_direction = (signal1[k] * signal2[k] < 0)

                        # 基于相关性和信号方向计算风险敞口
                        if correlation > 0.7 and same_direction:
                            # 高正相关 + 同向信号 = 高风险敞口
                            exposure[k] = 2  # 高风险
                        elif correlation < -0.7 and opposite_direction:
                            # 高负相关 + 反向信号 = 低风险(对冲效果)
                            exposure[k] = 0  # 低风险
                        elif abs(correlation) < 0.3:
                            # 低相关 = 中等风险
                            exposure[k] = 1  # 中等风险
                        else:
                            # 其他情况
                            # 计算加权风险得分
                            risk_score = (abs(signal1[k]) + abs(signal2[k])) * (0.5 + abs(correlation)/2)
                            exposure[k] = np.clip(risk_score, 0, 2).astype(int)  # 限制在0-2范围内

                    # 将敞口值调整为信号(-1, 0, 1)
                    multi_signals[combo_key] = exposure - 1

            return multi_signals

        except Exception as e:
            self.logger.error(f"生成多货币对信号时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def predict(self, data_dict):
        """预测所有货币对的走势"""
        try:
            predictions = {}
            signals = {}
            feature_importance = {}

            for pair in self.dl_models:
                self.logger.info(f"\n预测 {pair} 的走势...")

                # 准备预测数据
                X_seq, X_static = self._prepare_prediction_data(data_dict[pair])

                if X_seq is None or X_static is None:
                    continue

                # 获取深度学习模型预测
                dl_predictions = []
                for model in self.dl_models[pair]['base_models']:
                    pred = model.predict([X_seq, X_static])
                    dl_predictions.append(pred)

                # 合并深度学习预测结果
                dl_meta_features = np.concatenate(dl_predictions, axis=1)
                
                # 使用元模型进行预测
                meta_pred = self.dl_models[pair]['meta_model'].predict_proba(dl_meta_features)
                
                # 准备最终预测的特征
                final_features = np.concatenate([
                    X_static,
                    X_seq.reshape(X_seq.shape[0], -1),
                    meta_pred
                ], axis=1)
                
                # 最终预测
                final_pred = self.dl_models[pair]['final_model'].predict(final_features)
                
                # 存储预测结果
                predictions[pair] = {
                    'dl_models': dl_predictions,
                    'meta_model': meta_pred,
                    'ensemble': final_pred
                }
                
                # 生成交易信号
                signals[pair] = pd.Series(final_pred, index=data_dict[pair].index[-len(final_pred):])
                
                # 计算特征重要性（如果是随机森林或梯度提升树模型）
                if hasattr(self.dl_models[pair]['final_model'], 'feature_importances_'):
                    feature_importance[pair] = self.dl_models[pair]['final_model'].feature_importances_

            return predictions, signals, feature_importance

        except Exception as e:
            self.logger.error(f"预测过程中出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None, None

    def _generate_signals(self, predictions, probabilities, threshold=0.6):
        """根据预测结果和概率生成交易信号"""
        signals = pd.DataFrame(index=dates)
        signals['prediction'] = predictions
        signals['probability'] = np.max(probabilities, axis=1)

        # 生成交易信号
        # 0: 持有, 1: 买入, 2: 卖出
        signals['signal'] = 0  # 默认持有

        # 当预测概率超过阈值时生成信号
        high_conf_mask = signals['probability'] >= threshold

        # 买入信号
        buy_mask = (predictions == 1) & high_conf_mask
        signals.loc[buy_mask, 'signal'] = 1

        # 卖出信号
        sell_mask = (predictions == 2) & high_conf_mask
        signals.loc[sell_mask, 'signal'] = 2
                    
        return signals
            
    def save_models(self, models_dir):
        """保存所有模型，包括深度学习模型和最终的集成模型"""
        try:
            for pair in self.dl_models:
                # 保存深度学习模型
                for i, model in enumerate(self.dl_models[pair]['base_models']):  # 排除最后的集成模型
                    model.save(models_dir / f"{pair}_model{i+1}.keras")

                # 保存深度学习元模型
                self.dl_models[pair]['meta_model'].save(models_dir / f"{pair}_dl_meta.keras")

                # 保存最终的集成模型
                final_model = self.dl_models[pair]['final_model']
                joblib.dump(final_model, models_dir / f"{pair}_final_ensemble.joblib")

            self.logger.info("所有模型保存完成")
            return True

        except Exception as e:
            self.logger.error(f"保存模型时出错: {str(e)}")
            return False

    def _get_feature_importance(self, pair):
        """获取特征重要性

        参数:
        - pair: 货币对名称

        返回:
        - 特征重要性字典
        """
        try:
            # 获取基础特征名称
            feature_names = [
                'Open', 'High', 'Low',
                'MA5', 'MA10', 'RSI', 'MACD',
                'seasonal', 'trend', 'residual',
                'mean_5d', 'std_5d', 'max_5d', 'min_5d',
                'price_change_1d', 'price_change_7d', 'price_change_21d'
            ]

            # 初始化综合特征重要性
            importance_dict = {}

            # 从各个模型获取特征重要性
            if pair in self.ml_models:
                importances = []
                weights = []

                # 随机森林特征重要性
                if 'rf' in self.ml_models[pair]:
                    rf_model = self.ml_models[pair]['rf']
                    rf_importance = rf_model.feature_importances_
                    importances.append(rf_importance)
                    weights.append(0.5)  # RF权重

                # 梯度提升特征重要性
                if 'gb' in self.ml_models[pair]:
                    gb_model = self.ml_models[pair]['gb']
                    gb_importance = gb_model.feature_importances_
                    importances.append(gb_importance)
                    weights.append(0.3)  # GB权重

                # 堆叠集成模型特征重要性
                if pair in self.meta_models:
                    meta_model = self.meta_models[pair]
                    if hasattr(meta_model, 'feature_importances_'):
                        meta_importance = meta_model.feature_importances_
                        importances.append(meta_importance)
                        weights.append(0.2)  # Meta模型权重

                if importances:
                    # 计算加权平均特征重要性
                    weights = np.array(weights) / sum(weights)
                    combined_importance = np.zeros_like(importances[0])
                    for imp, w in zip(importances, weights):
                        combined_importance += imp * w

                    # 确保特征数量匹配
                    n_features = len(combined_importance)
                    feature_names = feature_names[:n_features]

                    # 创建特征重要性字典
                    for name, importance in zip(feature_names, combined_importance):
                        importance_dict[name] = float(importance)

                    # 检查是否有有效的特征重要性值
                    if any(importance_dict.values()):
                        # 绘制特征重要性图
                        self._plot_feature_importance(importance_dict, pair)

                        # 记录特征重要性排名
                        self.logger.info(f"\n{pair} 特征重要性排名:")
                        for name, value in sorted(importance_dict.items(),
                                               key=lambda x: x[1],
                                               reverse=True):
                            self.logger.info(f"{name}: {value:.4f}")

                        return importance_dict

                    self.logger.warning(f"{pair} 没有有效的特征重要性值")
            return None
            
            self.logger.warning(f"{pair} 没有可用的模型来计算特征重要性")
            return None

        except Exception as e:
            self.logger.error(f"获取特征重要性时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
        return None

    def _plot_feature_importance(self, importance_dict, pair, save_path='plots'):
        """绘制特征重要性图

        参数:
        - importance_dict: 特征重要性字典
        - pair: 货币对名称
        - save_path: 保存路径
        """
        try:
            if not importance_dict:
                self.logger.warning(f"{pair} 特征重要性字典为空")
                return

            # 创建保存目录
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 排序特征重要性
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features)

            # 检查数据
            if not features or not importances:
                self.logger.warning(f"{pair} 特征或重要性值为空")
                return

            # 创建图形
            plt.figure(figsize=(12, 8))

            # 创建水平条形图
            y_pos = np.arange(len(features))
            bars = plt.barh(y_pos, importances)

            # 设置y轴标签
            plt.yticks(y_pos, features)

            # 添加标题和标签
            plt.title(f'{pair} 集成模型特征重要性分析', fontsize=12)
            plt.xlabel('重要性得分', fontsize=10)

            # 为每个条形添加数值标签和颜色
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(importance, i, f'{importance:.4f}',
                        va='center', ha='left', fontsize=8)
                # 根据重要性设置颜色深浅
                bar.set_alpha(0.4 + 0.6 * importance / max(importances))
                # 设置渐变色
                bar.set_color(plt.cm.viridis(importance / max(importances)))

            # 添加网格线
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)

            # 调整布局
            plt.tight_layout()

            # 保存图片
            save_path = save_dir / f'{pair}_feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"特征重要性图已保存到: {save_path}")

        except Exception as e:
            self.logger.error(f"绘制特征重要性图时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _save_signals_to_csv(self, single_signals, multi_signals, data):
        """保存信号到CSV文件，包含时间列和收盘价列"""
        try:
            # 创建signals文件夹
            signals_dir = Path("signals")
            signals_dir.mkdir(exist_ok=True, parents=True)

            # 从第一个货币对获取时间索引
            first_pair = list(data.keys())[0]
            dates = data[first_pair].index

            # 创建基础DataFrame，包含时间列
            signals_df = pd.DataFrame(index=dates[-len(single_signals[first_pair]):])

            # 为每个货币对添加信号和收盘价
            for pair, signal in single_signals.items():
                # 添加信号
                signals_df[f'{pair}_signal'] = signal
                # 添加收盘价
                signals_df[f'{pair}_close'] = data[pair]['Close'].values[-len(signal):]

            # 添加多货币对信号和风险指标
            for pair_combo, signal in multi_signals.items():
                signals_df[f'{pair_combo}_signal'] = signal

                # 添加风险指标解释列
                risk_explanation = []
                for s in signal:
                    if s == -1:
                        risk_explanation.append("低风险")
                    elif s == 0:
                        risk_explanation.append("中等风险")
                    else:  # s == 1
                        risk_explanation.append("高风险")
                signals_df[f'{pair_combo}_risk'] = risk_explanation

            # 重置索引，将日期作为列
            signals_df.reset_index(inplace=True)
            signals_df.rename(columns={'index': 'date'}, inplace=True)

            # 保存到CSV
            signals_df.to_csv(signals_dir / 'forex_signals.csv', index=False)
            self.logger.info(f"信号已保存到 signals/forex_signals.csv")

            # 额外保存一个包含解释的MD文件
            with open(signals_dir / 'signal_explanation.md', 'w') as f:
                f.write("# 外汇信号解释\n\n")
                f.write("## 单一货币对信号\n")
                f.write("* -1: 看跌信号\n")
                f.write("* 0: 中性信号\n")
                f.write("* 1: 看涨信号\n\n")

                f.write("## 多货币对风险信号\n")
                f.write("* -1: 低风险敞口 - 可能存在良好的对冲效果\n")
                f.write("* 0: 中等风险敞口 - 风险在可接受范围内\n")
                f.write("* 1: 高风险敞口 - 多个货币对可能同向波动，风险集中\n\n")

                f.write("## 货币对相关性分析\n")
                try:
                    corr_matrix = self._calculate_correlation_matrix(data)
                    f.write("```\n")
                    f.write(f"{corr_matrix.to_string()}\n")
                    f.write("```\n\n")

                    f.write("### 相关性解释\n")
                    f.write("* 高度正相关(>0.7): 货币对可能同向波动\n")
                    f.write("* 高度负相关(<0.7): 货币对可能反向波动，提供对冲效果\n")
                    f.write("* 低相关(-0.3~0.3): 货币对波动关系较弱\n")
                except:
                    f.write("相关性分析不可用\n")

            self.logger.info(f"信号解释已保存到 signals/signal_explanation.md")

        except Exception as e:
            self.logger.error(f"保存信号到CSV时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def analyze_shap_values(self, model, features, pair, plot_path='plots'):
        """分析SHAP值"""
        try:
            if not features or pair not in features:
                raise ValueError(f"无效的特征数据或货币对: {pair}")

            # 获取特征数据
            X = features[pair]['static'].values  # 转换为numpy数组
            feature_names = features[pair]['static'].columns

            # 创建预测函数
            def model_predict(x):
                # 重塑输入以匹配模型期望的形状
                if len(x.shape) == 2:
                    batch_size = x.shape[0]
                    sequence_data = np.zeros((batch_size, 10, x.shape[1]))
                    pred = model.predict([sequence_data, x])
                    return pred[0] if isinstance(pred, list) else pred
                else:
                    sequence_data = np.zeros((1, 10, x.shape[0]))
                    pred = model.predict([sequence_data, x.reshape(1, -1)])
                    return pred[0] if isinstance(pred, list) else pred

            # 创建SHAP解释器
            sample_size = min(100, X.shape[0])
            background = shap.sample(X, sample_size)
            explainer = shap.KernelExplainer(model_predict, background)

            # 计算SHAP值
            shap_values = explainer.shap_values(X[:sample_size])

            # 处理SHAP值
            if isinstance(shap_values, list):
                # 多分类情况，取第一个类别的SHAP值
                shap_values = shap_values[0]

            # 计算特征重要性
            feature_importance = {}
            for i, name in enumerate(feature_names):
                importance = np.abs(shap_values[:, i]).mean()
                feature_importance[name] = float(importance)

            # 绘制SHAP摘要图
            if plot_path:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X[:sample_size],
                    feature_names=list(feature_names),
                    plot_type="bar",
                    show=False
                )
                plt.title(f"{pair} SHAP Feature Importance")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

            return feature_importance

        except Exception as e:
            self.logger.error(f"计算SHAP值时出错: {str(e)}")
            return None

    def evaluate_predictions(self, predictions, true_labels):
        """评估预测结果"""
        try:
            # 将预测概率转换为类别
            pred_classes = np.argmax(predictions, axis=1) - 1  # -1, 0, 1

            # 基础指标
            metrics = {
                'accuracy': accuracy_score(true_labels, pred_classes),
                'precision': precision_score(true_labels, pred_classes, average='weighted'),
                'recall': recall_score(true_labels, pred_classes, average='weighted'),
                'f1': f1_score(true_labels, pred_classes, average='weighted')
            }

            # 添加混淆矩阵
            cm = confusion_matrix(true_labels, pred_classes)
            metrics['confusion_matrix'] = cm.tolist()

            # 添加类别准确率
            for i, label in enumerate([-1, 0, 1]):
                mask = true_labels == label
                if mask.any():
                    metrics[f'accuracy_class_{label}'] = accuracy_score(
                        true_labels[mask],
                        pred_classes[mask]
                    )

            # 添加ROC曲线和AUC
            for i, label in enumerate([-1, 0, 1]):
                y_true_binary = (true_labels == label).astype(int)
                y_score = predictions[:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                metrics[f'auc_class_{label}'] = auc(fpr, tpr)
                metrics[f'roc_curve_class_{label}'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }

            return metrics

        except Exception as e:
            self.logger.error(f"评估预测时出错: {str(e)}")
            return None

    def plot_evaluation_results(self, metrics, pair, save_dir='plots'):
        """可视化评估结果"""
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 1. 混淆矩阵热图
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Down', 'Neutral', 'Up'],
                yticklabels=['Down', 'Neutral', 'Up']
            )
            plt.title(f'{pair} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(save_dir / f'{pair}_confusion_matrix.png')
            plt.close()

            # 2. ROC曲线
            plt.figure(figsize=(10, 6))
            for label in [-1, 0, 1]:
                label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
                roc_data = metrics[f'roc_curve_class_{label}']
                plt.plot(
                    roc_data['fpr'],
                    roc_data['tpr'],
                    label=f'{label_name} (AUC = {metrics[f"auc_class_{label}"]:.2f})'
                )
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{pair} ROC Curves')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f'{pair}_roc_curves.png')
            plt.close()

            # 3. 类别准确率条形图
            plt.figure(figsize=(8, 6))
            class_acc = {
                'Down': metrics['accuracy_class_-1'],
                'Neutral': metrics['accuracy_class_0'],
                'Up': metrics['accuracy_class_1']
            }
            plt.bar(class_acc.keys(), class_acc.values())
            plt.title(f'{pair} Class-wise Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            for i, v in enumerate(class_acc.values()):
                plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            plt.tight_layout()
            plt.savefig(save_dir / f'{pair}_class_accuracy.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"绘制评估结果时出错: {str(e)}")

    def cross_validate(self, data, n_splits=5):
        """执行交叉验证"""
        try:
            cv_results = {}

            for pair in data.keys():
                # 准备数据
                features = self._prepare_features(data[pair])
                labels = self._prepare_labels(data[pair])

                if features is None or labels is None:
                    continue

                # 创建时间序列分割器
                tscv = TimeSeriesSplit(n_splits=n_splits)

                # 存储每次分割的结果
                fold_metrics = []

                for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
                    # 分割数据
                    X_train = features.iloc[train_idx]
                    X_test = features.iloc[test_idx]
                    y_train = labels[train_idx]
                    y_test = labels[test_idx]

                    # 训练模型
                    model = self._create_model()
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        verbose=0
                    )

                    # 评估模型
                    predictions = model.predict(X_test)
                    metrics = self.evaluate_predictions(predictions, y_test)

                    if metrics:
                        fold_metrics.append(metrics)

                # 计算平均指标
                cv_results[pair] = {
                    metric: np.mean([fold[metric] for fold in fold_metrics])
                    for metric in fold_metrics[0].keys()
                    if not isinstance(fold_metrics[0][metric], (list, dict))
                }

                # 记录交叉验证结果
                self.logger.info(f"\n{pair} 交叉验证结果 ({n_splits} 折):")
                for metric, value in cv_results[pair].items():
                    self.logger.info(f"{metric}: {value:.4f}")

            return cv_results

        except Exception as e:
            self.logger.error(f"执行交叉验证时出错: {str(e)}")
            return None

    def _flatten_features(self, X_seq, X_static):
        """将序列特征和静态特征扁平化为机器学习模型可以使用的格式，并确保维度与训练时一致

        参数:
        - X_seq: 序列特征，形状为 (samples, sequence_length, features)
        - X_static: 静态特征，形状为 (samples, features)

        返回:
        - 扁平化后的特征矩阵，形状为 (samples, 20)
        """
        # 序列特征的简单统计量
        seq_mean = np.mean(X_seq, axis=1)  # 均值特征 (samples, features)
        seq_std = np.std(X_seq, axis=1)    # 标准差特征 (samples, features)
        seq_min = np.min(X_seq, axis=1)    # 最小值特征 (samples, features)
        seq_max = np.max(X_seq, axis=1)    # 最大值特征 (samples, features)

        # 合并所有提取的特征
        all_features = np.hstack([seq_mean, seq_std, seq_min, seq_max, X_static])

        # 确保输出维度为20
        if all_features.shape[1] > 20:
            return all_features[:, :20]  # 截取前20个特征
        elif all_features.shape[1] < 20:
            # 如果特征不够，用零填充
            padding = np.zeros((all_features.shape[0], 20 - all_features.shape[1]))
            return np.hstack([all_features, padding])
        else:
            return all_features

    def ensemble_predict(self, pair, X_seq, X_static):
        """使用各个模型进行预测并返回集成结果

        参数:
        - pair: 货币对名称
        - X_seq: 序列特征
        - X_static: 静态特征

        返回:
        - 预测类别
        - 预测概率
        """
        # 准备特征
        flattened_features = self._flatten_features(X_seq, X_static)

        # 获取模型
        rf_model = self.ml_models[pair]['rf']
        gb_model = self.ml_models[pair]['gb']
        svm_model = self.ml_models[pair]['svm']

        # 机器学习模型预测
        rf_pred = rf_model.predict(flattened_features)
        gb_pred = gb_model.predict(flattened_features)
        svm_pred = svm_model.predict(flattened_features)

        rf_proba = rf_model.predict_proba(flattened_features)
        gb_proba = gb_model.predict_proba(flattened_features)
        svm_proba = svm_model.predict_proba(flattened_features)

        # 深度学习模型预测
        dl_pred = {}
        dl_proba = {}

        for model_name in ['cnn_rnn', 'deep_cnn', 'transformer']:
            if model_name in self.dl_models[pair]:
                model = self.dl_models[pair][model_name]['model']
                # 预测
                predictions = model.predict([X_seq, X_static], verbose=0)
                # 获取类别和概率
                dl_pred[model_name] = np.argmax(predictions, axis=1)
                dl_proba[model_name] = predictions

        # 集成预测（平均概率）
        ensemble_proba = np.zeros_like(rf_proba)
        count = 1  # 至少有rf_proba

        # 添加所有可用模型的概率
        ensemble_proba += rf_proba
        ensemble_proba += gb_proba
        ensemble_proba += svm_proba
        count += 2  # 加上gb和svm

        for model_name in dl_proba:
            ensemble_proba += dl_proba[model_name]
            count += 1

        # 计算平均概率
        ensemble_proba /= count

        # 从平均概率获取预测类别
        ensemble_pred = np.argmax(ensemble_proba, axis=1)

        # 返回所有模型的预测和概率
        predictions = {
            'rf': rf_pred,
            'gb': gb_pred,
            'svm': svm_pred,
            'ensemble': ensemble_pred
        }

        probabilities = {
            'rf': rf_proba,
            'gb': gb_proba,
            'svm': svm_proba,
            'ensemble': ensemble_proba
        }

        # 添加深度学习模型的预测
        for model_name in dl_pred:
            predictions[model_name] = dl_pred[model_name]
            probabilities[model_name] = dl_proba[model_name]

        return predictions, probabilities

    def _prepare_data(self, df, sequence_length=15, target_days=1, prediction_mode=False):
        """准备模型输入数据

        参数:
        - df: 原始数据DataFrame
        - sequence_length: 序列长度
        - target_days: 预测天数
        - prediction_mode: 是否为预测模式

        返回:
        - X_seq: 序列特征
        - X_static: 静态特征
        - y: 标签（如果不是预测模式）
        - dates: 日期
        """
        try:
            # 准备特征
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'MA5', 'MA10', 'RSI', 'MACD']

            # 确保所有特征列都存在
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0  # 如果缺少某些特征，用0填充

            # 准备序列特征
            sequences = []
            static_features = []
            dates = []

            for i in range(len(df) - sequence_length - target_days + 1):
                seq = df[feature_columns].iloc[i:i+sequence_length].values
                sequences.append(seq)

                # 使用最后一天的数据作为静态特征
                static = df[feature_columns].iloc[i+sequence_length-1].values
                static_features.append(static)

                dates.append(df.index[i+sequence_length-1])

            X_seq = np.array(sequences)
            X_static = np.array(static_features)

            if prediction_mode:
                return X_seq, X_static, None, dates

            # 准备标签
            y = self._prepare_labels(df.iloc[sequence_length:])

            return X_seq, X_static, y, dates
            
        except Exception as e:
            self.logger.error(f"准备数据时出错: {str(e)}")
            return None, None, None, None

    def _prepare_labels(self, df):
        """准备标签数据

        参数:
        - df: 原始数据DataFrame

        返回:
        - 标签数组
        """
        try:
            # 计算价格变化
            price_changes = df['Close'].pct_change()

            # 定义阈值
            threshold = 0.001  # 0.1%的变化阈值

            # 创建标签
            labels = np.zeros(len(price_changes))
            labels[price_changes > threshold] = 2  # 上涨
            labels[price_changes < -threshold] = 0  # 下跌
            labels[(price_changes >= -threshold) & (price_changes <= threshold)] = 1  # 持平

            # 移除第一个NaN值
            return labels[1:]

        except Exception as e:
            self.logger.error(f"准备标签时出错: {str(e)}")
            return None
                
    def _create_stacking_ensemble(self, pair):
        """创建堆叠集成模型

        参数:
        - pair: 货币对名称

        返回:
        - 堆叠集成模型
        """
        try:
            # 第一层基模型
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=500,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=5,
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.0001,
                    random_state=42
                )),
                ('svm', SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                ))
            ]

            # 元模型（第二层）
            meta_model = GradientBoostingClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.0001,
                random_state=42
            )

            # 创建堆叠集成
            stacking = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )

            return stacking

        except Exception as e:
            self.logger.error(f"创建堆叠集成模型时出错: {str(e)}")
            return None

    def _time_series_split(self, n):
        """时间序列分割"""
        try:
            # 创建时间序列分割器
            tscv = TimeSeriesSplit(n_splits=5)

            # 获取分割索引
            train_idx, val_idx = next(tscv.split(np.zeros(n)))

            return train_idx, val_idx

        except Exception as e:
            self.logger.error(f"时间序列分割时出错: {str(e)}")
            return None, None

    def _train_dl_models(self, pair, X_train_seq, X_train_static, y_train,
                         X_val_seq, X_val_static, y_val):
        """训练深度学习模型"""
        try:
            # 获取特征维度
            sequence_length = X_train_seq.shape[1]
            n_features = X_train_seq.shape[2]
            static_features = X_train_static.shape[1]

            # 创建模型保存目录
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True, parents=True)

            # 计算类别权重
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            class_weight = {
                int(cls): total / (len(unique) * count)
                for cls, count in zip(unique, counts)
            }

            # 训练增强模型
            models, histories, model_names = self._train_enhanced_models(
                pair=pair,
                X_train_seq=X_train_seq,
                X_train_static=X_train_static,
                y_train=y_train,
                X_val_seq=X_val_seq,
                X_val_static=X_val_static,
                y_val=y_val,
                sequence_length=sequence_length,
                n_features=n_features,
                static_features=static_features,
                models_dir=models_dir,
                class_weight=class_weight
            )

            # 保存模型
            self.dl_models[pair] = {
                name: model for name, model in zip(model_names, models)
            }

            # 合并训练历史
            history = {}
            for name, hist in zip(model_names, histories):
                for key, value in hist.items():
                    history[f"{name}_{key}"] = value

            return history
                
        except Exception as e:
            self.logger.error(f"训练深度学习模型时出错: {str(e)}")
            self.logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            return None
                
    def _generate_risk_signals(self, predictions, probabilities, threshold=0.65):
        """使用IMIO方法生成风险信号

        参数:
        - predictions: 各模型的预测结果
        - probabilities: 各模型的预测概率
        - threshold: 基础概率阈值

        返回:
        - 风险信号数组 (-1: 卖出, 0: 持有, 1: 买入)
        """
        try:
            # 初始化信号数组
            signals = np.zeros(len(predictions['ensemble']))

            # 获取各个模型的预测和概率
            ensemble_pred = predictions['ensemble']
            ensemble_prob = probabilities['ensemble']
            rf_pred = predictions.get('rf')
            rf_prob = probabilities.get('rf')
            gb_pred = predictions.get('gb')
            gb_prob = probabilities.get('gb')
            dl_pred = predictions.get('CNN-RNN')
            dl_prob = probabilities.get('CNN-RNN')

            # IMIO方法实现
            for i in range(len(ensemble_pred)):
                # 1. 计算模型一致性得分
                consensus_score = self._calculate_consensus_score([
                    rf_pred[i] if rf_pred is not None else None,
                    gb_pred[i] if gb_pred is not None else None,
                    dl_pred[i] if dl_pred is not None else None,
                    ensemble_pred[i]
                ])

                # 2. 计算概率强度得分
                probability_score = self._calculate_probability_score([
                    rf_prob[i] if rf_prob is not None else None,
                    gb_prob[i] if gb_prob is not None else None,
                    dl_prob[i] if dl_prob is not None else None,
                    ensemble_prob[i]
                ])

                # 3. 计算趋势持续性得分
                trend_score = self._calculate_trend_score(
                    ensemble_pred, i, window=5
                )

                # 4. 综合评分
                final_score = (
                    0.4 * consensus_score +
                    0.4 * probability_score +
                    0.2 * trend_score
                )

                # 5. 生成信号
                if final_score >= threshold:
                    if ensemble_pred[i] == 2:  # 上涨预测
                        signals[i] = 1  # 买入信号
                    elif ensemble_pred[i] == 0:  # 下跌预测
                        signals[i] = -1  # 卖出信号

                # 6. 风险控制
                if self._check_risk_conditions(ensemble_prob[i], final_score):
                    signals[i] = 0  # 高风险情况下保持观望

            return signals
            
        except Exception as e:
            self.logger.error(f"生成风险信号时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.zeros(len(predictions['ensemble']))

    def _calculate_consensus_score(self, predictions):
        """计算模型一致性得分"""
        try:
            valid_predictions = [p for p in predictions if p is not None]
            if not valid_predictions:
                return 0

            # 计算众数和其出现次数
            mode_pred = max(set(valid_predictions), key=valid_predictions.count)
            consensus_count = valid_predictions.count(mode_pred)

            # 计算一致性得分
            return consensus_count / len(valid_predictions)

        except Exception as e:
            self.logger.error(f"计算一致性得分时出错: {str(e)}")
            return 0

    def _calculate_probability_score(self, probabilities):
        """计算概率强度得分"""
        try:
            valid_probs = [p for p in probabilities if p is not None]
            if not valid_probs:
                return 0

            # 计算平均概率
            avg_prob = np.mean([np.max(p) for p in valid_probs])

            # 计算概率方差
            prob_var = np.var([np.max(p) for p in valid_probs])

            # 综合得分：高平均概率，低方差
            return avg_prob * (1 - prob_var)

        except Exception as e:
            self.logger.error(f"计算概率强度得分时出错: {str(e)}")
            return 0

    def _calculate_trend_score(self, predictions, current_idx, window=5):
        """计算趋势持续性得分"""
        try:
            if current_idx < window:
                return 0

            # 获取历史预测
            history = predictions[current_idx-window:current_idx]
            current_pred = predictions[current_idx]

            # 计算趋势一致性
            trend_count = sum(1 for p in history if p == current_pred)

            return trend_count / window

        except Exception as e:
            self.logger.error(f"计算趋势持续性得分时出错: {str(e)}")
            return 0

    def _check_risk_conditions(self, prob, score):
        """检查风险条件"""
        try:
            # 定义风险条件
            high_risk_conditions = [
                prob.max() < 0.4,  # 最高概率过低
                score < 0.5,  # 综合得分过低
                np.std(prob) < 0.1  # 概率分布过于平坦
            ]

            return any(high_risk_conditions)

        except Exception as e:
            self.logger.error(f"检查风险条件时出错: {str(e)}")
            return True

    def analyze_feature_importance(self, pair, models, X_seq, X_static, feature_names, plots_dir, y_true=None):
        """分析并可视化模型性能对比"""
        try:
            self.logger.info(f"开始分析 {pair} 的模型性能...")
            plots_dir = Path(plots_dir)

            # 将模型分类
            dl_models = {}
            ml_models = {}
            ensemble_model = None

            for name, model in models.items():
                if isinstance(model, tf.keras.Model):
                    dl_models[name] = model
                elif name == 'ensemble':
                    ensemble_model = model
                else:
                    ml_models[name] = model

            # 创建子图布局
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(3, 2, figure=fig)

            # 1. 深度学习模型性能对比
            if dl_models:
                ax1 = fig.add_subplot(gs[0, :])
                metrics_dl = {}
                for name, model in dl_models.items():
                    pred = model.predict([X_seq, X_static])
                    acc = accuracy_score(y_true, np.argmax(pred, axis=1))
                    metrics_dl[name] = {
                        'Accuracy': acc,
                        'Precision': precision_score(y_true, np.argmax(pred, axis=1), average='weighted'),
                        'Recall': recall_score(y_true, np.argmax(pred, axis=1), average='weighted'),
                        'F1': f1_score(y_true, np.argmax(pred, axis=1), average='weighted')
                    }

                # 绘制深度学习模型对比图
                df_metrics = pd.DataFrame(metrics_dl).T
                df_metrics.plot(kind='bar', ax=ax1)
                ax1.set_title(f'{pair} Deep Learning Models Performance')
                ax1.set_ylim(0, 1)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 2. 机器学习模型性能对比
            if ml_models:
                ax2 = fig.add_subplot(gs[1, :])
                metrics_ml = {}
                for name, model in ml_models.items():
                    pred = model.predict(X_static)
                    acc = accuracy_score(y_true, pred)
                    metrics_ml[name] = {
                        'Accuracy': acc,
                        'Precision': precision_score(y_true, pred, average='weighted'),
                        'Recall': recall_score(y_true, pred, average='weighted'),
                        'F1': f1_score(y_true, pred, average='weighted')
                    }

                # 绘制机器学习模型对比图
                df_metrics = pd.DataFrame(metrics_ml).T
                df_metrics.plot(kind='bar', ax=ax2)
                ax2.set_title(f'{pair} Machine Learning Models Performance')
                ax2.set_ylim(0, 1)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 3. 集成模型混淆矩阵
            if ensemble_model:
                ax3 = fig.add_subplot(gs[2, 0])
                pred = ensemble_model.predict(X_static)
                cm = confusion_matrix(y_true, pred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['下跌', '持平', '上涨'],
                    yticklabels=['下跌', '持平', '上涨'],
                    ax=ax3
                )
                ax3.set_title(f'{pair} Ensemble Model Confusion Matrix')

                # 4. 集成模型ROC曲线
                ax4 = fig.add_subplot(gs[2, 1])
                pred_proba = ensemble_model.predict_proba(X_static)
                for i, label in enumerate(['下跌', '持平', '上涨']):
                    fpr, tpr, _ = roc_curve(y_true == i, pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax4.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

                ax4.plot([0, 1], [0, 1], 'k--')
                ax4.set_title(f'{pair} Ensemble Model ROC Curves')
                ax4.set_xlabel('False Positive Rate')
                ax4.set_ylabel('True Positive Rate')
                ax4.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / f'{pair}_model_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"{pair} 模型性能分析完成")

        except Exception as e:
            self.logger.error(f"分析模型性能时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

    def preprocess_data(self, df):
        """数据预处理"""
        try:
            # 1. 删除不需要的列
            columns_to_keep = ['Open', 'High', 'Low', 'Close','Volume']
            df = df[columns_to_keep].copy()

            # 2. 处理缺失值
            df.fillna(method='ffill', inplace=True)

            # 3. 特征缩放
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )

            return df_scaled

        except Exception as e:
            self.logger.error(f"数据预处理时出错: {str(e)}")
            return None
                
    def _create_deep_cnn(self, sequence_shape, static_features):
        """创建深度CNN模型"""
        # 从sequence_shape中获取序列长度和特征数
        sequence_length, n_features = sequence_shape

        sequence_input = Input(shape=(sequence_length, n_features), name='sequence_input')
        static_input = Input(shape=(static_features,), name='static_input')

        # 更深的CNN结构
        conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(sequence_input)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)

        # 全局池化
        gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        # 静态特征处理
        dense1 = Dense(128, activation='relu')(static_input)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dense1 = tf.keras.layers.Dropout(0.2)(dense1)

        # 特征融合
        concat = tf.keras.layers.concatenate([gap, dense1])

        # 全连接层
        fc1 = tf.keras.layers.Dense(128, activation='relu')(concat)
        fc1 = tf.keras.layers.BatchNormalization()(fc1)
        fc1 = tf.keras.layers.Dropout(0.3)(fc1)

        # 输出层
        output = Dense(3, activation='softmax')(fc1)

        # 创建模型
        model = Model(inputs=[sequence_input, static_input], outputs=output)

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_transformer(self, sequence_length, n_features, static_features):
        """创建Transformer模型"""
        sequence_input = Input(shape=(sequence_length, n_features), name='sequence_input')
        static_input = Input(shape=(static_features,), name='static_input')

        # 将序列输入投影到64维
        x = tf.keras.layers.Dense(64)(sequence_input)  # [batch_size, seq_len, 64]

        # 创建位置编码
        pos_encoding = PositionEmbeddingLayer(
                         sequence_length=sequence_length,
                         output_dim=64
                                            )(x)  # [batch_size, seq_len, 64]

        # 添加位置编码
        x = tf.keras.layers.Add()([x, pos_encoding])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 多头注意力层和Transformer块
        for _ in range(2):  # 2个Transformer块
            # 自注意力
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=16
            )(x, x)

            # 残差连接和层归一化
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # 前馈网络
            ffn = tf.keras.layers.Dense(128, activation='relu')(x)
            ffn = tf.keras.layers.Dense(64)(ffn)

            # 残差连接和层归一化
            x = tf.keras.layers.Add()([x, ffn])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 全局特征池化
        x = tf.keras.layers.GlobalAveragePooling1D()(x)  # [batch_size, 64]

        # 静态特征处理
        static_features = tf.keras.layers.Dense(64, activation='relu')(static_input)
        static_features = tf.keras.layers.BatchNormalization()(static_features)

        # 特征融合
        x = tf.keras.layers.concatenate([x, static_features])
        x = tf.keras.layers.Dropout(0.4)(x)

        # 分类头
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        output = tf.keras.layers.Dense(3, activation='softmax')(x)

        # 创建模型
        model = Model(inputs=[sequence_input, static_input], outputs=output)

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _prepare_prediction_data(self, df):
        """准备预测数据"""
        try:
            data = df.copy()

            # 1. 特征工程
            # 计算技术指标
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA10'] = data['Close'].rolling(window=10).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'] = self._calculate_macd(data['Close'])

            # 计算趋势特征
            data['trend_5'] = (data['Close'] - data['MA5']) / data['MA5']
            data['trend_10'] = (data['Close'] - data['MA10']) / data['MA10']
            data['trend_20'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).mean()

            # 波动性特征
            data['volatility_10'] = data['Close'].rolling(10).std() / data['Close'].rolling(10).mean()

            # 2. 准备序列特征 (9个特征)
            sequence_features = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA10', 'RSI', 'MACD', 'trend_5']
            X_seq = []
            for i in range(len(data) - 15 + 1):
                X_seq.append(data[sequence_features].iloc[i:i+15].values)
            X_seq = np.array(X_seq)

            # 3. 准备静态特征 (9个特征)
            static_features = [
                'RSI', 'MACD',
                'trend_5', 'trend_10', 'trend_20',
                'volatility_10',
                'MA5', 'MA10',
                'Close'
            ]
            X_static = data[static_features].iloc[14:].values

            return X_seq, X_static

        except Exception as e:
            self.logger.error(f"准备预测数据时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None

    def plot_training_history(self, history, save_path='plots/training_history.png'):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        for key in history.keys():
            if 'loss' in key and 'val' not in key:
                plt.plot(history[key], label=f'Training {key}')
                if f'val_{key}' in history:
                    plt.plot(history[f'val_{key}'], label=f'Validation {key}')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(2, 2, 2)
        for key in history.keys():
            if 'accuracy' in key and 'val' not in key:
                plt.plot(history[key], label=f'Training {key}')
                if f'val_{key}' in history:
                    plt.plot(history[f'val_{key}'], label=f'Validation {key}')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存图像
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def select_important_features(self, X, y):
        """特征选择"""
        try:
            # 1. 移除低方差特征
            selector = VarianceThreshold(threshold=0.01)
            X_reduced = selector.fit_transform(X)

            # 2. 使用L1正则化进行特征选择
            lasso = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=0.7,
                random_state=42
            )
            selector = SelectFromModel(lasso, prefit=False)
            X_selected = selector.fit_transform(X_reduced, y)

            # 记录选择的特征
            selected_features = np.where(selector.get_support())[0]

            return X_selected, selected_features
                
        except Exception as e:
            self.logger.error(f"特征选择时出错: {str(e)}")
            return None, None
            
    def _calculate_rsi(self, prices, period=14):
        """计算相对强弱指标(RSI)"""
        try:
            # 计算价格变化
            delta = prices.diff()

            # 分离上涨和下跌
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # 计算相对强弱指标
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"计算RSI时出错: {str(e)}")
            return pd.Series(np.nan, index=prices.index)

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算移动平均收敛散度(MACD)"""
        try:
            # 计算快速和慢速EMA
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()

            # 计算MACD线
            macd = exp1 - exp2

            # 计算信号线
            signal_line = macd.ewm(span=signal, adjust=False).mean()

            # 计算MACD柱状图
            histogram = macd - signal_line

            return histogram  # 返回MACD柱状图作为特征

        except Exception as e:
            self.logger.error(f"计算MACD时出错: {str(e)}")
            return pd.Series(np.nan, index=prices.index)

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带指标"""
        try:
            # 计算移动平均线
            middle_band = prices.rolling(window=window).mean()

            # 计算标准差
            std = prices.rolling(window=window).std()

            # 计算上下轨
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)

            return middle_band, upper_band, lower_band

        except Exception as e:
            self.logger.error(f"计算布林带时出错: {str(e)}")
            return None, None, None

    def _prepare_training_data(self, df, pair):
        """准备训练数据"""
        try:
            data = df.copy()

            # 1. 基础价格特征
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log(data['Close']/data['Close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=20).std()

            # 2. 技术指标
            # RSI - 多周期
            for period in [6, 14, 21]:
                data[f'RSI_{period}'] = self._calculate_rsi(data['Close'], period)

            # 移动平均和趋势
            for window in [5, 10, 20, 50]:
                data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
                # 趋势强度
                data[f'trend_strength_{window}'] = (data['Close'] - data[f'MA_{window}']) / data[f'MA_{window}']

            # MACD
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # 布林带
            for window in [20, 50]:
                ma = data['Close'].rolling(window=window).mean()
                std = data['Close'].rolling(window=window).std()
                data[f'BB_upper_{window}'] = ma + (std * 2)
                data[f'BB_lower_{window}'] = ma - (std * 2)
                data[f'BB_width_{window}'] = (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}']) / ma
                data[f'BB_position_{window}'] = (data['Close'] - data[f'BB_lower_{window}']) / (data[f'BB_upper_{window}'] - data[f'BB_lower_{window}'])

            # 3. 波动性指标
            # ATR
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['ATR'] = true_range.rolling(14).mean()
            data['ATR_ratio'] = data['ATR'] / data['Close']

            # 4. 动量指标
            # ROC - Rate of Change
            for period in [5, 10, 20]:
                data[f'ROC_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period) * 100

            # 5. 成交量相关特征（如果有）
            if 'Volume' in data.columns:
                data['volume_ma10'] = data['Volume'].rolling(window=10).mean()
                data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
                data['volume_ratio'] = data['Volume'] / data['volume_ma20']

            # 删除包含NaN的行
            data = data.dropna()

            # 准备序列特征
            sequence_features = [
                'returns', 'log_returns', 'volatility',
                'RSI_14', 'trend_strength_20', 'MACD_hist',
                'BB_position_20', 'ATR_ratio', 'ROC_10'
            ]

            # 准备静态特征
            static_features = [
                'RSI_6', 'RSI_21',
                'trend_strength_5', 'trend_strength_50',
                'BB_width_20', 'BB_width_50',
                'ATR_ratio', 'ROC_5', 'ROC_20'
            ]

            # 创建序列数据
            X_seq = []
            for i in range(len(data) - 15):
                X_seq.append(data[sequence_features].iloc[i:i+15].values)
            X_seq = np.array(X_seq)

            # 创建静态特征
            X_static = data[static_features].iloc[14:-1].values

            # 准备标签
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]
            thresholds = [-volatility, volatility]  # 动态阈值

            y = pd.cut(returns,
                      bins=[-np.inf] + thresholds + [np.inf],
                      labels=[0, 1, 2])
            y = y.iloc[14:-1].values

            # 标准化特征
            if pair not in self.scalers:
                self.scalers[pair] = {
                    'scaler_seq': StandardScaler(),
                    'scaler_static': StandardScaler()
                }
                self.scalers[pair]['scaler_seq'].fit(X_seq.reshape(-1, X_seq.shape[-1]))
                self.scalers[pair]['scaler_static'].fit(X_static)

            X_seq = self.scalers[pair]['scaler_seq'].transform(
                X_seq.reshape(-1, X_seq.shape[-1])
            ).reshape(X_seq.shape)
            X_static = self.scalers[pair]['scaler_static'].transform(X_static)

            # 验证数据维度
            self.logger.info(f"X_seq shape: {X_seq.shape}")
            self.logger.info(f"X_static shape: {X_static.shape}")
            self.logger.info(f"y shape: {y.shape}")

            return X_seq, X_static, y

        except Exception as e:
            self.logger.error(f"准备训练数据时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None, None

    def _create_cnn_rnn_model(self, sequence_shape, static_features):
        """创建改进的CNN-RNN混合模型"""
        sequence_input = Input(shape=sequence_shape, name='sequence_input')
        static_input = Input(shape=(static_features,), name='static_input')

        # 1. 增加正则化
        regularizer = tf.keras.regularizers.l2(1e-6)

        # 2. CNN部分 - 使用残差连接
        x = sequence_input
        for filters in [32, 64, 128]:
            # 主路径
            conv = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding='same',
                kernel_regularizer=regularizer
            )(x)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation('relu')(conv)
            conv = tf.keras.layers.Dropout(0.5)(conv)

            # 残差连接
            if x.shape[-1] != filters:
                x = tf.keras.layers.Conv1D(filters, 1)(x)
            x = tf.keras.layers.Add()([x, conv])

        # 3. RNN部分 - 双向LSTM with Attention
        lstm1 = tf.keras.layers.Bidirectional(
            LSTM(64, return_sequences=True,
                 kernel_regularizer=regularizer,
                 recurrent_regularizer=regularizer)
        )(x)
        lstm1 = tf.keras.layers.LayerNormalization()(lstm1)

        # 注意力机制
        attention = tf.keras.layers.Dense(1)(lstm1)
        attention = tf.keras.layers.Reshape((-1,))(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        lstm1 = tf.keras.layers.Multiply()([lstm1, attention])
        lstm1 = tf.keras.layers.GlobalAveragePooling1D()(lstm1)

        # 4. 静态特征处理
        static_dense = tf.keras.layers.Dense(
            64,
            activation='selu',
            kernel_regularizer=regularizer
        )(static_input)
        static_dense = tf.keras.layers.BatchNormalization()(static_dense)
        static_dense = tf.keras.layers.Dropout(0.2)(static_dense)

        # 5. 特征融合
        merged = tf.keras.layers.concatenate([lstm1, static_dense])

        # 6. 分类头部 - 使用更深的结构
        x = tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizer)(merged)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # 7. 输出层
        output = tf.keras.layers.Dense(3, activation='softmax')(x)

        model = Model(inputs=[sequence_input, static_input], outputs=output)

        # 8. 使用自定义学习率调度器
        initial_learning_rate = 0.005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_deep_cnn_model(self, sequence_shape, static_features):
        """创建深度CNN模型"""
        # 序列输入
        sequence_input = Input(shape=sequence_shape, name='sequence_input')

        # CNN部分
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(sequence_input)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(filters=256, kernel_size=4, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Global Pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # 静态特征输入
        static_input = Input(shape=(static_features,), name='static_input')
        static_dense = Dense(64, activation='relu')(static_input)
        static_bn = BatchNormalization()(static_dense)

        # 合并特征
        concat = concatenate([x, static_bn])

        # 全连接层
        x = Dense(128, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.44)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.35)(x)

        # 输出层
        output = Dense(3, activation='softmax')(x)

        # 创建模型
        model = Model(inputs=[sequence_input, static_input], outputs=output)

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_transformer_model(self, sequence_shape, static_features):
        """创建Transformer模型"""
        sequence_input = Input(shape=sequence_shape, name='sequence_input')
        static_input = Input(shape=(static_features,), name='static_input')

        # 将序列输入投影到64维
        x = tf.keras.layers.Dense(64)(sequence_input)  # [batch_size, seq_len, 64]

        # 创建位置编码
        pos_encoding = PositionEmbeddingLayer(
            sequence_length=sequence_shape[0],
            output_dim=64
        )(x)  # [batch_size, seq_len, 64]

        # 添加位置编码
        x = tf.keras.layers.Add()([x, pos_encoding])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 多头注意力层和Transformer块
        for _ in range(2):  # 2个Transformer块
            # 自注意力
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=16
            )(x, x)

            # 残差连接和层归一化
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # 前馈网络
            ffn = tf.keras.layers.Dense(128, activation='relu')(x)
            ffn = tf.keras.layers.Dense(64)(ffn)

            # 残差连接和层归一化
            x = tf.keras.layers.Add()([x, ffn])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 全局特征池化
        x = tf.keras.layers.GlobalAveragePooling1D()(x)  # [batch_size, 64]

        # 静态特征处理
        static_features = tf.keras.layers.Dense(64, activation='relu')(static_input)
        static_features = tf.keras.layers.BatchNormalization()(static_features)

        # 特征融合
        x = tf.keras.layers.concatenate([x, static_features])
        x = tf.keras.layers.Dropout(0.5)(x)

        # 分类头
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        output = tf.keras.layers.Dense(3, activation='softmax')(x)

        # 创建模型
        model = Model(inputs=[sequence_input, static_input], outputs=output)

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def optimize_hyperparameters(self, X_seq_train, X_static_train, y_train,
                               X_seq_val, X_static_val, y_val):
        """使用粒子群优化算法优化超参数"""
        try:
            # 定义参数边界 [lstm_units, dropout_rate, learning_rate, l2_reg]
            bounds = np.array([
                [32, 256],      # lstm_units
                [0.1, 0.5],     # dropout_rate
                [1e-4, 1e-2],   # learning_rate
                [1e-6, 1e-3]    # l2_reg
            ])

            # 初始化PSO优化器
            pso = ParticleSwarmOptimizer(
                n_particles=3,
                bounds=bounds,
                w=0.7,
                c1=1.5,
                c2=1.5
            )

            # 定义适应度函数
            def fitness_function(params):
                try:
                    # 创建模型
                    model = self._create_model(
                        sequence_length=X_seq_train.shape[1],
                        n_features=X_seq_train.shape[2],
                        static_features=X_static_train.shape[1],
                        lstm_units=int(params[0]),
                        dropout_rate=params[1],
                        learning_rate=params[2],
                        l2_reg=params[3]
                    )

                    # 训练模型
                    history = model.fit(
                        [X_seq_train, X_static_train],
                        y_train,
                        validation_data=([X_seq_val, X_static_val], y_val),
                        epochs=26,
                        batch_size=32,
                        verbose=0
                    )

                    # 返回验证集准确率作为适应度
                    return history.history['val_accuracy'][-1]

                except Exception as e:
                    self.logger.error(f"适应度函数计算出错: {str(e)}")
                    return 0.0

            # 运行优化
            best_params, best_score = pso.optimize(fitness_function, max_iter=1)

            if best_params is not None:
                self.logger.info(f"最佳参数: {best_params}, 最佳得分: {best_score}")
                return best_params
            else:
                self.logger.warning("优化失败，使用默认参数")
                return [128, 0.3, 0.001, 1e-4]

        except Exception as e:
            self.logger.error(f"超参数优化过程中出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [128, 0.3, 0.001, 1e-4]


class ParticleSwarmOptimizer:
    """粒子群优化器"""

    def __init__(self, n_particles, bounds, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.bounds = bounds
        self.w = w  # 添加惯性权重
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子位置和速度
        self.positions = np.random.uniform(
            low=bounds[:, 0],
            high=bounds[:, 1],
            size=(n_particles, len(bounds))
        )
        self.velocities = np.zeros((n_particles, len(bounds)))

        # 初始化个体最优和全局最优
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, -np.inf)
        self.gbest_position = None
        self.gbest_score = -np.inf

    def optimize(self, fitness_function, max_iter=5):
        """优化函数"""
        try:
            for _ in range(max_iter):
                # 评估适应度
                scores = np.array([fitness_function(p) for p in self.positions])

                # 更新个体最优
                improved = scores > self.pbest_scores
                self.pbest_scores[improved] = scores[improved]
                self.pbest_positions[improved] = self.positions[improved]

                # 更新全局最优
                best_idx = np.argmax(scores)
                if scores[best_idx] > self.gbest_score:
                    self.gbest_score = scores[best_idx]
                    self.gbest_position = self.positions[best_idx].copy()

                # 更新速度和位置
                r1, r2 = np.random.rand(2)
                self.velocities = (self.w * self.velocities +
                                   self.c1 * r1 * (self.pbest_positions - self.positions) +
                                   self.c2 * r2 * (self.gbest_position - self.positions))

                self.positions += self.velocities

                # 边界处理
                self.positions = np.clip(
                    self.positions,
                    self.bounds[:, 0],
                    self.bounds[:, 1]
                )

            return self.gbest_position, self.gbest_score

        except Exception as e:
            logging.error(f"PSO优化过程中出错: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None


def init_logging():
    """初始化日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_analysis.log')
        ]
    )
    return logging.getLogger(__name__)

def load_currency_data(currency_pair):
    """加载货币对数据"""
    try:
        file_path = f"../try/data/{currency_pair}.csv"
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"加载{currency_pair}数据时出错: {str(e)}")
        return None

def align_data(data_dict):
    """对齐不同货币对的数据"""
    # 找到共同的日期范围
    common_dates = None
    for df in data_dict.values():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    common_dates = sorted(list(common_dates))

    # 对齐所有数据
    aligned_data = {}
    for pair, df in data_dict.items():
        aligned_data[pair] = df.loc[common_dates].copy()

    return aligned_data

# 修复Transformer模型的位置编码

@register_keras_serializable()
class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.output_dim
        )
        super().build(input_shape)

    def call(self, inputs):
        positions = tf.range(start=0.0, limit=float(self.sequence_length), delta=1.0)
        positions = tf.expand_dims(positions, axis=0)
        position_embeddings = self.embedding(tf.cast(positions, tf.int32))
        batch_size = tf.shape(inputs)[0]
        position_embeddings = tf.repeat(position_embeddings, repeats=batch_size, axis=0)
        return position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim
        })
        return config

class CastLayer(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.int32, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.dtype)

def create_kelm_model(sequence_length, n_features, static_features):
    # 创建输入层
    sequence_input = tf.keras.layers.Input(
        shape=(sequence_length, n_features),
        name='sequence_input'
    )
    static_input = tf.keras.layers.Input(
        shape=(static_features,),
        name='static_input'
    )

    # 序列特征处理
    flatten = tf.keras.layers.Flatten()(sequence_input)

    # 静态特征处理
    static_dense = tf.keras.layers.Dense(64, activation='relu')(static_input)

    # 合并特征
    concat = tf.keras.layers.concatenate([flatten, static_dense])

    # KELM 层
    n_hidden = 100
    gamma = 2

    # 创建随机权重层
    dense = tf.keras.layers.Dense(
        n_hidden,
        use_bias=True,
        trainable=False,
        kernel_initializer='random_normal',
        name='kelm_weights'
    )(concat)

    # RBF kernel
    rbf = tf.keras.layers.Lambda(
        lambda x: tf.exp(-gamma * tf.square(x)),
        name='rbf_kernel'
    )(dense)

    # 输出层
    output = tf.keras.layers.Dense(
        3,
        activation='softmax',
        name='output'
    )(rbf)

    # 创建模型
    model = tf.keras.Model(
        inputs=[sequence_input, static_input],
        outputs=output,
        name='kelm_model'
    )

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def apply_iceemdan(data, num_imfs=3):
    """
    对时间序列数据应用 ICEEMDAN 分解
    """
    try:
        # 确保数据是一维数组
        data = np.array(data).flatten()

        # 初始化CEEMDAN
        ceemdan = CEEMDAN()

        # 执行分解
        imfs = ceemdan.ceemdan(data)

        # 确保所有IMF长度与原始数据相同
        imfs_reshaped = []
        for imf in imfs[:num_imfs]:
            if len(imf) != len(data):
                # 如果长度不匹配，进行重采样
                imf = np.interp(
                    np.linspace(0, 1, len(data)),
                    np.linspace(0, 1, len(imf)),
                    imf
                )
            imfs_reshaped.append(imf)

        return np.array(imfs_reshaped)
            
    except Exception as e:
        print(f"ICEEMDAN decomposition failed: {str(e)}")
        # 如果分解失败，返回原始数据作为单个IMF
        return np.array([data])

def calculate_technical_indicators(df):
    """计算技术指标"""
    # 复制数据框以避免修改原始数据
    df = df.copy()

    # 计算移动平均线
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # 计算RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 计算MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def preprocess_data(df):
    """预处理数据，包括计算技术指标和ICEEMDAN分解"""
    # 首先计算技术指标
    df = calculate_technical_indicators(df)

    # 对技术指标进行ICEEMDAN分解
    # 移除价格相关的特征，只对技术指标进行分解
    features = ['MA5', 'MA10', 'RSI', 'MACD']
    decomposed_features = {}

    for feature in features:
        if feature in df.columns:  # 确保特征存在
            imfs = apply_iceemdan(df[feature].values)
            if imfs is not None:
                # 确保每个IMF的长度与原始数据相同
                for i, imf in enumerate(imfs):
                    if len(imf) == len(df):
                        decomposed_features[f'{feature}_imf{i}'] = imf
                    else:
                        print(f"Warning: Skipping {feature}_imf{i} due to length mismatch")

    # 将分解后的特征添加到原始数据中
    for name, values in decomposed_features.items():
        try:
            if len(values) == len(df):
                df[name] = values
            else:
                print(f"Warning: Length mismatch for {name}, expected {len(df)}, got {len(values)}")
        except Exception as e:
            print(f"Error adding feature {name}: {str(e)}")
            continue

    # 使用更新的方法填充NaN值
    df = df.ffill().bfill()

    return df

def main():
    """主函数"""
    logger = init_logging()
    logger.info("开始模型分析...")

    try:
        # 创建必要的文件夹
        plots_dir = Path("plots")
        models_dir = Path("models")
        signals_dir = Path("signals")

        for directory in [plots_dir, models_dir, signals_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # 加载数据
        currency_pairs = ["CNYAUD", "CNYEUR"]
        data_dict = {}

        for pair in currency_pairs:
            df = load_currency_data(pair)
            if df is not None:
                data_dict[pair] = df
                logger.info(f"成功加载 {pair} 数据，形状: {df.shape}")

        if not data_dict:
            logger.error("没有成功加载任何数据")
            return

        # 对齐数据
        data_dict = align_data(data_dict)
        logger.info(f"使用共同日期范围: {data_dict['CNYAUD'].index[0]} 到 {data_dict['CNYAUD'].index[-1]}, 共 {len(data_dict['CNYAUD'])} 天")

        # 初始化混合模型
        hybrid_model = HybridForexModel()

        # 训练模型
        logger.info("开始训练混合模型...")
        history_dict = hybrid_model.train(data_dict)

        if history_dict:
            logger.info("模型训练完成")

            # 绘制训练历史
            hybrid_model.plot_training_history(history_dict, save_path=plots_dir)

            # 输出训练结果
            for pair, history in history_dict.items():
                logger.info(f"\n{pair} 最终训练结果:")
                for key, value in history.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"{key}: {value:.4f}")
                    elif isinstance(value, list) and value:
                        logger.info(f"{key}: {value[-1]:.4f}")

            # 模型预测
            predictions, signals, feature_importance = hybrid_model.predict(data_dict)

            if predictions:
                # 分析特征重要性
                for pair in predictions.keys():
                    if pair in data_dict:
                        try:
                            # 准备特征名称
                            sequence_features = [
                                'Open', 'High', 'Low',
                                'MA5', 'MA10', 'RSI', 'MACD'
                            ]

                            technical_features = [
                                'trend_5', 'trend_10', 'trend_20',
                                'volatility_10', 'volatility_20',
                                'bb_width', 'bb_pos',
                                'macd_norm', 'rsi_14'
                            ]

                            derived_features = [
                                'price_change_1d', 'price_change_7d',
                                'mean_5d', 'std_5d',
                                'seasonal', 'trend', 'residual'
                            ]

                            feature_names = (sequence_features + technical_features + derived_features)

                            # 获取训练数据
                            X_seq, X_static, y, _ = hybrid_model._prepare_data(data_dict[pair])

                            # 收集可用的模型
                            available_models = {}

                            # 添加机器学习模型（如果存在）
                            if pair in hybrid_model.ml_models:
                                for model_name in ['rf', 'gb', 'ensemble']:
                                    if model_name in hybrid_model.ml_models[pair]:
                                        available_models[model_name] = hybrid_model.ml_models[pair][model_name]

                            # 添加深度学习模型（如果存在）
                            if pair in hybrid_model.dl_models:
                                for model_name, model_info in hybrid_model.dl_models[pair].items():
                                    if isinstance(model_info, dict) and 'model' in model_info:
                                        available_models[model_name] = model_info['model']
                                    elif hasattr(model_info, 'predict'):
                                        available_models[model_name] = model_info

                            if available_models:
                                # 分析特征重要性
                                hybrid_model.analyze_feature_importance(
                                    pair=pair,
                                    models=available_models,
                                    X_seq=X_seq,
                                    X_static=X_static,
                                    feature_names=feature_names,
                                    plots_dir=plots_dir
                                )

                            # 绘制混淆矩阵和评估指标
                            true_labels = hybrid_model._prepare_labels(data_dict[pair])
                            pred_labels = predictions[pair]['ensemble']

                            # 调整数据长度
                            min_len = min(len(true_labels), len(pred_labels))
                            true_labels = true_labels[-min_len:]
                            pred_labels = pred_labels[-min_len:]

                            # 绘制混淆矩阵
                            plt.figure(figsize=(10, 8))
                            cm = confusion_matrix(true_labels, pred_labels)
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt='d',
                                cmap='Blues',
                                xticklabels=['下跌', '持平', '上涨'],
                                yticklabels=['下跌', '持平', '上涨']
                            )
                            plt.title(f'{pair} 预测混淆矩阵')
                            plt.ylabel('真实标签')
                            plt.xlabel('预测标签')
                            plt.tight_layout()
                            plt.savefig(plots_dir / f'{pair}_confusion_matrix.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # 计算并记录评估指标
                            metrics = {
                                '准确率': accuracy_score(true_labels, pred_labels),
                                '精确率': precision_score(true_labels, pred_labels, average='weighted'),
                                '召回率': recall_score(true_labels, pred_labels, average='weighted'),
                                'F1分数': f1_score(true_labels, pred_labels, average='weighted')
                            }

                            # 绘制评估指标条形图
                            plt.figure(figsize=(10, 6))
                            colors = sns.color_palette('husl', len(metrics))
                            bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
                            plt.title(f'{pair} 模型评估指标')
                            plt.ylim(0, 1)

                            # 在柱状图上添加具体数值
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(
                                    bar.get_x() + bar.get_width()/2.,
                                    height,
                                    f'{height:.4f}',
                                    ha='center',
                                    va='bottom'
                                )

                            plt.tight_layout()
                            plt.savefig(plots_dir / f'{pair}_metrics.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # 记录评估结果
                            logger.info(f"\n{pair} 预测评估指标:")
                            for metric, value in metrics.items():
                                logger.info(f"{metric}: {value:.4f}")

                        except Exception as e:
                            logger.error(f"处理 {pair} 的可视化和评估时出错: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue

                logger.info("模型评估完成!")

            logger.info("分析完成!")

    except Exception as e:
        logger.error(f"运行过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()