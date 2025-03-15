import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.layers import Input, Conv1D, BatchNormalization, LSTM, Dense, MaxPooling1D, concatenate, Dropout, \
    GlobalAveragePooling1D, LayerNormalization, Bidirectional, MultiHeadAttention, Add
from keras.models import Model
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.utils import to_categorical
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class HybridForexModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_models = {
            'ml_models': {},  # 机器学习模型
            'dl_models': {},  # 深度学习模型
        }
        self.ensemble_model = None  # 最终集成模型
        self.scalers = {}  # 特征标准化器
        self.pso_optimizer = None  # PSO优化器
        self.best_params = {}  # 最佳参数
        self.current_dir = Path(__file__).parent
        self.data_dir = self.current_dir / "FE"
        self.n_features = None  # 特征数量
        self.n_static_features = 10  # 静态特征数量（可以根据实际情况调整）

    def _create_base_models(self):
        """创建基础模型"""
        try:
            # 1. 机器学习模型
            ml_models = {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                ),
                'svm': self._create_svm_model(),
                'kelm': self._create_kelm_model()
            }

            # 2. 深度学习模型
            dl_models = {
                'cnn_rnn': self._create_cnn_rnn_model(),
                'deep_cnn': self._create_deep_cnn_model(),
                'transformer': self._create_transformer_model()
            }

            self.base_models['ml_models'] = ml_models
            self.base_models['dl_models'] = dl_models

            return True

        except Exception as e:
            self.logger.error(f"创建基础模型失败: {str(e)}")
            return False

    def _create_kelm_model(self, hidden_units=1000, gamma=0.1, C=1.0):
        """创建KELM模型"""
        try:
            class KELMClassifier:
                def __init__(self, hidden_units=1000, gamma=0.1, C=1.0):
                    self.hidden_units = hidden_units
                    self.gamma = gamma
                    self.C = C
                    self.random_state = 42
                    self.weights = None
                    self.bias = None
                    self.beta = None
                    self.X_train = None

                def set_params(self, **params):
                    """设置模型参数"""
                    for param, value in params.items():
                        if hasattr(self, param):
                            setattr(self, param, value)
                        else:
                            raise ValueError(f"无效的参数: {param}")
                    return self

                def get_params(self, deep=True):
                    """获取模型参数"""
                    return {
                        'hidden_units': self.hidden_units,
                        'gamma': self.gamma,
                        'C': self.C
                    }

                def fit(self, X, y):
                    """训练模型"""
                    if X is None or y is None:
                        raise ValueError("输入数据不能为空")

                    # 保存训练数据用于预测
                    self.X_train = np.array(X, dtype=np.float64)
                    n_samples, n_features = X.shape

                    # 随机生成输入权重和偏置
                    rng = np.random.RandomState(self.random_state)
                    self.weights = rng.normal(size=(n_features, self.hidden_units))
                    self.bias = rng.normal(size=self.hidden_units)

                    # 计算隐藏层输出
                    H = self._kernel_matrix(self.X_train, self.X_train)

                    # 添加正则化项
                    I = np.eye(H.shape[0])
                    # 计算输出权重
                    self.beta = np.linalg.solve(H.T @ H + self.C * I, H.T @ self._one_hot_encode(y))

                    return self

                def predict(self, X):
                    """预测类别"""
                    if X is None:
                        raise ValueError("输入数据不能为空")
                    if self.X_train is None:
                        raise ValueError("模型尚未训练")

                    X = np.array(X, dtype=np.float64)
                    # 计算核矩阵
                    H = self._kernel_matrix(X, self.X_train)
                    # 预测
                    y_pred = H @ self.beta
                    # 返回类别标签
                    return np.argmax(y_pred, axis=1) - 1

                def _kernel_matrix(self, X1, X2):
                    """计算RBF核矩阵"""
                    if X1 is None or X2 is None:
                        raise ValueError("输入数据不能为空")

                    X1 = np.array(X1, dtype=np.float64)
                    X2 = np.array(X2, dtype=np.float64)

                    # 计算欧氏距离的平方
                    distances = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                                np.sum(X2 ** 2, axis=1) - \
                                2 * np.dot(X1, X2.T)
                    # 返回核矩阵
                    return np.exp(-self.gamma * np.maximum(distances, 0))

                def _one_hot_encode(self, y):
                    """将标签转换为one-hot编码"""
                    if y is None:
                        raise ValueError("输入标签不能为空")

                    y = np.array(y)
                    n_classes = len(np.unique(y))
                    n_samples = len(y)
                    one_hot = np.zeros((n_samples, n_classes))
                    one_hot[np.arange(n_samples), y + 1] = 1
                    return one_hot

            return KELMClassifier(hidden_units=hidden_units, gamma=gamma, C=C)

        except Exception as e:
            self.logger.error(f"创建KELM模型失败: {str(e)}")
            return None

    def _create_cnn_rnn_model(self, n_filters=64, n_lstm_units=64, learning_rate=0.001, dropout_rate=0.3):
        """创建CNN-RNN混合模型"""
        try:
            # 序列输入
            seq_input = Input(shape=(None, 1))

            # CNN部分
            conv1 = Conv1D(n_filters, 3, padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(seq_input)
            conv1 = BatchNormalization()(conv1)
            conv2 = Conv1D(n_filters // 2, 5, padding='same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(seq_input)
            conv2 = BatchNormalization()(conv2)

            # 合并CNN特征
            conv_merged = concatenate([conv1, conv2])
            conv_merged = MaxPooling1D(2)(conv_merged)

            # RNN部分
            lstm1 = Bidirectional(LSTM(n_lstm_units, return_sequences=True,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01)))(conv_merged)
            lstm1 = BatchNormalization()(lstm1)
            lstm2 = Bidirectional(LSTM(n_lstm_units // 2,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01)))(lstm1)
            lstm2 = BatchNormalization()(lstm2)

            # 全连接层
            dense1 = Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(lstm2)
            dense1 = Dropout(dropout_rate)(dense1)
            output = Dense(3, activation='softmax')(dense1)

            model = Model(inputs=seq_input, outputs=output)
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            self.logger.error(f"创建CNN-RNN模型失败: {str(e)}")
            return None

    def _create_deep_cnn_model(self, n_filters=64, learning_rate=0.001, dropout_rate=0.3):
        """创建深度CNN模型"""
        try:
            # 序列输入
            seq_input = Input(shape=(None, 1))

            # CNN层
            x = Conv1D(n_filters, 3, padding='same', activation='relu')(seq_input)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)

            x = Conv1D(n_filters * 2, 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)

            # 全局池化
            x = GlobalAveragePooling1D()(x)

            # 全连接层
            x = Dense(64, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            output = Dense(3, activation='softmax')(x)

            # 创建模型
            model = Model(inputs=seq_input, outputs=output)

            # 编译模型
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            self.logger.error(f"创建深度CNN模型失败: {str(e)}")
            return None

    def _create_svm_model(self, C=1.0, gamma='scale'):
        """创建SVM模型，增强正则化"""
        try:
            # 进一步增强正则化
            return SVC(
                C=C,  # C值会通过PSO在更小范围内优化
                gamma=gamma,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=2000,  # 增加缓存以提高性能
                shrinking=True,  # 启用收缩启发式
                tol=1e-4  # 提高收敛精度
            )
        except Exception as e:
            self.logger.error(f"创建SVM模型失败: {str(e)}")
            return None

    def _create_transformer_model(self, n_heads=4, ff_dim=64, learning_rate=0.001, dropout_rate=0.3):
        """创建改进的Transformer模型"""
        try:
            # 序列输入
            seq_input = Input(shape=(None, 1))

            # 位置编码
            x = seq_input

            # 多层Transformer块
            for _ in range(3):  # 增加到3层
                # 自注意力层
                x = LayerNormalization(epsilon=1e-6)(x)
                attn_output = MultiHeadAttention(
                    num_heads=n_heads,
                    key_dim=ff_dim // n_heads,
                    dropout=dropout_rate
                )(x, x)
                x = Dropout(dropout_rate)(attn_output)
                x = Add()([x, attn_output])

                # 前馈网络
                x = LayerNormalization(epsilon=1e-6)(x)
                ffn = Dense(ff_dim * 4, activation='gelu')(x)  # 使用GELU激活
                ffn = Dropout(dropout_rate)(ffn)
                ffn = Dense(ff_dim)(ffn)
                x = Add()([x, ffn])

            # 全局池化
            x = GlobalAveragePooling1D()(x)

            # 分类头
            x = Dense(ff_dim * 2, activation='gelu')(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(ff_dim, activation='gelu')(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            output = Dense(3, activation='softmax')(x)

            model = Model(inputs=seq_input, outputs=output)

            # 使用带warmup的自定义学习率调度
            initial_learning_rate = learning_rate

            class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, initial_lr, warmup_steps=1000):
                    super().__init__()
                    self.initial_lr = initial_lr
                    self.warmup_steps = warmup_steps

                def __call__(self, step):
                    arg1 = tf.math.rsqrt(step)
                    arg2 = step * (self.warmup_steps ** -1.5)
                    return self.initial_lr * tf.math.minimum(arg1, arg2)

            optimizer = Adam(learning_rate=initial_learning_rate)  # 直接使用固定学习率
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            self.logger.error(f"创建Transformer模型失败: {str(e)}")
            return None

    def _create_final_ensemble(self, meta_features_train, y_train, meta_features_val=None, y_val=None):
        """创建最终的Stacking集成模型"""
        try:
            # 定义所有元模型
            meta_learners = {
                'xgb': xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=3,
                    max_depth=6,
                    learning_rate=0.01,
                    n_estimators=500,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    eval_metric='mlogloss',
                    use_label_encoder=False,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    multi_class='multinomial',  # 多项式逻辑回归
                    solver='lbfgs',  # 适用于多分类
                    C=1.0,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=10,  # 早停参数
                    random_state=42
                )
            }

            # 训练所有元模型并记录准确率
            model_accuracies = {}
            trained_models = {}

            for name, model in meta_learners.items():
                if name == 'xgb':
                    model.fit(
                        meta_features_train,
                        y_train,
                        eval_set=[(meta_features_train, y_train)],
                        verbose=False
                    )
                else:
                    model.fit(meta_features_train, y_train)

                # 计算准确率
                train_pred = model.predict(meta_features_train)
                accuracy = np.mean(train_pred == y_train)
                model_accuracies[name] = accuracy
                trained_models[name] = model

            # 选择准确率最高的模型
            best_model_name = max(model_accuracies, key=model_accuracies.get)
            self.ensemble_model = trained_models[best_model_name]

            return True

        except Exception as e:
            self.logger.error(f"创建集成模型失败: {str(e)}")
            return False

    def train_multi_step(self, X_seq, X_static, y, time_steps=[1, 3, 5, 10]):
        """多步预测训练"""
        try:
            self.logger.info("开始多步预测训练...")

            # 为每个时间步长创建标签
            y_multi = {}
            for step in time_steps:
                y_multi[step] = self._create_future_labels(y, step)

            # 训练基础模型
            meta_features = []

            # 1. 训练机器学习模型
            for name, model in self.base_models['ml_models'].items():
                self.logger.info(f"训练 {name} 模型...")
                model_preds = []

                for step in time_steps:
                    # PSO优化超参数
                    best_params = self.pso_optimizer.optimize(
                        model, X_seq, y_multi[step]
                    )

                    # 使用最佳参数更新模型
                    model.set_params(**best_params)

                    # 训练模型
                    model.fit(X_seq, y_multi[step])

                    # 生成预测
                    preds = model.predict_proba(X_seq)
                    model_preds.append(preds)

                # 合并不同时间步长的预测
                meta_features.append(np.hstack(model_preds))

            # 2. 训练深度学习模型
            for name, model in self.base_models['dl_models'].items():
                self.logger.info(f"训练 {name} 模型...")
                model_preds = []

                for step in time_steps:
                    # PSO优化超参数
                    best_params = self.pso_optimizer.optimize(
                        model, [X_seq, X_static], y_multi[step]
                    )

                    # 使用最佳参数更新模型
                    model.compile(**best_params)

                    # 训练模型
                    history = model.fit(
                        [X_seq, X_static],
                        y_multi[step],
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=15,          # 增加耐心值
                                restore_best_weights=True,
                                min_delta=0.001
                            ),
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.2,          # 更激进的学习率衰减
                                patience=10,         # 增加耐心值
                                min_lr=1e-7,        # 降低最小学习率
                                verbose=0
                            )
                        ],
                        verbose=0
                    )

                    # 生成预测
                    preds = model.predict([X_seq, X_static])
                    model_preds.append(preds)

                # 合并不同时间步长的预测
                meta_features.append(np.hstack(model_preds))

            # 合并所有模型的预测作为元特征
            meta_features = np.hstack(meta_features)

            # 训练最终的集成模型
            self._create_final_ensemble(meta_features, y)

            self.logger.info("多步预测训练完成")
            return True

        except Exception as e:
            self.logger.error(f"多步预测训练失败: {str(e)}")
            return False

    def predict_multi_step(self, X_seq, X_static):
        """多步预测"""
        try:
            if not self.ensemble_model:
                raise ValueError("模型未训练")

            # 收集所有基础模型的预测
            meta_features = []

            # 1. 机器学习模型预测
            for model in self.base_models['ml_models'].values():
                preds = model.predict_proba(X_seq)
                meta_features.append(preds)

            # 2. 深度学习模型预测
            for model in self.base_models['dl_models'].values():
                preds = model.predict([X_seq, X_static])
                meta_features.append(preds)

            # 合并元特征
            meta_features = np.hstack(meta_features)

            # 使用集成模型进行最终预测
            predictions = self.ensemble_model.predict_proba(meta_features)

            return predictions

        except Exception as e:
            self.logger.error(f"多步预测失败: {str(e)}")
            return None

    def load_data(self, pair):
        """加载原始数据"""
        try:
            file_path = self.data_dir / f"{pair}_processed.csv"
            self.logger.info(f"加载数据: {file_path}")

            # 检查文件是否存在
            if not file_path.exists():
                self.logger.error(f"找不到文件: {file_path}")
                return None

            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # 记录特征数量（用于创建深度学习模型）
            self.n_features = len(df.columns) - 5  # 排除 Open, High, Low, Close, reSignal

            self.logger.info(f"成功加载数据，形状: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None

    def prepare_features(self, df):
        """准备特征和标签"""
        try:
            # 获取所有列名
            all_columns = df.columns.tolist()

            # 定义要排除的列
            exclude_cols = [
                'Open', 'High', 'Low', 'Close', 'reSignal',
                'returns', 'log_returns'  # 添加这两个要排除的特征
            ]

            # 获取特征列（排除价格列、标签列和收益率特征）
            feature_cols = [col for col in all_columns if col not in exclude_cols]
            features = df[feature_cols]

            # 确保所有特征都是数值型
            features = features.select_dtypes(include=[np.float64, np.int64])

            # 处理无穷值
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill')

            # 获取标签
            labels = df['reSignal']

            # 输出特征信息
            self.logger.info(f"\n使用的特征数量: {len(features.columns)}")

            return features, labels

        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            return None, None

    def _format_classification_report(self, y_true, y_pred):
        """格式化分类报告输出"""
        report = classification_report(y_true, y_pred, output_dict=True)

        # 构建格式化字符串
        lines = []
        lines.append("\n              precision    recall  f1-score   support")
        lines.append("")

        # 添加每个类别的指标
        for label in sorted(list(set(y_true))):
            metrics = report[str(label)]
            line = f"{label:>8}       {metrics['precision']:8.2f} {metrics['recall']:8.2f} {metrics['f1-score']:8.2f} {int(metrics['support']):8d}"
            lines.append(line)

        lines.append("")
        # 添加平均指标
        avg_metrics = report['weighted avg']
        lines.append(
            f"weighted avg   {avg_metrics['precision']:8.2f} {avg_metrics['recall']:8.2f} {avg_metrics['f1-score']:8.2f} {int(avg_metrics['support']):8d}")

        return "\n".join(lines)

    def _pso_optimize(self, model_name, X_train, y_train, X_val, y_val):
        """使用PSO优化模型超参数"""
        try:
            param_bounds = {
                'rf': {
                    'n_estimators': (100, 300),
                    'max_depth': (5, 15),
                    'min_samples_split': (5, 10)
                },
                'svm': {
                    'C': (0.001, 0.1),  # 进一步减小C的范围，增强正则化
                    'gamma': (0.0001, 0.01)  # 减小gamma范围，使决策边界更平滑
                },
                'kelm': {
                    'hidden_units': (500, 1500),
                    'gamma': (0.001, 0.1),
                    'C': (0.1, 5.0)
                },
                'cnn_rnn': {
                    'n_filters': (32, 256),
                    'n_lstm_units': (32, 256),
                    'learning_rate': (0.00001, 0.01),
                    'dropout_rate': (0.1, 0.5)
                },
                'deep_cnn': {
                    'n_filters': (32, 256),
                    'learning_rate': (0.00001, 0.01),
                    'dropout_rate': (0.1, 0.5)
                },
                'transformer': {
                    'n_heads': (2, 16),
                    'ff_dim': (32, 256),
                    'learning_rate': (0.00001, 0.01),
                    'dropout_rate': (0.1, 0.5)
                }
            }

            if model_name not in param_bounds:
                return None

            # 判断是否为深度学习模型
            is_dl = model_name in ['cnn_rnn', 'deep_cnn', 'transformer']

            class ParticleSwarmOptimizer:
                def __init__(self, n_particles, param_bounds, model_creator, is_dl=False):
                    self.n_particles = n_particles
                    self.param_bounds = param_bounds
                    self.model_creator = model_creator
                    self.is_dl = is_dl
                    self.positions = []
                    self.velocities = []
                    self.best_positions = []
                    self.best_scores = np.zeros(n_particles)
                    self.global_best_position = None
                    self.global_best_score = -np.inf
                    self.w = 0.7
                    self.c1 = 1.5
                    self.c2 = 1.5
                    self._initialize_particles()

                def _initialize_particles(self):
                    for _ in range(self.n_particles):
                        position = {}
                        velocity = {}
                        for param, bounds in self.param_bounds.items():
                            if isinstance(bounds[0], int):
                                position[param] = np.random.randint(bounds[0], bounds[1])
                                velocity[param] = np.random.randint(-2, 3)
                            else:
                                position[param] = np.random.uniform(bounds[0], bounds[1])
                                velocity[param] = np.random.uniform(-0.1, 0.1)
                        self.positions.append(position)
                        self.velocities.append(velocity)
                        self.best_positions.append(position.copy())

                def _update_velocity_and_position(self, i):
                    """更新粒子的速度和位置"""
                    for param in self.param_bounds.keys():
                        bounds = self.param_bounds[param]
                        r1, r2 = np.random.rand(2)

                        # 更新速度
                        self.velocities[i][param] = (
                                self.w * self.velocities[i][param] +
                                self.c1 * r1 * (self.best_positions[i][param] - self.positions[i][param]) +
                                self.c2 * r2 * (self.global_best_position[param] - self.positions[i][param])
                        )

                        # 更新位置
                        self.positions[i][param] += self.velocities[i][param]

                        # 确保在边界内
                        if isinstance(bounds[0], int):
                            self.positions[i][param] = int(np.clip(
                                self.positions[i][param], bounds[0], bounds[1]
                            ))
                        else:
                            self.positions[i][param] = np.clip(
                                self.positions[i][param], bounds[0], bounds[1]
                            )

                def evaluate_model(self, model, X_train, y_train, X_val, y_val):
                    if self.is_dl:
                        # 深度学习模型评估
                        history = model.fit(
                            X_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            verbose=0
                        )
                        # 使用验证集准确率作为得分
                        val_pred = model.predict(X_val)
                        val_pred_classes = np.argmax(val_pred, axis=1)
                        val_true_classes = np.argmax(y_val, axis=1)
                        score = (val_pred_classes == val_true_classes).mean()
                    else:
                        # 机器学习模型评估
                        model.fit(X_train, y_train)
                        score = (model.predict(X_val) == y_val).mean()
                    return score

                def optimize(self, X_train, y_train, X_val, y_val, n_iterations=1):
                    for _ in range(n_iterations):
                        for i in range(self.n_particles):
                            # 创建并评估模型
                            model = self.model_creator(**self.positions[i])
                            score = self.evaluate_model(model, X_train, y_train, X_val, y_val)

                            # 更新个体最优
                            if score > self.best_scores[i]:
                                self.best_scores[i] = score
                                self.best_positions[i] = self.positions[i].copy()

                                # 更新全局最优
                                if score > self.global_best_score:
                                    self.global_best_score = score
                                    self.global_best_position = self.positions[i].copy()

                            # 更新速度和位置
                            self._update_velocity_and_position(i)

                    return self.global_best_position, self.global_best_score

            # 创建PSO优化器
            pso = ParticleSwarmOptimizer(
                n_particles=2,
                param_bounds=param_bounds[model_name],
                model_creator=self._get_model_creator(model_name),
                is_dl=is_dl
            )

            # 运行优化
            best_params, best_score = pso.optimize(X_train, y_train, X_val, y_val, n_iterations=2)

            self.logger.info(f"\nPSO优化结果 ({model_name}):")
            self.logger.info(f"最佳参数: {best_params}")
            self.logger.info(f"最佳得分: {best_score:.4f}")

            return best_params

        except Exception as e:
            self.logger.error(f"PSO优化失败: {str(e)}")
            return None

    def _get_model_creator(self, model_name):
        """获取模型创建函数"""
        creators = {
            'rf': lambda **kwargs: RandomForestClassifier(random_state=42, **kwargs),
            'svm': lambda **kwargs: self._create_svm_model(**kwargs),
            'kelm': lambda **kwargs: self._create_kelm_model(**kwargs),
            'cnn_rnn': lambda **kwargs: self._create_cnn_rnn_model(**kwargs),
            'deep_cnn': lambda **kwargs: self._create_deep_cnn_model(**kwargs),
            'transformer': lambda **kwargs: self._create_transformer_model(**kwargs)
        }
        return creators.get(model_name)


class SignalAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_dir = Path(__file__).parent
        self.data_dir = self.current_dir / "FE"

    def load_data(self, pair):
        """加载处理后的数据"""
        try:
            file_path = self.data_dir / f"{pair}_processed.csv"
            self.logger.info(f"加载数据: {file_path}")

            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None

    def prepare_features(self, df):
        """准备特征和标签"""
        try:
            # 获取所有列名
            all_columns = df.columns.tolist()

            # 定义要排除的列
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'reSignal']

            # 获取特征列（排除价格列和标签列）
            feature_cols = [col for col in all_columns if col not in exclude_cols]
            features = df[feature_cols]

            # 确保所有特征都是数值型
            features = features.select_dtypes(include=[np.float64, np.int64])

            # 获取标签
            labels = df['reSignal']

            # 输出特征信息
            self.logger.info(f"\n使用的特征数量: {len(features.columns)}")

            return features, labels

        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            return None, None

    def analyze_signals(self, df):
        """分析原始信号和特征的关系"""
        try:
            # 获取特征和标签
            X, y = self.prepare_features(df)
            if X is None or y is None:
                return

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # 分析信号分布
            signal_dist = y.value_counts(normalize=True)
            self.logger.info("\n信号分布:")
            for signal in sorted(signal_dist.index):
                self.logger.info(f"信号 {signal}: {signal_dist[signal] * 100:.2f}%")

            # 分析每个信号对应的特征统计
            self.logger.info("\n各信号的特征统计:")
            for signal in [-1, 0, 1]:
                mask = y == signal
                if mask.any():
                    self.logger.info(f"\n信号 {signal} 的特征均值:")
                    feature_means = X[mask].mean()
                    for feature, mean in feature_means.nlargest(5).items():
                        self.logger.info(f"{feature}: {mean:.4f}")

            return X_test, y_test

        except Exception as e:
            self.logger.error(f"分析信号失败: {str(e)}")
            return None, None

    def compare_with_true_signals(self, df):
        """比较生成的信号与真实信号"""
        try:
            # 计算基于固定阈值的信号准确性
            returns = df['returns']
            threshold = 0.001  # 0.1%

            # 生成基于阈值的信号
            predicted_signals = np.zeros_like(returns)
            predicted_signals[returns.shift(-1) > threshold] = 1
            predicted_signals[returns.shift(-1) < -threshold] = -1

            # 获取真实信号
            true_signals = df['reSignal']

            # 计算准确率
            accuracy = (predicted_signals == true_signals).mean()
            self.logger.info(f"\n信号预测准确率: {accuracy:.4f}")

            # 输出混淆矩阵
            cm = confusion_matrix(true_signals, predicted_signals)
            self.logger.info("\n混淆矩阵:")
            self.logger.info(cm)

            # 输出分类报告
            self.logger.info("\n分类报告:")
            self.logger.info(classification_report(true_signals, predicted_signals))



        except Exception as e:
            self.logger.error(f"比较信号失败: {str(e)}")


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建信号保存目录
    signal_dir = Path(__file__).parent / "signals"
    signal_dir.mkdir(exist_ok=True)

    # 创建集成模型实例
    model = HybridForexModel()

    # 记录所有货币对的集成模型性能
    ensemble_performances = {}

    for pair in [ "CNYEUR", "CNYGBP", "CNYJPY", "CNYUSD"]:
        logging.info(f"\n分析 {pair}...")

        # 加载数据
        df = model.load_data(pair)
        if df is None:
            continue

        # 准备特征和标签
        X, y = model.prepare_features(df)
        if X is None or y is None:
            continue

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建基础模型
        if not model._create_base_models():  # 添加这个检查
            continue

        # 初始化预测列表
        all_predictions = []
        train_predictions = []  # 添加这行
        valid_models = []

        # 创建早停和学习率衰减的回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,          # 增加耐心值
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,          # 更激进的学习率衰减
                patience=10,         # 增加耐心值
                min_lr=1e-7,        # 降低最小学习率
                verbose=0
            )
        ]

        # 机器学习模型训练和预测
        for name, ml_model in model.base_models['ml_models'].items():
            if ml_model is None:
                continue

            logging.info(f"\n训练 {name} 模型...")

            try:
                # PSO优化超参数
                best_params = model._pso_optimize(name, X_train_scaled, y_train, X_test_scaled, y_test)
                if best_params:
                    ml_model.set_params(**best_params)

                # 根据模型类型使用不同的训练方法
                if name == 'kelm':
                    # KELM不支持validation_data参数，使用自定义的早停逻辑
                    best_score = float('-inf')
                    patience_count = 0
                    max_units = best_params.get('hidden_units', 1000)
                    step_size = 100
                    current_units = step_size

                    while patience_count < 5 and current_units <= max_units:
                        ml_model.set_params(hidden_units=current_units)
                        ml_model.fit(X_train_scaled, y_train)
                        test_pred = ml_model.predict(X_test_scaled)
                        score = np.mean(test_pred == y_test)

                        if score > best_score + 0.001:
                            best_score = score
                            patience_count = 0
                        else:
                            patience_count += 1

                        current_units += step_size
                elif name == 'rf':
                    # RandomForest的warm_start方式实现早停
                    ml_model.set_params(warm_start=True)
                    best_score = float('-inf')
                    patience_count = 0
                    n_estimators = 50  # 初始树的数量

                    while patience_count < 5 and n_estimators <= 500:
                        ml_model.set_params(n_estimators=n_estimators)
                        ml_model.fit(X_train_scaled, y_train)
                        score = ml_model.score(X_test_scaled, y_test)

                        if score > best_score + 0.001:
                            best_score = score
                            patience_count = 0
                        else:
                            patience_count += 1

                        n_estimators += 50
                elif name == 'svm':
                    # SVM的增量训练实现早停
                    ml_model.set_params(probability=True)  # 启用概率估计
                    best_score = float('-inf')
                    patience_count = 0
                    max_iter = 100
                    ml_model.set_params(
                        probability=True,  # 启用概率估计
                        cache_size=2000,  # 增加缓存大小
                        tol=1e-3,  # 调整容差
                        max_iter=1000  # 直接设置较大的迭代次数
                    )

                    while patience_count < 5 and max_iter <= 1000:
                        ml_model.set_params(max_iter=max_iter)
                        ml_model.fit(X_train_scaled, y_train)
                        score = ml_model.score(X_test_scaled, y_test)

                        if score > best_score + 0.001:
                            best_score = score
                            patience_count = 0
                        else:
                            patience_count += 1

                        max_iter += 100
                    # 对训练数据进行额外的缩放
                    X_train_svm = np.clip(X_train_scaled, -3, 3)
                    X_train_svm = X_train_svm / np.max(np.abs(X_train_svm), axis=0)
                    X_test_svm = np.clip(X_test_scaled, -3, 3)
                    X_test_svm = X_test_svm / np.max(np.abs(X_test_svm), axis=0)

                    # 一次性训练模型
                    ml_model.fit(X_train_svm, y_train)

                    # 保存处理后的数据用于后续预测
                    X_train_scaled = X_train_svm
                    X_test_scaled = X_test_svm
                else:
                    ml_model.fit(X_train_scaled, y_train)

                # 训练集预测
                y_train_pred = ml_model.predict(X_train_scaled)
                train_predictions.append(y_train_pred)  # 添加这行
                logging.info(f"{name} 训练集准确率: {(y_train_pred == y_train).mean():.4f}")

                # 测试集预测
                y_test_pred = ml_model.predict(X_test_scaled)
                all_predictions.append(y_test_pred)
                logging.info(f"{name} 测试集准确率: {(y_test_pred == y_test).mean():.4f}")
                logging.info("\n分类报告:")
                logging.info(model._format_classification_report(y_test, y_test_pred))

                valid_models.append(name)

            except Exception as e:
                logging.error(f"{name} 模型训练失败: {str(e)}")
                continue

        # 深度学习模型训练和预测
        X_train_reshaped = X_train_scaled.reshape(-1, X_train.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(-1, X_test.shape[1], 1)
        y_train_cat = to_categorical(y_train + 1)
        y_test_cat = to_categorical(y_test + 1)

        for name, dl_model in model.base_models['dl_models'].items():
            if dl_model is None:
                continue

            logging.info(f"\n训练 {name} 模型...")

            try:
                # PSO优化超参数
                best_params = model._pso_optimize(name, X_train_reshaped, y_train_cat, X_test_reshaped, y_test_cat)
                if best_params:
                    if name == 'cnn_rnn':
                        dl_model = model._create_cnn_rnn_model(**best_params)
                    elif name == 'deep_cnn':
                        dl_model = model._create_deep_cnn_model(**best_params)
                    else:  # transformer
                        dl_model = model._create_transformer_model(**best_params)

                # 所有深度学习模型使用相同的回调函数
                # 使用回调函数训练模型
                history = dl_model.fit(
                    X_train_reshaped, y_train_cat,
                    epochs=200,           # 增加最大轮数
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )

                # 检查训练历史
                if min(history.history['val_loss']) > min(history.history['loss']) * 1.5:
                    logging.warning(f"{name} 可能存在过拟合")

                # 训练集预测
                y_train_pred_prob = dl_model.predict(X_train_reshaped)
                y_train_pred = np.argmax(y_train_pred_prob, axis=1) - 1
                train_predictions.append(y_train_pred)
                logging.info(f"{name} 训练集准确率: {(y_train_pred == y_train).mean():.4f}")

                # 测试集预测
                y_test_pred_prob = dl_model.predict(X_test_reshaped)
                y_test_pred = np.argmax(y_test_pred_prob, axis=1) - 1
                all_predictions.append(y_test_pred)
                logging.info(f"{name} 测试集准确率: {(y_test_pred == y_test).mean():.4f}")
                logging.info("\n分类报告:")
                logging.info(model._format_classification_report(y_test, y_test_pred))

                valid_models.append(name)

            except Exception as e:
                logging.error(f"{name} 模型训练失败: {str(e)}")
                continue

        # 检查是否有有效的预测结果
        if not all_predictions or not train_predictions:  # 修改这行
            logging.error(f"没有成功训练的模型，跳过 {pair}")
            continue

        # 集成预测
        all_predictions = np.array(all_predictions).T
        train_predictions = np.array(train_predictions).T

        # 集成训练集预测
        ensemble_train_pred = np.zeros(len(y_train))
        for i in range(len(y_train)):
            votes = train_predictions[i]
            votes = votes.astype(int)
            unique_vals, counts = np.unique(votes, return_counts=True)
            ensemble_train_pred[i] = unique_vals[np.argmax(counts)]

        # 集成测试集预测
        ensemble_pred = np.zeros(len(y_test))
        for i in range(len(y_test)):
            votes = all_predictions[i]
            votes = votes.astype(int)
            unique_vals, counts = np.unique(votes, return_counts=True)
            ensemble_pred[i] = unique_vals[np.argmax(counts)]

        # 计算性能指标
        train_accuracy = (ensemble_train_pred == y_train).mean()
        ensemble_accuracy = (ensemble_pred == y_test).mean()

        # 输出性能
        logging.info(f"\n集成模型训练集准确率: {train_accuracy:.4f}")
        logging.info(f"\n集成模型测试集准确率: {ensemble_accuracy:.4f}")
        logging.info("\n分类报告:")
        logging.info(model._format_classification_report(y_test, ensemble_pred))

        # 保存性能记录
        ensemble_performances[pair] = ensemble_accuracy

        # 计算训练集上的集成预测
        X_train_reshaped = X_train_scaled.reshape(-1, X_train.shape[1], 1)
        train_predictions = []

        # 获取每个模型在训练集上的预测
        for name, ml_model in model.base_models['ml_models'].items():
            if ml_model is not None and hasattr(ml_model, 'predict'):
                try:
                    # 检查是否是KELM模型且未训练
                    if isinstance(ml_model, model._create_kelm_model().__class__) and \
                            (not hasattr(ml_model, 'X_train') or ml_model.X_train is None):
                        continue
                    pred = ml_model.predict(X_train_scaled)
                    train_predictions.append(pred)
                except Exception as e:
                    logging.error(f"{name} 模型训练集预测失败: {str(e)}")
                    continue

        for name, dl_model in model.base_models['dl_models'].items():
            if dl_model is not None:
                try:
                    pred_prob = dl_model.predict(X_train_reshaped)
                    train_predictions.append(np.argmax(pred_prob, axis=1) - 1)
                except Exception as e:
                    logging.error(f"{name} 模型训练集预测失败: {str(e)}")
                    continue

        # 检查是否有有效的预测结果
        if not train_predictions:
            logging.error("没有模型能够在训练集上进行预测")
            continue

        # 生成完整时间序列的预测
        full_predictions = {}
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape(X.shape[0], X.shape[1], 1)

        # 机器学习模型预测
        for name, ml_model in model.base_models['ml_models'].items():
            if ml_model is not None and hasattr(ml_model, 'predict'):
                try:
                    if isinstance(ml_model, model._create_kelm_model().__class__) and \
                            (not hasattr(ml_model, 'X_train') or ml_model.X_train is None):
                        continue
                    full_predictions[name] = ml_model.predict(X_scaled)
                except Exception as e:
                    logging.error(f"{name} 完整预测失败: {str(e)}")
                    continue

        # 深度学习模型预测
        for name, dl_model in model.base_models['dl_models'].items():
            if dl_model is not None:
                try:
                    pred_prob = dl_model.predict(X_reshaped)
                    full_predictions[name] = np.argmax(pred_prob, axis=1) - 1
                except Exception as e:
                    logging.error(f"{name} 完整预测失败: {str(e)}")
                    continue

        # 检查是否有有效的预测结果
        if not full_predictions:
            logging.error("没有模型能够生成完整预测")
            continue

        # 集成预测
        all_full_preds = np.array(list(full_predictions.values())).T
        ensemble_full_pred = np.zeros(len(X))

        for i in range(len(X)):
            votes = all_full_preds[i]
            votes = votes.astype(int)
            unique_vals, counts = np.unique(votes, return_counts=True)
            ensemble_full_pred[i] = unique_vals[np.argmax(counts)]

        # 保存预测结果
        results_df = pd.DataFrame({
            'Date': df.index,
            'Close': df['Close'],
            'True_Signal': df['reSignal'],
            'Ensemble_Signal': ensemble_full_pred  # 只保留集成信号
        })

        # 保存结果
        results_df.to_csv(signal_dir / f'{pair}_signals.csv', index=True)
        logging.info(f"\n预测结果已保存到: {signal_dir / f'{pair}_signals.csv'}")

    # 输出所有货币对的集成模型性能总结
    logging.info("\n=== 集成模型性能总结 ===")
    for pair, accuracy in ensemble_performances.items():
        logging.info(f"{pair}: {accuracy:.4f}")

    # 计算平均性能
    avg_accuracy = np.mean(list(ensemble_performances.values()))
    logging.info(f"\n平均准确率: {avg_accuracy:.4f}")


if __name__ == "__main__":
    main()

