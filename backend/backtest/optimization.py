import numpy as np
from backend.backtest.backtest import Backtester
import logging
import random


class BacktestOptimizer:
    """回测优化器类"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize(self, df, currency_pair, method, params):
        """
        执行回测优化

        Args:
            df: 历史数据DataFrame
            currency_pair: 货币对
            method: 优化方法 ('grid_search', 'random_search', 'bayesian_optimization', 'genetic_algorithm')
            params: 优化参数范围

        Returns:
            dict: 优化结果
        """
        try:
            # 设置更细致的参数范围
            params.update({
                'stop_loss_range': [0.01, 0.015, 0.02, 0.025, 0.03],  # 更细致的止损范围
                'take_profit_range': [0.02, 0.025, 0.03, 0.035, 0.04],  # 更细致的止盈范围
                'position_size_range': [1000, 2000, 3000, 4000, 5000],  # 更多的仓位选择
                'max_holding_days': 15,  # 最大持仓天数
                'min_holding_days': 3,  # 最小持仓天数
            })

            if method == 'grid_search':
                return self._grid_search(df, currency_pair, params)
            elif method == 'random_search':
                return self._random_search(df, currency_pair, params)
            elif method == 'bayesian_optimization':
                return self._bayesian_optimization(df, currency_pair, params)
            elif method == 'genetic_algorithm':
                return self._genetic_algorithm(df, currency_pair, params)
            else:
                raise ValueError(f"不支持的优化方法: {method}")

        except Exception as e:
            self.logger.error(f"优化过程出错: {str(e)}")
            return {
                'success': False,
                'error': f"优化过程出错: {str(e)}"
            }

    def _grid_search(self, df, currency_pair, params):
        """网格搜索优化"""
        try:
            stop_loss_range = params.get('stop_loss_range', [0.01, 0.02, 0.03])
            take_profit_range = params.get('take_profit_range', [0.02, 0.03, 0.04])
            position_size_range = params.get('position_size_range', [1000, 2000, 3000])

            best_params = None
            best_metrics = None
            best_score = float('-inf')
            optimization_history = []
            parameter_distribution = {
                'stop_loss': [],
                'take_profit': [],
                'position_size': []
            }

            backtester = Backtester()

            # 生成交易信号
            from .model_analysis import ForexModelAnalyzer
            analyzer = ForexModelAnalyzer()
            signals = analyzer.generate_ensemble_signals(df, currency_pair)

            if signals is None:
                raise ValueError("生成交易信号失败")

            # 遍历所有参数组合
            for stop_loss in stop_loss_range:
                for take_profit in take_profit_range:
                    for position_size in position_size_range:
                        current_params = {
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size,
                            'initial_capital': params.get('initial_capital', 100000),
                            'commission': params.get('commission', 0.0002),
                            'slippage': params.get('slippage', 0.0001)
                        }

                        # 运行回测
                        result = backtester.run_backtest(df, signals, current_params)

                        if result['success']:
                            metrics = result['data']['metrics']
                            score = self._calculate_score(metrics)

                            # 记录参数分布
                            parameter_distribution['stop_loss'].append(stop_loss)
                            parameter_distribution['take_profit'].append(take_profit)
                            parameter_distribution['position_size'].append(position_size)

                            # 记录优化历史
                            optimization_history.append({
                                'params': current_params,
                                'score': score,
                                'metrics': metrics
                            })

                            # 更新最佳参数
                            if score > best_score:
                                best_score = score
                                best_params = current_params.copy()
                                best_metrics = metrics.copy()

            if best_params is None:
                raise ValueError("未找到有效的优化结果")

            return {
                'success': True,
                'data': {
                    'optimization_results': {
                        'best_params': best_params,
                        'best_metrics': best_metrics,
                        'optimization_history': optimization_history,
                        'parameter_distribution': parameter_distribution
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"网格搜索优化出错: {str(e)}")
            return {
                'success': False,
                'error': f"网格搜索优化出错: {str(e)}"
            }

    def _random_search(self, df, currency_pair, params):
        """随机搜索优化"""
        try:
            n_iterations = params.get('n_iterations', 20)
            stop_loss_range = params.get('stop_loss_range', [0.01, 0.03])
            take_profit_range = params.get('take_profit_range', [0.02, 0.04])
            position_size_range = params.get('position_size_range', [1000, 5000])

            best_params = None
            best_metrics = None
            best_score = float('-inf')
            optimization_history = []
            parameter_distribution = {
                'stop_loss': [],
                'take_profit': [],
                'position_size': []
            }

            backtester = Backtester()

            # 生成交易信号
            from .model_analysis import ForexModelAnalyzer
            analyzer = ForexModelAnalyzer()
            signals = analyzer.generate_ensemble_signals(df, currency_pair)

            if signals is None:
                raise ValueError("生成交易信号失败")

            for _ in range(n_iterations):
                # 随机生成参数
                current_params = {
                    'stop_loss': random.uniform(stop_loss_range[0], stop_loss_range[1]),
                    'take_profit': random.uniform(take_profit_range[0], take_profit_range[1]),
                    'position_size': random.uniform(position_size_range[0], position_size_range[1]),
                    'initial_capital': params.get('initial_capital', 100000),
                    'commission': params.get('commission', 0.0002),
                    'slippage': params.get('slippage', 0.0001)
                }

                # 运行回测
                result = backtester.run_backtest(df, signals, current_params)

                if result['success']:
                    metrics = result['data']['metrics']
                    score = self._calculate_score(metrics)

                    # 记录参数分布
                    parameter_distribution['stop_loss'].append(current_params['stop_loss'])
                    parameter_distribution['take_profit'].append(current_params['take_profit'])
                    parameter_distribution['position_size'].append(current_params['position_size'])

                    # 记录优化历史
                    optimization_history.append({
                        'params': current_params,
                        'score': score,
                        'metrics': metrics
                    })

                    # 更新最佳参数
                    if score > best_score:
                        best_score = score
                        best_params = current_params.copy()
                        best_metrics = metrics.copy()

            if best_params is None:
                raise ValueError("未找到有效的优化结果")

            return {
                'success': True,
                'data': {
                    'optimization_results': {
                        'best_params': best_params,
                        'best_metrics': best_metrics,
                        'optimization_history': optimization_history,
                        'parameter_distribution': parameter_distribution
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"随机搜索优化出错: {str(e)}")
            return {
                'success': False,
                'error': f"随机搜索优化出错: {str(e)}"
            }

    def _bayesian_optimization(self, df, currency_pair, params):
        """贝叶斯优化"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel

            n_iterations = params.get('n_iterations', 15)
            stop_loss_range = params.get('stop_loss_range', [0.01, 0.03])
            take_profit_range = params.get('take_profit_range', [0.02, 0.04])
            position_size_range = params.get('position_size_range', [1000, 5000])

            best_params = None
            best_metrics = None
            best_score = float('-inf')
            optimization_history = []
            parameter_distribution = {
                'stop_loss': [],
                'take_profit': [],
                'position_size': []
            }

            backtester = Backtester()

            # 生成交易信号
            from .model_analysis import ForexModelAnalyzer
            analyzer = ForexModelAnalyzer()
            signals = analyzer.generate_ensemble_signals(df, currency_pair)

            if signals is None:
                raise ValueError("生成交易信号失败")

            # 初始化高斯过程
            kernel = ConstantKernel(1.0) * RBF([1.0, 1.0, 1.0])
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

            # 初始随机采样
            X = []
            y = []

            # 确保至少有一个有效的初始样本
            initial_samples = 0
            max_attempts = 10

            while initial_samples < 5 and max_attempts > 0:
                params_sample = {
                    'stop_loss': random.uniform(stop_loss_range[0], stop_loss_range[1]),
                    'take_profit': random.uniform(take_profit_range[0], take_profit_range[1]),
                    'position_size': random.uniform(position_size_range[0], position_size_range[1]),
                    'initial_capital': params.get('initial_capital', 100000),
                    'commission': params.get('commission', 0.0002),
                    'slippage': params.get('slippage', 0.0001)
                }

                result = backtester.run_backtest(df, signals, params_sample)

                if result['success']:
                    metrics = result['data']['metrics']
                    score = self._calculate_score(metrics)

                    X.append([params_sample['stop_loss'], params_sample['take_profit'], params_sample['position_size']])
                    y.append(score)

                    parameter_distribution['stop_loss'].append(params_sample['stop_loss'])
                    parameter_distribution['take_profit'].append(params_sample['take_profit'])
                    parameter_distribution['position_size'].append(params_sample['position_size'])

                    optimization_history.append({
                        'params': params_sample,
                        'score': score,
                        'metrics': metrics
                    })

                    if score > best_score:
                        best_score = score
                        best_params = params_sample.copy()
                        best_metrics = metrics.copy()

                    initial_samples += 1

                max_attempts -= 1

            if len(X) == 0:
                raise ValueError("无法获取有效的初始样本")

            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)

            # 贝叶斯优化迭代
            for _ in range(n_iterations - len(X)):
                # 拟合高斯过程
                gp.fit(X, y)

                # 采样候选点
                candidates = []
                candidate_scores = []

                for _ in range(100):
                    candidate = [
                        random.uniform(stop_loss_range[0], stop_loss_range[1]),
                        random.uniform(take_profit_range[0], take_profit_range[1]),
                        random.uniform(position_size_range[0], position_size_range[1])
                    ]
                    mean, std = gp.predict(np.array([candidate]), return_std=True)
                    acquisition = mean + 1.96 * std  # Upper confidence bound
                    candidates.append(candidate)
                    candidate_scores.append(acquisition)

                # 选择最佳候选点
                best_candidate_idx = np.argmax(candidate_scores)
                best_candidate = candidates[best_candidate_idx]

                current_params = {
                    'stop_loss': best_candidate[0],
                    'take_profit': best_candidate[1],
                    'position_size': best_candidate[2],
                    'initial_capital': params.get('initial_capital', 100000),
                    'commission': params.get('commission', 0.0002),
                    'slippage': params.get('slippage', 0.0001)
                }

                result = backtester.run_backtest(df, signals, current_params)

                if result['success']:
                    metrics = result['data']['metrics']
                    score = self._calculate_score(metrics)

                    X = np.vstack([X, best_candidate])
                    y = np.append(y, score)

                    parameter_distribution['stop_loss'].append(current_params['stop_loss'])
                    parameter_distribution['take_profit'].append(current_params['take_profit'])
                    parameter_distribution['position_size'].append(current_params['position_size'])

                    optimization_history.append({
                        'params': current_params,
                        'score': score,
                        'metrics': metrics
                    })

                    if score > best_score:
                        best_score = score
                        best_params = current_params.copy()
                        best_metrics = metrics.copy()

            if best_params is None:
                raise ValueError("未找到有效的优化结果")

            return {
                'success': True,
                'data': {
                    'optimization_results': {
                        'best_params': best_params,
                        'best_metrics': best_metrics,
                        'optimization_history': optimization_history,
                        'parameter_distribution': parameter_distribution
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"贝叶斯优化出错: {str(e)}")
            return {
                'success': False,
                'error': f"贝叶斯优化出错: {str(e)}"
            }

    def _genetic_algorithm(self, df, currency_pair, params):
        """遗传算法优化"""
        try:
            population_size = params.get('population_size', 20)
            n_generations = params.get('n_generations', 10)
            stop_loss_range = params.get('stop_loss_range', [0.01, 0.03])
            take_profit_range = params.get('take_profit_range', [0.02, 0.04])
            position_size_range = params.get('position_size_range', [1000, 5000])

            best_params = None
            best_metrics = None
            best_score = float('-inf')
            optimization_history = []
            parameter_distribution = {
                'stop_loss': [],
                'take_profit': [],
                'position_size': []
            }

            backtester = Backtester()

            # 生成交易信号
            from .model_analysis import ForexModelAnalyzer
            analyzer = ForexModelAnalyzer()
            signals = analyzer.generate_ensemble_signals(df, currency_pair)

            if signals is None:
                raise ValueError("生成交易信号失败")

            # 初始化种群
            population = []
            for _ in range(population_size):
                individual = {
                    'stop_loss': random.uniform(stop_loss_range[0], stop_loss_range[1]),
                    'take_profit': random.uniform(take_profit_range[0], take_profit_range[1]),
                    'position_size': random.uniform(position_size_range[0], position_size_range[1]),
                    'initial_capital': params.get('initial_capital', 100000),
                    'commission': params.get('commission', 0.0002),
                    'slippage': params.get('slippage', 0.0001)
                }
                population.append(individual)

            # 遗传算法迭代
            for generation in range(n_generations):
                # 评估适应度
                fitness_scores = []
                generation_results = []

                for individual in population:
                    result = backtester.run_backtest(df, signals, individual)

                    if result['success']:
                        metrics = result['data']['metrics']
                        score = self._calculate_score(metrics)
                        fitness_scores.append(score)
                        generation_results.append({
                            'params': individual.copy(),
                            'metrics': metrics.copy(),
                            'score': score
                        })

                        parameter_distribution['stop_loss'].append(individual['stop_loss'])
                        parameter_distribution['take_profit'].append(individual['take_profit'])
                        parameter_distribution['position_size'].append(individual['position_size'])

                        optimization_history.append({
                            'params': individual.copy(),
                            'score': score,
                            'metrics': metrics.copy(),
                            'generation': generation
                        })

                        if score > best_score:
                            best_score = score
                            best_params = individual.copy()
                            best_metrics = metrics.copy()
                    else:
                        fitness_scores.append(float('-inf'))
                        generation_results.append(None)

                # 如果没有有效的结果，继续下一代
                if all(score == float('-inf') for score in fitness_scores):
                    continue

                # 选择
                selected_indices = np.argsort(fitness_scores)[-population_size // 2:]
                selected = [population[i] for i in selected_indices]

                # 交叉和变异
                new_population = []

                # 保留精英个体
                elite_count = max(1, population_size // 10)
                elite_indices = np.argsort(fitness_scores)[-elite_count:]
                new_population.extend([population[i].copy() for i in elite_indices])

                # 生成新个体
                while len(new_population) < population_size:
                    if len(selected) >= 2:
                        parent1, parent2 = random.sample(selected, 2)

                        # 交叉
                        child = {
                            'stop_loss': (parent1['stop_loss'] + parent2['stop_loss']) / 2,
                            'take_profit': (parent1['take_profit'] + parent2['take_profit']) / 2,
                            'position_size': (parent1['position_size'] + parent2['position_size']) / 2,
                            'initial_capital': params.get('initial_capital', 100000),
                            'commission': params.get('commission', 0.0002),
                            'slippage': params.get('slippage', 0.0001)
                        }

                        # 变异
                        if random.random() < 0.1:  # 变异率
                            mutation_type = random.choice(['stop_loss', 'take_profit', 'position_size'])
                            if mutation_type == 'stop_loss':
                                child['stop_loss'] = random.uniform(stop_loss_range[0], stop_loss_range[1])
                            elif mutation_type == 'take_profit':
                                child['take_profit'] = random.uniform(take_profit_range[0], take_profit_range[1])
                            else:
                                child['position_size'] = random.uniform(position_size_range[0], position_size_range[1])

                        new_population.append(child)
                    else:
                        # 如果选择的个体不足，生成新的随机个体
                        new_individual = {
                            'stop_loss': random.uniform(stop_loss_range[0], stop_loss_range[1]),
                            'take_profit': random.uniform(take_profit_range[0], take_profit_range[1]),
                            'position_size': random.uniform(position_size_range[0], position_size_range[1]),
                            'initial_capital': params.get('initial_capital', 100000),
                            'commission': params.get('commission', 0.0002),
                            'slippage': params.get('slippage', 0.0001)
                        }
                        new_population.append(new_individual)

                population = new_population

            if best_params is None:
                raise ValueError("未找到有效的优化结果")

            return {
                'success': True,
                'data': {
                    'optimization_results': {
                        'best_params': best_params,
                        'best_metrics': best_metrics,
                        'optimization_history': optimization_history,
                        'parameter_distribution': parameter_distribution
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"遗传算法优化出错: {str(e)}")
            return {
                'success': False,
                'error': f"遗传算法优化出错: {str(e)}"
            }

    def _calculate_score(self, metrics):
        """计算优化得分"""
        try:
            # 提取指标并处理可能的无效值
            total_return = float(metrics.get('total_return', 0))
            sharpe_ratio = float(metrics.get('sharpe_ratio', 0))
            max_drawdown = abs(float(metrics.get('max_drawdown', 0)))
            win_rate = float(metrics.get('win_rate', 0))
            avg_trade_duration = float(metrics.get('avg_trade_duration', 0))  # 平均持仓时间（天）

            # 处理无效值
            if np.isnan(total_return) or np.isinf(total_return):
                total_return = 0
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                sharpe_ratio = 0
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                max_drawdown = 100
            if np.isnan(win_rate) or np.isinf(win_rate):
                win_rate = 0
            if np.isnan(avg_trade_duration) or np.isinf(avg_trade_duration):
                avg_trade_duration = 30  # 默认30天

            # 标准化指标
            total_return = np.clip(total_return, -100, 100) / 100
            sharpe_ratio = np.clip(sharpe_ratio, -10, 10) / 10
            max_drawdown = np.clip(max_drawdown, 0, 100) / 100
            win_rate = np.clip(win_rate, 0, 100) / 100

            # 持仓时间评分（偏好3-15天的交易）
            duration_score = 1.0
            if avg_trade_duration < 3:
                duration_score = avg_trade_duration / 3
            elif avg_trade_duration > 15:
                duration_score = max(0, 1 - (avg_trade_duration - 15) / 15)

            # 最大回撤惩罚（当最大回撤超过15%时显著降低得分）
            drawdown_penalty = 1.0
            if max_drawdown > 0.15:  # 15%
                drawdown_penalty = max(0, 1 - (max_drawdown - 0.15) * 3)

            # 计算综合得分
            # 总收益率权重30%
            # 夏普比率权重20%
            # 最大回撤权重25%（负相关）
            # 胜率权重15%
            # 持仓时间评分权重10%
            score = (
                            0.30 * total_return +
                            0.20 * sharpe_ratio -
                            0.25 * max_drawdown +
                            0.15 * win_rate +
                            0.10 * duration_score
                    ) * drawdown_penalty  # 应用最大回撤惩罚

            return float(score)

        except Exception as e:
            self.logger.error(f"计算优化得分时出错: {str(e)}")
            return float('-inf')