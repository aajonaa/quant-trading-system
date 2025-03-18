import numpy as np
from deap import base, creator, tools, algorithms
import random
from .backtester import ForexBacktester
import pandas as pd
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class BacktestOptimizer:
    def __init__(self):
        # 添加进度跟踪相关的属性
        self.current_generation = 0
        self.total_generations = 0
        self.best_solution_history = []
        self.optimization_status = {
            'in_progress': False,
            'progress': 0,
            'current_best': None,
            'generation_stats': []
        }
        
        # 创建遗传算法的适应度类和个体类
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))  # 第一个是收益率(最大化)，第二个是回撤(最小化)
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.setup_toolbox()
        
    def setup_toolbox(self):
        """设置遗传算法工具箱"""
        # 参数范围定义
        self.param_ranges = {
            'ma_short': (5, 50),    # 短期均线
            'ma_long': (20, 200),   # 长期均线
            'rsi_period': (5, 30),  # RSI周期
            'stop_loss': (0.01, 0.1)  # 止损比例
        }
        
        # 注册参数生成函数
        self.toolbox.register("ma_short", random.randint, 
                            self.param_ranges['ma_short'][0], 
                            self.param_ranges['ma_short'][1])
        self.toolbox.register("ma_long", random.randint, 
                            self.param_ranges['ma_long'][0], 
                            self.param_ranges['ma_long'][1])
        self.toolbox.register("rsi_period", random.randint, 
                            self.param_ranges['rsi_period'][0], 
                            self.param_ranges['rsi_period'][1])
        self.toolbox.register("stop_loss", random.uniform, 
                            self.param_ranges['stop_loss'][0], 
                            self.param_ranges['stop_loss'][1])
        
        # 注册个体和种群
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.ma_short, self.toolbox.ma_long, 
                             self.toolbox.rsi_period, self.toolbox.stop_loss), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传操作
        self.toolbox.register("evaluate", self.evaluate_strategy)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selNSGA2)

    def mutate_individual(self, individual):
        """自定义突变函数"""
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            if random.random() < 0.2:  # 20%的突变概率
                if isinstance(min_val, int):
                    individual[i] = random.randint(min_val, max_val)
                else:
                    individual[i] = random.uniform(min_val, max_val)
        return individual,

    def evaluate_strategy(self, individual):
        """评估策略的适应度函数"""
        try:
            ma_short, ma_long, rsi_period, stop_loss = individual
            
            if ma_short >= ma_long:
                return -np.inf, np.inf
            
            backtester = ForexBacktester()
            strategy_params = {
                'ma_short': int(ma_short),
                'ma_long': int(ma_long),
                'rsi_period': int(rsi_period),
                'stop_loss': float(stop_loss)
            }
            
            results = backtester.run_backtest(
                currency_pair=self.currency_pair,
                start_date=self.start_date,
                end_date=self.end_date,
                strategy_params=strategy_params
            )
            
            # 更新当前最佳解
            fitness = (results['total_return'], abs(results['max_drawdown']))
            if (self.optimization_status['current_best'] is None or 
                fitness[0] > self.optimization_status['current_best']['metrics']['total_return']):
                self.optimization_status['current_best'] = {
                    'parameters': strategy_params,
                    'metrics': {
                        'total_return': fitness[0],
                        'max_drawdown': fitness[1]
                    }
                }
            
            return fitness
            
        except Exception as e:
            logger.error(f"策略评估错误: {str(e)}")
            return -np.inf, np.inf

    def update_progress(self, gen, population, stats):
        """更新优化进度"""
        self.current_generation = gen
        progress = (gen + 1) / self.total_generations * 100
        
        # 计算当前代的统计信息
        gen_stats = stats.compile(population)
        
        # 更新优化状态
        self.optimization_status.update({
            'in_progress': True,
            'progress': progress,
            'generation_stats': self.optimization_status['generation_stats'] + [{
                'generation': gen,
                'avg_return': float(gen_stats['avg'][0]),
                'max_return': float(gen_stats['max'][0]),
                'min_drawdown': float(gen_stats['min'][1]),
                'std_return': float(gen_stats['std'][0])
            }]
        })

    def optimize(self, currency_pair, start_date, end_date, 
                population_size=50, generations=30):
        """运行优化过程"""
        self.currency_pair = currency_pair
        self.start_date = start_date
        self.end_date = end_date
        self.total_generations = generations
        
        # 重置优化状态
        self.optimization_status = {
            'in_progress': False,
            'progress': 0,
            'current_best': None,
            'generation_stats': []
        }
        
        try:
            pop = self.toolbox.population(n=population_size)
            hof = tools.ParetoFront()
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            # 定义回调函数来更新进度
            def callback(gen, population):
                self.update_progress(gen, population, stats)
            
            # 运行优化算法
            final_pop, logbook = algorithms.eaMuPlusLambda(
                pop, self.toolbox,
                mu=population_size,
                lambda_=population_size,
                cxpb=0.7,
                mutpb=0.3,
                ngen=generations,
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            
            # 获取最优解
            best_solutions = []
            for solution in hof:
                ma_short, ma_long, rsi_period, stop_loss = solution
                fitness = solution.fitness.values
                best_solutions.append({
                    'parameters': {
                        'ma_short': int(ma_short),
                        'ma_long': int(ma_long),
                        'rsi_period': int(rsi_period),
                        'stop_loss': float(stop_loss)
                    },
                    'metrics': {
                        'total_return': float(fitness[0]),
                        'max_drawdown': float(fitness[1])
                    }
                })
            
            # 更新最终状态
            self.optimization_status['in_progress'] = False
            self.optimization_status['progress'] = 100
            
            return {
                'success': True,
                'solutions': best_solutions,
                'generations': self.optimization_status['generation_stats'],
                'best_solution': self.optimization_status['current_best']
            }
            
        except Exception as e:
            logger.error(f"优化过程错误: {str(e)}")
            self.optimization_status['in_progress'] = False
            return {
                'success': False,
                'error': str(e)
            }

    def get_optimization_progress(self):
        """获取优化进度"""
        return {
            'in_progress': self.optimization_status['in_progress'],
            'progress': self.optimization_status['progress'],
            'current_best': self.optimization_status['current_best'],
            'generation_stats': self.optimization_status['generation_stats']
        }

    def save_optimization_results(self, file_path):
        """保存优化结果到文件"""
        try:
            results = {
                'optimization_status': self.optimization_status,
                'best_solution_history': self.best_solution_history,
                'parameters': {
                    'currency_pair': self.currency_pair,
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            return True
        except Exception as e:
            logger.error(f"保存优化结果失败: {str(e)}")
            return False

    def load_optimization_results(self, file_path):
        """从文件加载优化结果"""
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            self.optimization_status = results['optimization_status']
            self.best_solution_history = results['best_solution_history']
            
            return True
        except Exception as e:
            logger.error(f"加载优化结果失败: {str(e)}")
            return False
