import requests
import pandas as pd
import os
import time
from datetime import datetime

# Alpha Vantage API密钥
ALPHA_VANTAGE_API_KEY = "ELWWYU1314CR68AL"

class ForexDataManager:
    def __init__(self, data_dir=r"..\try\data"):
        """初始化数据管理器"""
        self.data_dir = data_dir
        
        # 只创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def check_existing_data(self, pair):
        """检查已有的数据文件"""
        try:
            file_path = os.path.join(self.data_dir, f"{pair}.csv")
            if os.path.exists(file_path):
                # 读取CSV文件，将第一列作为索引，并命名为'Date'
                df = pd.read_csv(file_path, index_col=0)
                df.index.name = 'Date'
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            print(f"读取已有数据文件失败: {str(e)}")
        return None

    def merge_data(self, old_data, new_data):
        """合并新旧数据"""
        if old_data is None:
            return new_data
        if new_data is None:
            return old_data
        
        # 合并数据并删除重复项
        merged = pd.concat([old_data, new_data])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.sort_index(inplace=True)
        return merged

    def _download_historical_data(self, pair, start_date, end_date):
        """下载历史数据"""
        try:
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            # 尝试使用不同的数据源或API端点
            sources = [
                {
                    'name': 'Alpha Vantage FX Daily',
                    'url': f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}",
                    'data_key': "Time Series FX (Daily)"
                },
                {
                    'name': 'Alpha Vantage Time Series Daily',
                    'url': f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={from_currency}{to_currency}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}",
                    'data_key': "Time Series (Daily)"
                }
            ]
            
            for source in sources:
                try:
                    print(f"尝试从 {source['name']} 获取数据...")
                    response = requests.get(source['url'])
                    data = response.json()
                    
                    if source['data_key'] in data:
                        df = pd.DataFrame.from_dict(data[source['data_key']], orient="index")
                        if "1. open" in df.columns:
                            df.columns = [col.split(". ")[1].capitalize() for col in df.columns]
                        
                        df.index = pd.to_datetime(df.index)
                        df = df[(df.index >= start_date) & (df.index <= end_date)]
                        
                        if not df.empty:
                            print(f"成功获取数据，范围: {df.index[0]} 到 {df.index[-1]}")
                            return df
                
                except Exception as e:
                    print(f"从 {source['name']} 获取数据失败: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"下载历史数据失败: {str(e)}")
            return None

    def _download_new_data(self, pair, last_date, end_date):
        """下载新数据"""
        try:
            # 确保日期格式正确
            last_date = pd.to_datetime(last_date)
            end_date = pd.to_datetime(end_date)
            
            # 使用相同的下载逻辑
            return self._download_historical_data(pair, last_date, end_date)
        
        except Exception as e:
            print(f"下载新数据失败: {str(e)}")
            return None

    def download_currency_data(self, currency_pairs, start_date, end_date):
        """下载货币对数据"""
        data = {}
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for pair in currency_pairs:
            try:
                # 检查已有数据
                existing_data = self.check_existing_data(pair)
                if existing_data is not None:
                    print(f"找到已有数据: {pair}")
                    print(f"已有数据范围: {existing_data.index[0]} 到 {existing_data.index[-1]}")
                    
                    # 检查是否需要更新开始日期之前的数据
                    if existing_data.index[0] > start_dt:
                        print(f"需要获取 {start_dt} 到 {existing_data.index[0]} 的历史数据")
                        historical_data = self._download_historical_data(pair, start_dt, existing_data.index[0])
                        if historical_data is not None:
                            existing_data = self.merge_data(historical_data, existing_data)
                    
                    # 检查是否需要更新结束日期之后的数据
                    if existing_data.index[-1] < end_dt:
                        print(f"需要获取 {existing_data.index[-1]} 到 {end_dt} 的新数据")
                        new_data = self._download_new_data(pair, existing_data.index[-1], end_dt)
                        if new_data is not None:
                            existing_data = self.merge_data(existing_data, new_data)
                    
                    data[pair] = existing_data
                    continue
                
                # 如果没有现有数据，尝试下载完整范围的数据
                print(f"没有找到现有数据，下载完整范围的数据: {pair}")
                full_data = self._download_historical_data(pair, start_dt, end_dt)
                if full_data is not None:
                    data[pair] = full_data
                
                time.sleep(15)  # API调用间隔
                
            except Exception as e:
                print(f"处理 {pair} 时发生错误: {str(e)}")
                continue
        
        # 保存数据
        for pair, df in data.items():
            if not df.empty:
                try:
                    # 生成完整的日期范围
                    date_range = pd.date_range(start=start_dt, end=end_dt, freq='B')
                    df = df.reindex(date_range)
                    
                    # 处理缺失值
                    df['Close'] = df['Close'].ffill()
                    for col in ['Open', 'High', 'Low']:
                        df[col] = df[col].fillna(df['Close'])
                    
                    # 将索引转换为名为'Date'的列
                    df.index.name = 'Date'
                    df.to_csv(os.path.join(self.data_dir, f"{pair}.csv"))
                    print(f"数据已保存: {pair}")
                    
                except Exception as e:
                    print(f"保存 {pair} 数据时出错: {str(e)}")
        
        return data

    def process_data_and_save(self, currency_pairs, start_date, end_date):
        """处理数据并确保所有货币对的数据对齐"""
        data = self.download_currency_data(currency_pairs, start_date, end_date)
        self._print_data_statistics(data)
        return self._align_data(data)
    
    def _print_data_statistics(self, data):
        """打印数据统计信息"""
        print("\n=== 数据统计信息 ===")
        for pair, df in data.items():
            print(f"\n{pair}:")
            print(f"开始日期: {df.index[0]}")
            print(f"结束日期: {df.index[-1]}")
            print(f"总天数: {len(df)}")
            print(f"缺失天数: {df['Close'].isna().sum()}")
            
            missing_dates = df.index[df['Close'].isna()].tolist()
            if missing_dates:
                print("缺失日期示例:")
                for date in missing_dates[:5]:
                    print(f"  - {date}")
    
    def _align_data(self, data):
        """对齐所有数据集的日期"""
        common_dates = None
        for df in data.values():
            dates = df.index[df['Close'].notna()]
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates = common_dates.intersection(set(dates))
        
        if common_dates:
            common_dates = sorted(list(common_dates))
            print(f"\n使用共同日期范围: {common_dates[0]} 到 {common_dates[-1]}")
            print(f"共同日期总数: {len(common_dates)}")
            
            aligned_data = {}
            for pair in data:
                aligned_data[pair] = data[pair].loc[common_dates]
            return aligned_data
        else:
            print("警告: 没有找到共同的有效日期范围")
            return None

# 测试代码
if __name__ == "__main__":
    manager = ForexDataManager()
    currency_pairs = ["CNYUSD", "CNYAUD", "CNYEUR", "CNYGBP", "CNYJPY"]
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    currency_data = manager.process_data_and_save(currency_pairs, start_date, end_date)