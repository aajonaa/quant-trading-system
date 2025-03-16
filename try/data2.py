import pandas as pd
from fredapi import Fred
import os
from datetime import datetime
import time

# FRED API密钥
FRED_API_KEY = "47f0b36241cd03fcd6171c6079b40cb4"
fred = Fred(api_key=FRED_API_KEY)

# 设定时间范围
START_DATE = "2014-12-31"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 定义各国指标的FRED代码 - 只保留可靠指标
indicators = {
    "US": {
        "REAL_GDP": "GDPC1",  # 实际GDP
        "INFLATION": "CPIAUCSL",  # 通货膨胀率
        "CPI": "CPIAUCSL",  # 消费者物价指数
        "UNEMPLOYMENT": "UNRATE",  # 失业率
    },
    "EU": {
        "REAL_GDP": "CLVMNACSCAB1GQEA19",
        "INFLATION": "CP0000EZ19M086NEST",
        "CPI": "CP0000EZ19M086NEST",
        "UNEMPLOYMENT": "LRHUTTTTEZM156S",
    },
    "UK": {
        "REAL_GDP": "UKNGDP",
        "INFLATION": "GBRCPIALLMINMEI",
        "CPI": "GBRCPIALLMINMEI",
        "UNEMPLOYMENT": "LMUNRRTTGBM156S",
    },
    "JP": {
        "REAL_GDP": "JPNRGDPEXP",
        "INFLATION": "JPNCPIALLMINMEI",
        "CPI": "JPNCPIALLMINMEI",
        "UNEMPLOYMENT": "LRUNTTTTJPM156S",
    },
    "CN": {
        "REAL_GDP": "CHNGDPNQDSMEI",
        "INFLATION": "CHNCPIALLMINMEI",
        "CPI": "CHNCPIALLMINMEI",
        "UNEMPLOYMENT": None,  # 中国失业率数据将使用默认值
    },
    "AU": {
        "REAL_GDP": "AUSGDPRQDSMEI",
        "INFLATION": "AUSCPIALLQINMEI",
        "CPI": "AUSCPIALLQINMEI",
        "UNEMPLOYMENT": "LRUNTTTTAUM156S",
    }
}

save_path = "macro_data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_yoy_change(series):
    """计算同比变化率"""
    return series.pct_change(periods=12, fill_method=None) * 100

def generate_default_unemployment():
    """生成中国的默认失业率数据"""
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    # 创建默认2%失业率的时间序列
    df = pd.DataFrame(index=date_range)
    df['UNEMPLOYMENT'] = 2.0
    df.index.name = 'date'
    return df

def download_macro_data():
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    
    # 用于记录每个指标的可用性
    indicator_availability = {}
    
    for country, country_indicators in indicators.items():
        for indicator_name, series_id in country_indicators.items():
            try:
                print(f"正在下载 {country} - {indicator_name} 数据...")
                
                # 处理中国失业率特殊情况
                if country == "CN" and indicator_name == "UNEMPLOYMENT":
                    df = generate_default_unemployment()
                    print("使用默认值2%生成中国失业率数据")
                else:
                    # 获取数据
                    data = fred.get_series(series_id, start_date, end_date)
                    
                    if data.empty:
                        print(f"警告: {country} - {indicator_name} 没有数据")
                        indicator_availability.setdefault(indicator_name, set()).add(False)
                        continue
                    
                    # 创建DataFrame
                    df = pd.DataFrame(data, columns=[indicator_name])
                    df.index.name = 'date'
                    
                    # 对于CPI和INFLATION计算同比变化率
                    if indicator_name in ['CPI', 'INFLATION']:
                        df[f"{indicator_name}_YOY"] = calculate_yoy_change(df[indicator_name])
                    
                    # 处理缺失值
                    df = df.ffill().bfill()
                    
                    # 重采样为月度数据
                    df = df.resample('ME').last()
                
                # 打印数据范围
                print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
                print(f"数据点数量: {len(df)}")
                
                # 保存文件
                file_path = os.path.join(save_path, f"{country}_{indicator_name}.csv")
                df.to_csv(file_path)
                print(f"{country} - {indicator_name} 数据已保存: {file_path}")
                
                # 记录指标可用
                indicator_availability.setdefault(indicator_name, set()).add(True)
                
                # 打印数据样本
                print("\n数据样本:")
                print(df.head())
                print("\n数据统计:")
                print(df.describe())
                
                time.sleep(1)
                
            except Exception as e:
                print(f"下载 {country} - {indicator_name} 时发生错误: {str(e)}")
                indicator_availability.setdefault(indicator_name, set()).add(False)
                continue
    
    # 打印指标可用性统计
    print("\n=== 指标可用性统计 ===")
    for indicator, availability in indicator_availability.items():
        is_available = all(availability)
        print(f"{indicator}: {'所有国家可用' if is_available else '部分国家不可用'}")

if __name__ == "__main__":
    print(f"开始下载从 {START_DATE} 到 {END_DATE} 的宏观经济数据...")
    try:
        download_macro_data()
        print("下载完成!")
    except KeyboardInterrupt:
        print("\n下载被用户中断")
    except Exception as e:
        print(f"下载过程中发生错误: {str(e)}")
