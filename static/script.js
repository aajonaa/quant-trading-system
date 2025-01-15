// 全局变量存储图表实例
let priceChart = null;
let indicatorsChart = null;
let volumeChart = null;
let correlationChart = null;
let riskDistributionChart = null;
let featureImportanceChart = null;

// 当文档加载完成时执行
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('forexForm');
    
    form.addEventListener('submit', async function(event) {
        event.preventDefault();  // 阻止表单默认提交行为
        
        const button = event.submitter;
        const action = button.getAttribute('data-action');
        const currencyPair = document.getElementById('currency_pair').value;
        
        if (!currencyPair) {
            showError('请选择货币对');
            return;
        }
        
        // 隐藏欢迎区域
        const welcomeContainer = document.getElementById('welcomeContainer');
        if (welcomeContainer) {
            welcomeContainer.style.display = 'none';
        }
        
        // 显示加载动画
        document.getElementById('loading').style.display = 'block';
        // 隐藏之前的错误和成功消息
        document.getElementById('error').style.display = 'none';
        document.getElementById('success').style.display = 'none';
        
        try {
            console.log('Action:', action);  // 调试日志
            // 根据不同的操作调用相应的API
            switch(action) {
                case 'loadData':
                    await fetchData(currencyPair);
                    break;
                case 'generateSignals':
                    await generateSignals();
                    break;
                case 'runBacktest':
                    await runBacktest(currencyPair);
                    break;
                case 'explain_signal':
                    await explainSignal(currencyPair);
                    break;
                case 'analyze_portfolio_risk':
                    await analyzePortfolioRisk();
                    break;
                case 'optimize_backtest':
                    const method = document.getElementById('optimization_method').value;
                    await optimizeBacktest(currencyPair, method);
                    break;
                case 'runOptimizedBacktest':
                    const optimizationMethod = document.getElementById('optimization_method').value;
                    await runOptimizedBacktest(currencyPair, optimizationMethod);
                    break;
            }
        } catch (error) {
            showError(error.message);
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    });
});

// API调用函数
async function fetchData(currencyPair) {
    console.log('开始加载数据:', currencyPair);
    const response = await fetch('/load_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ currency_pair: currencyPair })
    });
    
    const result = await response.json();
    
    if (!response.ok) {
        throw new Error(result.error || '加载数据失败');
    }
    
    if (result.success) {
        displayData(result.data);
        showSuccess('数据加载成功');
    } else {
        throw new Error(result.error || '加载数据失败');
    }
}

// 添加生成信号函数
async function generateSignals() {
    try {
        const currencyPair = document.getElementById('currency_pair').value;
        if (!currencyPair) {
            showError('请选择货币对');
            return;
        }
        
        // 显示加载动画
        document.getElementById('loading').style.display = 'block';
        
        // 清空之前的内容
        const contentContainer = document.getElementById('contentContainer');
        contentContainer.innerHTML = `
            <div class="row">
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title">价格和交易信号</h3>
                            <div style="height: 400px;">
                                <canvas id="priceChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title">相关性矩阵</h3>
                            <div style="height: 400px;">
                                <canvas id="correlationChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title">风险指标</h3>
                            <div id="riskMetricsContainer"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 生成信号
        const signalResponse = await fetch('/generate_signals', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ currency_pair: currencyPair })
        });

        const signalResponseText = await signalResponse.text();
        let signalResult;
        try {
            const cleanedSignalText = signalResponseText.replace(/:\s*NaN/g, ': null');
            signalResult = JSON.parse(cleanedSignalText);
            console.log('信号生成数据:', signalResult);

            if (!signalResult.success) {
                throw new Error(signalResult.error || '生成信号失败');
            }

            // 更新价格图表
            await updateChart(signalResult.data);

            // 处理风险分析数据
            if (signalResult.risk_analysis) {
                const riskAnalysis = signalResult.risk_analysis;
                
                // 创建相关性矩阵
                if (riskAnalysis.data && riskAnalysis.data.correlation_matrix) {
                    createCorrelationChart(riskAnalysis.data.correlation_matrix);
                }

                // 显示风险指标
                if (riskAnalysis.data && riskAnalysis.data.risk_signals) {
                    const riskMetricsContainer = document.getElementById('riskMetricsContainer');
                    riskMetricsContainer.innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h4>投资组合风险指标</h4>
                                <table class="table table-striped">
                                    <tbody>
                                        <tr>
                                            <th>年化收益率</th>
                                            <td>${riskAnalysis.data.portfolio_risk?.annual_return || 0}%</td>
                                        </tr>
                                        <tr>
                                            <th>组合波动率</th>
                                            <td>${riskAnalysis.data.portfolio_risk?.portfolio_volatility || 0}%</td>
                                        </tr>
                                        <tr>
                                            <th>夏普比率</th>
                                            <td>${riskAnalysis.data.portfolio_risk?.sharpe_ratio || 0}</td>
                                        </tr>
                                        <tr>
                                            <th>95% VaR</th>
                                            <td>${riskAnalysis.data.portfolio_risk?.var_95 || 0}%</td>
                                        </tr>
                                        <tr>
                                            <th>95% CVaR</th>
                                            <td>${riskAnalysis.data.portfolio_risk?.cvar_95 || 0}%</td>
                                        </tr>
                                        <tr>
                                            <th>风险等级</th>
                                            <td>
                                                <span class="badge ${getBadgeClass(riskAnalysis.data.portfolio_risk?.risk_level)}">
                                                    ${riskAnalysis.data.portfolio_risk?.risk_level || '未知'}
                                                </span>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h4>货币对风险信号</h4>
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>货币对</th>
                                            <th>风险评分</th>
                                            <th>信号</th>
                                            <th>风险等级</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${Object.entries(riskAnalysis.data.risk_signals || {}).map(([pair, signal]) => `
                                            <tr>
                                                <td>${pair}</td>
                                                <td>${signal.risk_score || 0}</td>
                                                <td>
                                                    <span class="badge ${getSignalBadgeClass(signal.signal)}">
                                                        ${signal.signal || '未知'}
                                                    </span>
                                                </td>
                                                <td>
                                                    <span class="badge ${getBadgeClass(signal.risk_level)}">
                                                        ${signal.risk_level || '未知'}
                                                    </span>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }
            }

            // 显示成功消息
            showSuccess('信号生成和风险分析完成');
        } catch (e) {
            console.error('处理数据时出错:', e);
            console.error('原始响应:', signalResponseText);
            throw new Error('处理数据时出错: ' + e.message);
        }
    } catch (error) {
        console.error('操作出错:', error);
        showError(error.message);
    } finally {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';
    }
}

// 获取风险等级对应的Badge类
function getBadgeClass(riskLevel) {
    switch (riskLevel) {
        case '低风险':
            return 'bg-success';
        case '中等风险':
            return 'bg-warning';
        case '高风险':
            return 'bg-danger';
        default:
            return 'bg-secondary';
    }
}

// 获取信号对应的Badge类
function getSignalBadgeClass(signal) {
    switch (signal) {
        case '看涨':
            return 'bg-success';
        case '看跌':
            return 'bg-danger';
        case '中性':
            return 'bg-warning';
        default:
            return 'bg-secondary';
    }
}

// 修改相关性矩阵图表的创建函数
function createCorrelationChart(correlationData) {
    try {
        const correlationCanvas = document.getElementById('correlationChart');
        if (!correlationCanvas) {
            throw new Error('找不到相关性矩阵Canvas元素');
        }

        // 销毁旧的图表实例
        if (window.correlationChart instanceof Chart) {
            window.correlationChart.destroy();
        }

        const labels = Object.keys(correlationData);
        const values = [];
        
        // 构建数据点
        labels.forEach((row, i) => {
            labels.forEach((col, j) => {
                values.push({
                    x: j,
                    y: i,
                    value: correlationData[row][col]
                });
            });
        });

        const ctx = correlationCanvas.getContext('2d');
        window.correlationChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: values.map(v => {
                        const value = v.value;
                        if (Math.abs(value - 1) < 0.001) {
                            return 'rgb(165, 0, 38)';  // 深红色，表示完全正相关
                        } else if (value > 0) {
                            const intensity = Math.abs(value);
                            return `rgba(165, 0, 38, ${intensity})`;  // 红色系
                        } else {
                            const intensity = Math.abs(value);
                            return `rgba(49, 54, 149, ${intensity})`;  // 蓝色系
                        }
                    }),
                    pointStyle: 'rect',
                    pointRadius: 25,
                    pointHoverRadius: 30
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                const row = labels[point.y];
                                const col = labels[point.x];
                                return `${row} vs ${col}: ${point.value.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: labels.length - 0.5,
                        ticks: {
                            callback: function(value) {
                                return labels[value] || '';
                            },
                            maxRotation: 45,
                            minRotation: 45
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        type: 'linear',
                        min: -0.5,
                        max: labels.length - 0.5,
                        reverse: true,
                        ticks: {
                            callback: function(value) {
                                return labels[value] || '';
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                animation: false
            }
        });

        // 在图表上添加相关系数值
        const xScale = window.correlationChart.scales.x;
        const yScale = window.correlationChart.scales.y;
        
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px Arial';

        values.forEach(v => {
            const x = xScale.getPixelForValue(v.x);
            const y = yScale.getPixelForValue(v.y);
            ctx.fillText(v.value.toFixed(2), x, y);
        });

    } catch (error) {
        console.error('创建相关性矩阵时出错:', error);
        showError('创建相关性矩阵时出错: ' + error.message);
    }
}

// 显示信号数据的函数
function displaySignalsData(data) {
    const contentContainer = document.getElementById('contentContainer');
    
    // 验证数据
    if (!Array.isArray(data) || data.length === 0) {
        contentContainer.innerHTML = '<div class="alert alert-warning">没有可用的数据</div>';
        return;
    }
    
    // 创建图表容器
    const chartsHtml = `
        <div class="row">
            <!-- 主图表区域 -->
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">价格和交易信号</h3>
                        <div style="height: 400px;">
                            <canvas id="priceChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 相关性矩阵区域 -->
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">相关性分析</h3>
                        <div class="row">
                            <div class="col-md-8">
                                <div style="height: 400px;">
                                    <canvas id="correlationHeatmap"></canvas>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="correlation-legend mt-3">
                                    <h5>相关性说明</h5>
                                    <div class="correlation-scale">
                                        <div class="d-flex justify-content-between">
                                            <span>强负相关</span>
                                            <span>无相关</span>
                                            <span>强正相关</span>
                                        </div>
                                        <div class="correlation-gradient"></div>
                                        <div class="d-flex justify-content-between">
                                            <span>-1.0</span>
                                            <span>0.0</span>
                                            <span>1.0</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 技术指标区域 -->
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">技术指标</h3>
                        <div style="height: 300px;">
                            <canvas id="indicatorsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 交易量区域 -->
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">交易量分析</h3>
                        <div style="height: 300px;">
                            <canvas id="volumeChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 波动率分析 -->
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">波动率分析</h3>
                        <div style="height: 300px;">
                            <canvas id="volatilityChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 信号强度分析 -->
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">信号强度分析</h3>
                        <div style="height: 300px;">
                            <canvas id="signalStrengthChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 信号统计信息 -->
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">信号统计</h3>
                        <div id="signalStats" class="row">
                            <div class="col-md-3">
                                <div class="alert alert-info">
                                    <h5>买入信号</h5>
                                    <span id="buySignals">0</span>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="alert alert-danger">
                                    <h5>卖出信号</h5>
                                    <span id="sellSignals">0</span>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="alert alert-warning">
                                    <h5>持仓信号</h5>
                                    <span id="holdSignals">0</span>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="alert alert-success">
                                    <h5>信号准确率</h5>
                                    <span id="signalAccuracy">0%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 创建数据表格
    const tableHtml = createDataTable(data);
    
    // 更新页面内容
    contentContainer.innerHTML = chartsHtml + tableHtml;
    
    // 初始化所有图表
    initializePriceChart(data);
    initializeCorrelationHeatmap(data);
    initializeIndicatorsChart(data);
    initializeVolumeChart(data);
    initializeVolatilityChart(data);
    initializeSignalStrengthChart(data);
    updateSignalStats(data);
}

// 初始化价格图表
function initializePriceChart(data) {
    const ctx = document.getElementById('priceChart');
    if (!ctx) return;
    
    const chartData = data.map(item => ({
        x: new Date(item.Date),
        y: parseFloat(item.Close)
    })).sort((a, b) => a.x - b.x);
    
    // 添加信号点
    const signalPoints = data.map(item => ({
        x: new Date(item.Date),
        y: parseFloat(item.Close),
        signal: item.Signal
    })).filter(item => item.signal !== 0);
    
    if (window.priceChart instanceof Chart) {
        window.priceChart.destroy();
    }
    
    window.priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: '收盘价',
                    data: chartData,
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: '买入信号',
                    data: signalPoints.filter(p => p.signal > 0),
                    backgroundColor: 'green',
                    borderColor: 'green',
                    pointStyle: 'triangle',
                    pointRadius: 8,
                    showLine: false
                },
                {
                    label: '卖出信号',
                    data: signalPoints.filter(p => p.signal < 0),
                    backgroundColor: 'red',
                    borderColor: 'red',
                    pointStyle: 'triangle',
                    pointRadius: 8,
                    rotation: 180,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'YYYY-MM-DD'
                        }
                    },
                    title: {
                        display: true,
                        text: '日期'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '价格'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '价格走势与交易信号'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// 初始化技术指标图表
function initializeIndicatorsChart(data) {
    const ctx = document.getElementById('indicatorsChart');
    if (!ctx) return;
    
    const chartData = data.map(item => ({
        x: new Date(item.Date),
        ma10: parseFloat(item.MA_10),
        ma50: parseFloat(item.MA_50),
        rsi: parseFloat(item.RSI)
    })).sort((a, b) => a.x - b.x);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'MA10',
                    data: chartData.map(d => ({ x: d.x, y: d.ma10 })),
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'MA50',
                    data: chartData.map(d => ({ x: d.x, y: d.ma50 })),
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'RSI',
                    data: chartData.map(d => ({ x: d.x, y: d.rsi })),
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1,
                    fill: false,
                    yAxisID: 'rsi'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '价格'
                    }
                },
                rsi: {
                    position: 'right',
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'RSI'
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// 初始化交易量图表
function initializeVolumeChart(data) {
    const ctx = document.getElementById('volumeChart');
    if (!ctx) return;
    
    const chartData = data.map(item => ({
        x: new Date(item.Date),
        y: parseFloat(item.Volume)
    })).sort((a, b) => a.x - b.x);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [{
                label: '交易量',
                data: chartData,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '交易量'
                    }
                }
            }
        }
    });
}

// 初始化波动率图表
function initializeVolatilityChart(data) {
    const ctx = document.getElementById('volatilityChart');
    if (!ctx) return;
    
    // 计算20日波动率
    const volatility = [];
    for (let i = 20; i < data.length; i++) {
        const prices = data.slice(i-20, i).map(d => parseFloat(d.Close));
        const returns = prices.map((p, j) => j === 0 ? 0 : Math.log(p / prices[j-1]));
        const std = Math.sqrt(returns.reduce((a, b) => a + b * b, 0) / 19) * Math.sqrt(252) * 100;
        volatility.push({
            x: new Date(data[i].Date),
            y: std
        });
    }
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: '年化波动率 (%)',
                data: volatility,
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '波动率 (%)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// 初始化信号强度图表
function initializeSignalStrengthChart(data) {
    const ctx = document.getElementById('signalStrengthChart');
    if (!ctx) return;
    
    // 计算信号强度
    const signalStrength = data.map(item => ({
        x: new Date(item.Date),
        y: item.Signal ? Math.abs(item.Signal) : 0
    }));
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: '信号强度',
                data: signalStrength,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '信号强度'
                    },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// 更新信号统计信息
function updateSignalStats(data) {
    const buySignals = data.filter(d => d.Signal > 0).length;
    const sellSignals = data.filter(d => d.Signal < 0).length;
    const holdSignals = data.filter(d => d.Signal === 0).length;
    
    // 计算信号准确率（基于后续价格变动）
    let correctSignals = 0;
    let totalSignals = 0;
    
    for (let i = 0; i < data.length - 1; i++) {
        if (data[i].Signal !== 0) {
            totalSignals++;
            const priceChange = (data[i+1].Close - data[i].Close) / data[i].Close;
            if ((data[i].Signal > 0 && priceChange > 0) || (data[i].Signal < 0 && priceChange < 0)) {
                correctSignals++;
            }
        }
    }
    
    const accuracy = totalSignals > 0 ? (correctSignals / totalSignals * 100).toFixed(2) : 0;
    
    // 更新DOM
    document.getElementById('buySignals').textContent = buySignals;
    document.getElementById('sellSignals').textContent = sellSignals;
    document.getElementById('holdSignals').textContent = holdSignals;
    document.getElementById('signalAccuracy').textContent = `${accuracy}%`;
}

// 更新图表函数
async function updateChart(data) {
    try {
        // 检查必要的Canvas元素是否存在
        const priceCanvas = document.getElementById('priceChart');
        if (!priceCanvas) {
            throw new Error('找不到必要的图表Canvas元素');
        }

        // 销毁旧的图表实例
        if (priceChart) {
            priceChart.destroy();
        }

        // 准备数据
        const dates = data.map(d => d.Date);
        const prices = data.map(d => ({
            open: parseFloat(d.Open),
            high: parseFloat(d.High),
            low: parseFloat(d.Low),
            close: parseFloat(d.Close)
        }));
        const signals = data.map(d => d.Signal);

        // 创建价格图表
        const priceCtx = priceCanvas.getContext('2d');
        priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '收盘价',
                    data: prices.map(p => p.close),
                        borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                        fill: false,
                    yAxisID: 'y'
                }, {
                    label: '买入信号',
                    data: signals.map((s, i) => s === 1 ? prices[i].close : null),
                    pointBackgroundColor: 'rgb(0, 255, 0)',
                    pointBorderColor: 'rgb(0, 255, 0)',
                    pointStyle: 'triangle',
                    pointRadius: 8,
                    showLine: false,
                    yAxisID: 'y'
                }, {
                    label: '卖出信号',
                    data: signals.map((s, i) => s === -1 ? prices[i].close : null),
                    pointBackgroundColor: 'rgb(255, 0, 0)',
                    pointBorderColor: 'rgb(255, 0, 0)',
                    pointStyle: 'triangle',
                    pointRadius: 8,
                    showLine: false,
                    yAxisID: 'y'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: '日期'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: '价格'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(4);
                                }
                                return label;
                            }
                        }
                    },
                            title: {
                                display: true,
                        text: '价格走势与交易信号'
                        }
                    }
                }
            });
    } catch (error) {
        console.error('更新图表时出错:', error);
        showError('更新图表时出错: ' + error.message);
        throw error;
    }
}

// 创建风险聚类信息的函数
function createRiskClusterInfo(clusters) {
    const clusterCounts = {
        0: clusters.filter(x => x === 0).length,
        1: clusters.filter(x => x === 1).length,
        2: clusters.filter(x => x === 2).length
    };
    
    return `
        <div class="risk-cluster-stats">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>风险等级</th>
                        <th>数量</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="badge bg-success">低风险</span></td>
                        <td>${clusterCounts[0]}</td>
                        <td>${((clusterCounts[0] / clusters.length) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td><span class="badge bg-warning">中风险</span></td>
                        <td>${clusterCounts[1]}</td>
                        <td>${((clusterCounts[1] / clusters.length) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td><span class="badge bg-danger">高风险</span></td>
                        <td>${clusterCounts[2]}</td>
                        <td>${((clusterCounts[2] / clusters.length) * 100).toFixed(1)}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
}

// 添加显示信号数据的函数
function displaySignalsData(data) {
    const contentContainer = document.getElementById('contentContainer');
    contentContainer.innerHTML = createSignalsTable(data);
}

// 添加回测函数
async function runBacktest(currencyPair) {
    try {
        console.log('开始运行回测:', currencyPair);
        
        // 显示加载动画
        document.getElementById('loading').style.display = 'block';
        
        // 清空之前的内容
        const contentContainer = document.getElementById('contentContainer');
        contentContainer.innerHTML = `
            <div class="row">
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title">回测结果</h3>
                            <div id="backtest-results"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        const response = await fetch('/run_backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                currency_pair: currencyPair,
                params: {
                    stop_loss: 0.02,
                    take_profit: 0.03,
                    position_size: 1000,
                    initial_capital: 100000,
                    commission: 0.0002,
                    slippage: 0.0001
                }
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '运行回测失败');
        }
        
        // 显示回测结果
        displayBacktestResults(result.data);
        showSuccess('回测完成');
        
    } catch (error) {
        console.error('回测出错:', error);
        showError(error.message);
    } finally {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';
    }
}

// 显示回测结果
function displayBacktestResults(results) {
    const resultsContainer = document.getElementById('backtest-results');
    if (!resultsContainer) return;
    
    if (!results || !results.metrics) {
        resultsContainer.innerHTML = '<div class="alert alert-warning">无效的回测结果</div>';
        return;
    }
    
    const metrics = results.metrics;
    const trades = results.trades || [];
    
    resultsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h4>回测指标</h4>
                <table class="table table-striped">
                    <tbody>
                        <tr>
                            <th>总交易次数</th>
                            <td>${metrics.total_trades || 0}</td>
                        </tr>
                        <tr>
                            <th>盈利交易</th>
                            <td>${metrics.profitable_trades || 0}</td>
                        </tr>
                        <tr>
                            <th>胜率</th>
                            <td>${(metrics.win_rate || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>总收益率</th>
                            <td>${(metrics.total_return || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>最大回撤</th>
                            <td>${(metrics.max_drawdown || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>夏普比率</th>
                            <td>${(metrics.sharpe_ratio || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>盈亏比</th>
                            <td>${(metrics.profit_factor || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>平均盈利</th>
                            <td>${(metrics.average_profit || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>平均亏损</th>
                            <td>${(metrics.average_loss || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>风险收益比</th>
                            <td>${(metrics.risk_reward_ratio || 0).toFixed(2)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="col-md-6">
                <h4>最近交易记录</h4>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>类型</th>
                                <th>入场价</th>
                                <th>出场价</th>
                                <th>收益</th>
                                <th>收益率</th>
                                <th>持仓天数</th>
                                <th>平仓原因</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${trades.slice(-5).map(trade => `
                                <tr>
                                    <td>${trade.date || ''}</td>
                                    <td>${getTradeTypeText(trade.type)}</td>
                                    <td>${(trade.entry_price || 0).toFixed(4)}</td>
                                    <td>${(trade.exit_price || 0).toFixed(4)}</td>
                                    <td class="${(trade.profit || 0) > 0 ? 'text-success' : 'text-danger'}">${(trade.profit || 0).toFixed(2)}</td>
                                    <td class="${(trade.profit_pct || 0) > 0 ? 'text-success' : 'text-danger'}">${(trade.profit_pct || 0).toFixed(2)}%</td>
                                    <td>${trade.holding_days || 0}</td>
                                    <td>${getExitReasonText(trade.exit_reason)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// 获取交易类型文本
function getTradeTypeText(type) {
    switch (type) {
        case 'buy':
            return '<span class="badge bg-success">买入</span>';
        case 'sell':
            return '<span class="badge bg-danger">卖出</span>';
        default:
            return '<span class="badge bg-secondary">未知</span>';
    }
}

// 获取退出原因文本
function getExitReasonText(reason) {
    switch (reason) {
        case 'stop_loss':
            return '<span class="badge bg-danger">止损</span>';
        case 'take_profit':
            return '<span class="badge bg-success">止盈</span>';
        case 'signal':
            return '<span class="badge bg-primary">信号</span>';
        case 'final':
            return '<span class="badge bg-secondary">结束</span>';
        default:
            return '<span class="badge bg-secondary">未知</span>';
    }
}

// 添加解释函数
async function explainSignal(currencyPair) {
    console.log('开始生成解释:', currencyPair);
    const response = await fetch('/explain_signal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ currency_pair: currencyPair })
    });
    
    const result = await response.json();
    
    if (!response.ok) {
        throw new Error(result.error || '生成解释失败');
    }
    
    if (result.success) {
        displayExplanation(result.explanation);
        showSuccess('解释生成成功');
    } else {
        throw new Error(result.error || '生成解释失败');
    }
}

// 显示数据的函数
function displayData(data) {
    const contentContainer = document.getElementById('contentContainer');
    
    // 创建图表和表格的容器
    const html = `
        <div class="row">
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">价格走势图</h3>
                        <canvas id="priceChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">技术指标</h3>
                        <div class="row">
                            <div class="col-md-6">
                                <canvas id="maChart"></canvas>
                            </div>
                            <div class="col-md-6">
                                <canvas id="rsiChart"></canvas>
                            </div>
                        </div>
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <canvas id="volatilityChart"></canvas>
                            </div>
                            <div class="col-md-6">
                                <canvas id="atrChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12">
                ${createDataTable(data)}
            </div>
        </div>
    `;
    
    contentContainer.innerHTML = html;
    
    // 初始化所有图表
    initPriceChart(data);
    initMAChart(data);
    initRSIChart(data);
    initVolatilityChart(data);
    initATRChart(data);
}

// 初始化价格走势图
function initPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // 准备数据
    const dates = data.map(row => row.Date);
    const prices = {
        close: data.map(row => row.Close),
        high: data.map(row => row.High),
        low: data.map(row => row.Low)
    };
    
    // 创建图表
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: '收盘价',
                    data: prices.close,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                },
                {
                    label: '最高价',
                    data: prices.high,
                    borderColor: 'rgba(255, 99, 132, 0.5)',
                    tension: 0.1,
                    fill: false
                },
                {
                    label: '最低价',
                    data: prices.low,
                    borderColor: 'rgba(54, 162, 235, 0.5)',
                    tension: 0.1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '价格走势'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: '日期'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: '价格'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

// 初始化移动平均线图表
function initMAChart(data) {
    const ctx = document.getElementById('maChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(row => row.Date),
            datasets: [
                {
                    label: 'MA10',
                    data: data.map(row => row.MA_10),
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    label: 'MA50',
                    data: data.map(row => row.MA_50),
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '移动平均线'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// 初始化RSI图表
function initRSIChart(data) {
    const ctx = document.getElementById('rsiChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(row => row.Date),
            datasets: [{
                label: 'RSI',
                data: data.map(row => row.RSI),
                borderColor: 'rgb(153, 102, 255)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'RSI指标'
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 30 || context.tick.value === 70) {
                                return 'rgba(255, 0, 0, 0.2)';
                            }
                            return 'rgba(0, 0, 0, 0.1)';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// 初始化波动率图表
function initVolatilityChart(data) {
    const ctx = document.getElementById('volatilityChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(row => row.Date),
            datasets: [{
                label: '历史波动率',
                data: data.map(row => row.Historical_Volatility),
                borderColor: 'rgb(255, 159, 64)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '历史波动率'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// 初始化ATR图表
function initATRChart(data) {
    const ctx = document.getElementById('atrChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(row => row.Date),
            datasets: [{
                label: 'ATR',
                data: data.map(row => row.ATR),
                borderColor: 'rgb(54, 162, 235)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '平均真实波幅(ATR)'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// 显示回测结果
function displayBacktestResults(results) {
    const resultsContainer = document.getElementById('backtest-results');
    if (!resultsContainer) return;
    
    if (!results || !results.metrics) {
        resultsContainer.innerHTML = '<div class="alert alert-warning">无效的回测结果</div>';
        return;
    }
    
    const metrics = results.metrics;
    const trades = results.trades || [];
    
    resultsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h4>回测指标</h4>
                <table class="table table-striped">
                    <tbody>
                        <tr>
                            <th>总交易次数</th>
                            <td>${metrics.total_trades || 0}</td>
                        </tr>
                        <tr>
                            <th>盈利交易</th>
                            <td>${metrics.profitable_trades || 0}</td>
                        </tr>
                        <tr>
                            <th>胜率</th>
                            <td>${(metrics.win_rate || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>总收益率</th>
                            <td>${(metrics.total_return || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>最大回撤</th>
                            <td>${(metrics.max_drawdown || 0).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>夏普比率</th>
                            <td>${(metrics.sharpe_ratio || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>盈亏比</th>
                            <td>${(metrics.profit_factor || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>平均盈利</th>
                            <td>${(metrics.average_profit || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>平均亏损</th>
                            <td>${(metrics.average_loss || 0).toFixed(2)}</td>
                        </tr>
                        <tr>
                            <th>风险收益比</th>
                            <td>${(metrics.risk_reward_ratio || 0).toFixed(2)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="col-md-6">
                <h4>最近交易记录</h4>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>类型</th>
                                <th>入场价</th>
                                <th>出场价</th>
                                <th>收益</th>
                                <th>收益率</th>
                                <th>持仓天数</th>
                                <th>平仓原因</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${trades.slice(-5).map(trade => `
                                <tr>
                                    <td>${trade.date || ''}</td>
                                    <td>${getTradeTypeText(trade.type)}</td>
                                    <td>${(trade.entry_price || 0).toFixed(4)}</td>
                                    <td>${(trade.exit_price || 0).toFixed(4)}</td>
                                    <td class="${(trade.profit || 0) > 0 ? 'text-success' : 'text-danger'}">${(trade.profit || 0).toFixed(2)}</td>
                                    <td class="${(trade.profit_pct || 0) > 0 ? 'text-success' : 'text-danger'}">${(trade.profit_pct || 0).toFixed(2)}%</td>
                                    <td>${trade.holding_days || 0}</td>
                                    <td>${getExitReasonText(trade.exit_reason)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// 获取交易类型文本
function getTradeTypeText(type) {
    switch (type) {
        case 'buy':
            return '<span class="badge bg-success">买入</span>';
        case 'sell':
            return '<span class="badge bg-danger">卖出</span>';
        default:
            return '<span class="badge bg-secondary">未知</span>';
    }
}

// 获取退出原因文本
function getExitReasonText(reason) {
    switch (reason) {
        case 'stop_loss':
            return '<span class="badge bg-danger">止损</span>';
        case 'take_profit':
            return '<span class="badge bg-success">止盈</span>';
        case 'signal':
            return '<span class="badge bg-primary">信号</span>';
        case 'final':
            return '<span class="badge bg-secondary">结束</span>';
        default:
            return '<span class="badge bg-secondary">未知</span>';
    }
}

// 显示解释
function displayExplanation(data) {
    const contentContainer = document.getElementById('contentContainer');
    contentContainer.innerHTML = createExplanationContent(data);
}

// 添加风险分析显示函数
function displayRiskAnalysis(data) {
    const contentContainer = document.getElementById('contentContainer');
    
    // 添加数据验证和日志
    if (!data || !data.risk_report) {
        console.error('无效的风险分析数据:', data);
        contentContainer.innerHTML = '<div class="alert alert-danger">无效的风险分析数据</div>';
        return;
    }

    console.log('风险报告数据:', data.risk_report);

    // 创建相关性热图
    const correlationMatrix = data.risk_report.correlation_matrix || {};
    const correlationHtml = createCorrelationHeatmap(correlationMatrix);
    
    // 创建风险指标表格
    const riskMetricsHtml = createRiskMetricsTable(data.risk_report.portfolio_risk);
    
    // 构建HTML
    const analysisHtml = `
        <div class="row">
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">风险分析结果</h3>
                        <div class="row">
                            <div class="col-md-6">
                                <h4>相关性矩阵</h4>
                                ${correlationHtml}
                            </div>
                            <div class="col-md-6">
                                <h4>投资组合风险指标</h4>
                                ${riskMetricsHtml}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">风险分布</h3>
                        <div class="chart-container">
                            <canvas id="riskDistributionChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">特征重要性分析</h3>
                        <div class="chart-container">
                            <canvas id="featureImportanceChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    contentContainer.innerHTML = analysisHtml;
    
    // 初始化图表（添加延迟确保DOM已更新）
    setTimeout(() => {
        try {
            const riskDistCanvas = document.getElementById('riskDistributionChart');
            if (riskDistCanvas) {
                initRiskDistributionChart(data.risk_report.risk_clusters);
            } else {
                console.error('找不到风险分布图表canvas元素');
            }

            const featureImportCanvas = document.getElementById('featureImportanceChart');
            if (featureImportCanvas) {
                initFeatureImportanceChart(data.risk_report.feature_importance);
            } else {
                console.error('找不到特征重要性图表canvas元素');
            }
        } catch (error) {
            console.error('初始化图表时出错:', error);
        }
    }, 100);
}

function initRiskDistributionChart(riskClusters) {
    const ctx = document.getElementById('riskDistributionChart');
    if (!ctx) {
        console.error('找不到风险分布图表canvas元素');
        return;
    }

    // 确保数据有效
    if (!riskClusters || typeof riskClusters !== 'object') {
        console.error('无效的风险聚类数据');
        return;
    }

    const counts = {
        '低风险': riskClusters.low_risk ? riskClusters.low_risk.length : 0,
        '中等风险': riskClusters.medium_risk ? riskClusters.medium_risk.length : 0,
        '高风险': riskClusters.high_risk ? riskClusters.high_risk.length : 0
    };

    // 销毁现有图表
    const existingChart = Chart.getChart(ctx);
    if (existingChart) {
        existingChart.destroy();
    }
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(counts),
            datasets: [{
                data: Object.values(counts),
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(255, 99, 132, 0.6)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '风险分布'
                },
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

function initFeatureImportanceChart(featureImportance) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) {
        console.error('找不到图表canvas元素');
        return;
    }

    // 确保数据有效
    if (!featureImportance || typeof featureImportance !== 'object') {
        console.warn('没有特征重要性数据');
        return;
    }

    // 计算所有货币对的平均特征重要性
    const avgImportance = {};
    let count = 0;
    
    for (const pair in featureImportance) {
        const importance = featureImportance[pair];
        for (const feature in importance) {
            if (!avgImportance[feature]) {
                avgImportance[feature] = 0;
            }
            avgImportance[feature] += importance[feature];
        }
        count++;
    }

    // 计算平均值
    if (count > 0) {
        for (const feature in avgImportance) {
            avgImportance[feature] /= count;
        }
    }

    const features = Object.keys(avgImportance);
    const importance = features.map(f => avgImportance[f]);

    // 销毁现有图表
    const existingChart = Chart.getChart(ctx);
    if (existingChart) {
        existingChart.destroy();
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: '平均特征重要性',
                data: importance,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '重要性得分'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '特征重要性分析'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

function createCorrelationHeatmap(correlationData) {
    try {
        // 销毁旧的图表实例
        if (correlationChart instanceof Chart) {
            correlationChart.destroy();
        }

        const ctx = document.getElementById('correlationChart').getContext('2d');
        const labels = Object.keys(correlationData);
        const values = [];
        
        // 构建相关性矩阵数据
        labels.forEach((row, i) => {
            labels.forEach((col, j) => {
                values.push({
                    x: col,
                    y: row,
                    v: correlationData[row][col]
                });
            });
        });

        correlationChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    data: values,
                    backgroundColor: values.map(v => {
                        const value = v.v;
                        if (value === 1) return 'rgb(255, 0, 0)';
                        return value > 0 
                            ? `rgba(255, 0, 0, ${Math.abs(value)})`
                            : `rgba(0, 0, 255, ${Math.abs(value)})`;
                    }),
                    pointRadius: 15,
                    pointStyle: 'rect'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: '货币对'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        type: 'category',
                        position: 'left',
                        title: {
                            display: true,
                            text: '货币对'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `${point.x} vs ${point.y}: ${point.v.toFixed(2)}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    } catch (error) {
        console.error('创建相关性热图时出错:', error);
        showError('创建相关性热图时出错: ' + error.message);
    }
}

function createFeatureImportanceChart(features, importance) {
    try {
        const featureCanvas = document.getElementById('featureImportanceChart');
        if (!featureCanvas) {
            throw new Error('找不到特征重要性Canvas元素');
        }

        // 销毁旧的图表实例
        if (featureImportanceChart instanceof Chart) {
            featureImportanceChart.destroy();
        }

        // 对特征重要性进行排序
        const sortedIndices = importance.map((v, i) => i)
            .sort((a, b) => importance[b] - importance[a]);
        const sortedFeatures = sortedIndices.map(i => features[i]);
        const sortedImportance = sortedIndices.map(i => importance[i]);

        const ctx = featureCanvas.getContext('2d');
        featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedFeatures,
                datasets: [{
                    label: '特征重要性',
                    data: sortedImportance,
                    backgroundColor: sortedImportance.map((v, i) => {
                        const alpha = 1 - (i / sortedImportance.length) * 0.6;
                        return `rgba(54, 162, 235, ${alpha})`;
                    })
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: '特征'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '重要性'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `重要性: ${context.raw.toFixed(4)}`;
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('创建特征重要性图表时出错:', error);
        showError('创建特征重要性图表时出错: ' + error.message);
    }
}

// 添加风险指标表格创建函数
function createRiskMetricsTable(riskMetrics) {
    // 添加默认值和null值处理
    const metrics = riskMetrics || {};
    
    // 格式化百分比
    const formatPercent = (value) => {
        if (value === null || value === undefined || isNaN(value)) {
            return '暂无数据';
        }
        // 将小数转换为百分比，并限制小数位数
        return (value * 100).toFixed(2) + '%';
    };

    // 格式化比率
    const formatRatio = (value) => {
        if (value === null || value === undefined || isNaN(value)) {
            return '暂无数据';
        }
        return value.toFixed(2);
    };

    // 格式化风险等级
    const formatRiskLevel = (level) => {
        if (!level) return '暂无数据';
        const riskLevelClass = {
            '低风险': 'text-success',
            '中风险': 'text-warning',
            '高风险': 'text-danger'
        };
        return `<span class="${riskLevelClass[level] || ''}">${level}</span>`;
    };

    // 格式化边际贡献
    const formatMarginalContributions = (contributions) => {
        if (!contributions || typeof contributions !== 'object') {
            return '暂无数据';
        }
        return Object.entries(contributions)
            .map(([pair, value]) => `${pair}: ${(value * 100).toFixed(2)}%`)
            .join('<br>');
    };

    return `
        <table class="table table-striped">
            <tbody>
                <tr>
                    <th>年化收益率</th>
                    <td>${formatPercent(metrics.annual_return)}</td>
                </tr>
                <tr>
                    <th>组合波动率</th>
                    <td>${formatPercent(metrics.portfolio_volatility)}</td>
                </tr>
                <tr>
                    <th>夏普比率</th>
                    <td>${formatRatio(metrics.sharpe_ratio)}</td>
                </tr>
                <tr>
                    <th>95% VaR</th>
                    <td>${formatPercent(metrics.var_95)}</td>
                </tr>
                <tr>
                    <th>95% CVaR</th>
                    <td>${formatPercent(metrics.cvar_95)}</td>
                </tr>
                <tr>
                    <th>风险等级</th>
                    <td>${formatRiskLevel(metrics.risk_level)}</td>
                </tr>
                <tr>
                    <th>边际风险贡献</th>
                    <td>${formatMarginalContributions(metrics.marginal_contributions)}</td>
                </tr>
            </tbody>
        </table>
    `;
}

// 创建数据表格的函数
function createDataTable(data) {
    return `
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">外汇数据</h3>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>开盘价</th>
                                <th>最高价</th>
                                <th>最低价</th>
                                <th>收盘价</th>
                                <th>成交量</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.map(row => `
                                <tr>
                                    <td>${row.Date}</td>
                                    <td>${row.Open}</td>
                                    <td>${row.High}</td>
                                    <td>${row.Low}</td>
                                    <td>${row.Close}</td>
                                    <td>${row.Volume}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// 创建信号表格的函数
function createSignalsTable(data) {
    return `
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">交易信号</h3>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>收盘价</th>
                                <th>MA10</th>
                                <th>MA50</th>
                                <th>RSI</th>
                                <th>信号</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.map(row => `
                                <tr>
                                    <td>${row.Date}</td>
                                    <td>${row.Close}</td>
                                    <td>${row.MA_10}</td>
                                    <td>${row.MA_50}</td>
                                    <td>${row.RSI}</td>
                                    <td>${getSignalText(row.Signal)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// 创建回测结果表格的函数
function createBacktestTable(results) {
    return `
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">回测结果</h3>
                <table class="table table-striped">
                    <tbody>
                        <tr>
                            <th>总交易次数</th>
                            <td>${results.total_trades}</td>
                        </tr>
                        <tr>
                            <th>盈利交易</th>
                            <td>${results.winning_trades}</td>
                        </tr>
                        <tr>
                            <th>亏损交易</th>
                            <td>${results.losing_trades}</td>
                        </tr>
                        <tr>
                            <th>胜率</th>
                            <td>${results.win_rate}</td>
                        </tr>
                        <tr>
                            <th>盈亏比</th>
                            <td>${results.profit_factor}</td>
                        </tr>
                        <tr>
                            <th>总收益率</th>
                            <td>${results.total_return}</td>
                        </tr>
                        <tr>
                            <th>夏普比率</th>
                            <td>${results.sharpe_ratio}</td>
                        </tr>
                        <tr>
                            <th>最大回撤</th>
                            <td>${results.max_drawdown}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// 创建解释内容的函数
function createExplanationContent(explanation) {
    // 检查explanation是否为对象
    if (typeof explanation === 'object' && explanation !== null) {
        const { analysis_text, trades } = explanation;
        
        // 创建交易记录表格
        let tradesHtml = '';
        if (trades && trades.length > 0) {
            tradesHtml = `
                <div class="mt-4">
                    <h4>最近交易记录</h4>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>日期</th>
                                    <th>类型</th>
                                    <th>入场价</th>
                                    <th>出场价</th>
                                    <th>收益</th>
                                    <th>收益率</th>
                                    <th>持仓天数</th>
                                    <th>平仓原因</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${trades.slice(-5).map(trade => `
                                    <tr>
                                        <td>${trade.date || ''}</td>
                                        <td>${getTradeTypeText(trade.type)}</td>
                                        <td>${(trade.entry_price || 0).toFixed(4)}</td>
                                        <td>${(trade.exit_price || 0).toFixed(4)}</td>
                                        <td class="${(trade.profit || 0) > 0 ? 'text-success' : 'text-danger'}">
                                            ${(trade.profit || 0).toFixed(2)}
                                        </td>
                                        <td class="${(trade.profit_pct || 0) > 0 ? 'text-success' : 'text-danger'}">
                                            ${(trade.profit_pct || 0).toFixed(2)}%
                                        </td>
                                        <td>${trade.holding_days || 0}</td>
                                        <td>${getExitReasonText(trade.exit_reason)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }
        
        // 返回完整的解释内容
        return `
            <div class="card">
                <div class="card-body">
                    <h3 class="card-title">信号解释</h3>
                    <div class="explanation-text" style="white-space: pre-wrap; font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6;">
                        ${analysis_text || ''}
                    </div>
                    ${tradesHtml}
                </div>
            </div>
        `;
    } else {
        // 如果explanation是字符串，保持原有的显示方式
        return `
            <div class="card">
                <div class="card-body">
                    <h3 class="card-title">信号解释</h3>
                    <div class="explanation-text" style="white-space: pre-wrap; font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6;">
                        ${explanation}
                    </div>
                </div>
            </div>
        `;
    }
}

// 辅助函数：获取信号文本
function getSignalText(signal) {
    switch(signal) {
        case 1:
            return '<span class="badge bg-success">买入</span>';
        case -1:
            return '<span class="badge bg-danger">卖出</span>';
        default:
            return '<span class="badge bg-secondary">持有</span>';
    }
}

// 显示错误信息
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    errorDiv.style.display = 'block';
}

// 显示成功信息
function showSuccess(message) {
    const successDiv = document.getElementById('success');
    successDiv.innerHTML = `
        <div class="alert alert-success alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    successDiv.style.display = 'block';
}

// 添加回测优化函数
async function optimizeBacktest(currencyPair, method) {
    console.log('开始优化回测:', currencyPair, method);
    
    try {
        const response = await fetch('/optimize_backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                currency_pair: currencyPair,
                method: method
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || '优化回测失败');
        }
        
        if (result.success) {
            displayOptimizationResults(result);
            showSuccess('回测优化完成');
        } else {
            throw new Error(result.error || '优化回测失败');
        }
        
    } catch (error) {
        console.error('优化回测时出错:', error);
        showError(error.message);
    }
}

function displayOptimizationResults(data) {
    const contentContainer = document.getElementById('contentContainer');
    
    // 创建优化结果展示
    const resultsHtml = `
        <div class="card mb-4">
            <div class="card-body">
                <h3 class="card-title">优化结果 (${data.optimization_results.method})</h3>
                <div class="row">
                    <div class="col-md-6">
                        <h4>最佳参数</h4>
                        <table class="table table-striped">
                            <tbody>
                                ${Object.entries(data.optimization_results.best_params || {}).map(([key, value]) => `
                                    <tr>
                                        <th>${key}</th>
                                        <td>${value}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                        <div class="mt-3">
                            <strong>最佳得分：</strong> ${(data.optimization_results.best_score * 100).toFixed(2)}%
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h4>优化历史</h4>
                        <img src="${data.visualization_paths.history}" class="img-fluid" alt="优化历史">
                    </div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">参数相关性分析</h3>
                <img src="${data.visualization_paths.heatmap}" class="img-fluid" alt="参数相关性">
            </div>
        </div>
    `;
    
    contentContainer.innerHTML = resultsHtml;
}

// 运行优化回测
async function runOptimizedBacktest(currencyPair, method) {
    console.log(`开始运行优化回测: ${currencyPair} ${method}`);
    
    // 清空之前的所有内容
    const contentContainer = document.getElementById('contentContainer');
    if (!contentContainer) {
        console.error('找不到contentContainer元素');
        return;
    }
    contentContainer.innerHTML = '';
    
    // 创建结果容器
    let resultsContainer = document.createElement('div');
    resultsContainer.id = 'backtest-results';
    contentContainer.appendChild(resultsContainer);
    
    // 显示加载信息
    resultsContainer.innerHTML = '<div class="alert alert-info">正在执行回测，请稍候...</div>';
    
    try {
        // 显示加载动画
        showLoadingAnimation();
        
        // 如果是普通回测（无优化）
        if (method === 'none') {
            const response = await fetch('/run_backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    currency_pair: currencyPair,
                    params: {
                        stop_loss: 0.02,
                        take_profit: 0.03,
                        position_size: 1000,
                        initial_capital: 100000,
                        commission: 0.0002,
                        slippage: 0.0001
                    }
                })
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || '回测执行失败');
            }
            
            // 隐藏加载动画
            hideLoadingAnimation();
            
            // 显示普通回测结果
            if (result.success) {
                displayBacktestResults(result.data);
                showSuccess('回测完成');
            } else {
                throw new Error(result.error || '回测执行失败');
            }
            return;
        }
        
        // 如果是优化回测
        const requestData = {
            currency_pair: currencyPair,
            optimization_method: method,
            params: {
                stop_loss_range: [0.01, 0.015, 0.02, 0.025, 0.03],
                take_profit_range: [0.02, 0.025, 0.03, 0.035, 0.04],
                position_size_range: [1000, 2000, 3000, 4000, 5000],
                initial_capital: 100000,
                commission: 0.0002,
                slippage: 0.0001
            }
        };
        
        // 根据优化方法添加特定参数
        if (method === 'random_search') {
            requestData.params.n_iterations = 20;
        } else if (method === 'bayesian_optimization') {
            requestData.params.n_iterations = 15;
        } else if (method === 'genetic_algorithm') {
            requestData.params.population_size = 20;
            requestData.params.n_generations = 10;
        }
        
        // 发送优化回测请求
        const response = await fetch('/run_optimized_backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        // 检查响应状态
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 解析响应数据
        const result = await response.json();
        
        // 隐藏加载动画
        hideLoadingAnimation();
        
        // 显示结果
        if (result.success) {
            displayOptimizedBacktestResults(result.data);
            showSuccess('优化回测完成');
        } else {
            throw new Error(result.error || '优化回测失败');
        }
        
    } catch (error) {
        console.error('运行优化回测时出错:', error);
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <h5>运行回测时出错:</h5>
                    <p>${error.message}</p>
                </div>
            `;
        }
        hideLoadingAnimation();
    }
}

// 显示加载动画
function showLoadingAnimation() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-animation';
    loadingDiv.className = 'text-center mt-4';
    loadingDiv.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">加载中...</span>
        </div>
        <p class="mt-2">正在执行优化回测，请稍候...</p>
    `;
    document.getElementById('backtest-results').appendChild(loadingDiv);
}

// 隐藏加载动画
function hideLoadingAnimation() {
    const loadingDiv = document.getElementById('loading-animation');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// 获取优化方法的中文名称
function getOptimizationMethodName(method) {
    const methodNames = {
        grid_search: '网格搜索',
        random_search: '随机搜索',
        bayesian_optimization: '贝叶斯优化',
        genetic_algorithm: '遗传算法'
    };
    return methodNames[method] || method;
}

// 显示优化回测结果
function displayOptimizedBacktestResults(results) {
    const resultsContainer = document.getElementById('backtest-results');
    if (!resultsContainer) {
        console.error('找不到结果容器');
        return;
    }
    
    resultsContainer.innerHTML = '';  // 清空之前的结果
    
    // 检查结果是否有效
    if (!results || !results.metrics) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <h5>优化回测失败</h5>
                <p>${results?.error || '无效的回测结果'}</p>
            </div>`;
        return;
    }
    
    const { metrics, trades, optimization_results } = results;
    
    // 创建结果容器
    const resultDiv = document.createElement('div');
    resultDiv.className = 'optimization-results mt-4';
    
    // 1. 显示最优参数
    if (optimization_results?.best_params) {
        const paramsDiv = document.createElement('div');
        paramsDiv.className = 'card mb-4';
        paramsDiv.innerHTML = `
            <div class="card-header bg-primary text-white">
                <h5 class="card-title mb-0">最优参数</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <p><strong>止损:</strong> ${(optimization_results.best_params.stop_loss * 100).toFixed(2)}%</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>止盈:</strong> ${(optimization_results.best_params.take_profit * 100).toFixed(2)}%</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>仓位大小:</strong> ${optimization_results.best_params.position_size}</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>初始资金:</strong> ${optimization_results.best_params.initial_capital}</p>
                    </div>
                </div>
            </div>
        `;
        resultDiv.appendChild(paramsDiv);
    }
    
    // 2. 显示回测指标
    if (metrics) {
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'card mb-4';
        metricsDiv.innerHTML = `
            <div class="card-header bg-success text-white">
                <h5 class="card-title mb-0">回测指标</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <p><strong>总收益率:</strong> ${(metrics.total_return || 0).toFixed(2)}%</p>
                        <p><strong>最大回撤:</strong> ${(metrics.max_drawdown || 0).toFixed(2)}%</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>胜率:</strong> ${(metrics.win_rate || 0).toFixed(2)}%</p>
                        <p><strong>总交易次数:</strong> ${metrics.total_trades || 0}</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>平均收益:</strong> ${(metrics.average_profit || 0).toFixed(2)}</p>
                        <p><strong>平均损失:</strong> ${(metrics.average_loss || 0).toFixed(2)}</p>
                    </div>
                    <div class="col-md-3">
                        <p><strong>夏普比率:</strong> ${(metrics.sharpe_ratio || 0).toFixed(2)}</p>
                        <p><strong>盈亏比:</strong> ${(metrics.profit_factor || 0).toFixed(2)}</p>
                    </div>
                </div>
            </div>
        `;
        resultDiv.appendChild(metricsDiv);
    }
    
    // 3. 显示最近交易记录
    if (trades && trades.length > 0) {
        const tradesDiv = document.createElement('div');
        tradesDiv.className = 'card mb-4';
        tradesDiv.innerHTML = `
            <div class="card-header bg-warning">
                <h5 class="card-title mb-0">最近交易记录</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>类型</th>
                                <th>入场价</th>
                                <th>出场价</th>
                                <th>收益</th>
                                <th>收益率</th>
                                <th>持仓天数</th>
                                <th>平仓原因</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${trades.slice(-5).map(trade => `
                                <tr>
                                    <td>${trade.date}</td>
                                    <td>${trade.type === 'buy' ? '买入' : '卖出'}</td>
                                    <td>${(trade.entry_price || 0).toFixed(4)}</td>
                                    <td>${(trade.exit_price || 0).toFixed(4)}</td>
                                    <td class="${trade.profit > 0 ? 'text-success' : 'text-danger'}">${(trade.profit || 0).toFixed(2)}</td>
                                    <td class="${trade.profit_pct > 0 ? 'text-success' : 'text-danger'}">${(trade.profit_pct || 0).toFixed(2)}%</td>
                                    <td>${trade.holding_days || 0}</td>
                                    <td>${getExitReasonText(trade.exit_reason)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        resultDiv.appendChild(tradesDiv);
    }
    
    resultsContainer.appendChild(resultDiv);
}

// 更新获取退出原因文本的函数
function getExitReasonText(reason) {
    const reasonMap = {
        'stop_loss': '<span class="badge bg-danger">止损</span>',
        'take_profit': '<span class="badge bg-success">止盈</span>',
        'signal_reverse': '<span class="badge bg-warning">信号反转</span>',
        'time_limit': '<span class="badge bg-info">时间限制</span>',
        'final': '<span class="badge bg-secondary">结束</span>'
    };
    return reasonMap[reason] || '<span class="badge bg-secondary">未知</span>';
}

// 创建优化过程进度图表
function createOptimizationProgressChart(history) {
    const ctx = document.getElementById('optimizationProgressChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: history.length}, (_, i) => i + 1),
            datasets: [{
                label: '最佳收益率',
                data: history.map(h => h.best_return * 100),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: '收益率 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '迭代次数'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '优化进度'
                }
            }
        }
    });
}

// 创建参数分布图表
function createParameterDistributionChart(distribution) {
    const ctx = document.getElementById('parameterDistributionChart').getContext('2d');
    
    const parameters = Object.keys(distribution);
    const datasets = parameters.map((param, index) => ({
        label: getParameterName(param),
        data: distribution[param],
        borderColor: getParameterColor(index),
        backgroundColor: getParameterColor(index, 0.2),
        type: 'scatter'
    }));
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: '参数值'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '迭代次数'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '参数分布'
                }
            }
        }
    });
}

// 获取参数中文名称
function getParameterName(param) {
    const paramNames = {
        stop_loss: '止损',
        take_profit: '止盈',
        position_size: '仓位大小'
    };
    return paramNames[param] || param;
}

// 获取参数颜色
function getParameterColor(index, alpha = 1) {
    const colors = [
        `rgba(75, 192, 192, ${alpha})`,
        `rgba(255, 99, 132, ${alpha})`,
        `rgba(255, 206, 86, ${alpha})`,
        `rgba(54, 162, 235, ${alpha})`
    ];
    return colors[index % colors.length];
}

function createRiskDistributionChart(riskDistribution) {
    try {
        const riskCanvas = document.getElementById('riskDistributionChart');
        if (!riskCanvas) {
            throw new Error('找不到风险分布Canvas元素');
        }

        // 销毁旧的图表实例
        if (riskDistributionChart instanceof Chart) {
            riskDistributionChart.destroy();
        }

        const ctx = riskCanvas.getContext('2d');
        
        // 处理风险分布数据
        const labels = ['低风险', '中等风险', '高风险'];
        const data = [
            parseFloat(riskDistribution.low_risk) || 0,
            parseFloat(riskDistribution.medium_risk) || 0,
            parseFloat(riskDistribution.high_risk) || 0
        ];

        // 确保数据总和为100%
        const total = data.reduce((a, b) => a + b, 0);
        const normalizedData = data.map(v => (v / total) * 100);

        riskDistributionChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: normalizedData,
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.7)',  // 低风险 - 绿色
                        'rgba(255, 206, 86, 0.7)',  // 中等风险 - 黄色
                        'rgba(255, 99, 132, 0.7)'   // 高风险 - 红色
                    ],
                    borderColor: [
                        'rgb(75, 192, 192)',
                        'rgb(255, 206, 86)',
                        'rgb(255, 99, 132)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                return `${label}: ${value.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('创建风险分布图时出错:', error);
        showError('创建风险分布图时出错: ' + error.message);
    }
}

// 添加显示风险指标的函数
function displayRiskMetrics(metrics) {
    try {
        // 格式化百分比
        const formatPercent = (value) => {
            if (value === null || value === undefined || isNaN(value)) {
                return '暂无数据';
            }
            return (value).toFixed(2) + '%';
        };

        // 格式化比率
        const formatRatio = (value) => {
            if (value === null || value === undefined || isNaN(value)) {
                return '暂无数据';
            }
            return value.toFixed(2);
        };

        // 格式化风险等级
        const formatRiskLevel = (level) => {
            if (!level) return '暂无数据';
            const riskLevelClass = {
                '低风险': 'text-success',
                '中风险': 'text-warning',
                '高风险': 'text-danger'
            };
            return `<span class="${riskLevelClass[level] || ''}">${level}</span>`;
        };

        // 格式化边际贡献
        const formatMarginalContributions = (contributions) => {
            if (!contributions || typeof contributions !== 'object') {
                return '暂无数据';
            }
            return Object.entries(contributions)
                .map(([pair, value]) => `${pair}: ${(value * 100).toFixed(2)}%`)
                .join('<br>');
        };

        // 创建风险指标表格
        const riskMetricsHtml = `
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">风险指标</h3>
                        <table class="table table-striped">
                            <tbody>
                                <tr>
                                    <th>年化收益率</th>
                                    <td>${formatPercent(metrics.annual_return)}</td>
                                </tr>
                                <tr>
                                    <th>组合波动率</th>
                                    <td>${formatPercent(metrics.portfolio_volatility)}</td>
                                </tr>
                                <tr>
                                    <th>夏普比率</th>
                                    <td>${formatRatio(metrics.sharpe_ratio)}</td>
                                </tr>
                                <tr>
                                    <th>95% VaR</th>
                                    <td>${formatPercent(metrics.var_95)}</td>
                                </tr>
                                <tr>
                                    <th>95% CVaR</th>
                                    <td>${formatPercent(metrics.cvar_95)}</td>
                                </tr>
                                <tr>
                                    <th>风险等级</th>
                                    <td>${formatRiskLevel(metrics.risk_level)}</td>
                                </tr>
                                <tr>
                                    <th>边际风险贡献</th>
                                    <td>${formatMarginalContributions(metrics.marginal_contributions)}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        // 将风险指标表格添加到内容容器中
        const contentContainer = document.querySelector('#contentContainer .row');
        contentContainer.insertAdjacentHTML('beforeend', riskMetricsHtml);
    } catch (error) {
        console.error('显示风险指标时出错:', error);
        showError('显示风险指标时出错: ' + error.message);
    }
}

// 初始化相关性热力图
function initializeCorrelationHeatmap(data) {
    const ctx = document.getElementById('correlationHeatmap');
    if (!ctx) return;
    
    // 计算相关性矩阵
    const features = ['Close', 'Volume', 'MA_10', 'MA_50', 'RSI', 'Signal', 'Historical_Volatility'];
    const correlationMatrix = calculateCorrelationMatrix(data, features);
    const labels = features.map(f => getFeatureDisplayName(f));
    
    // 创建热力图数据
    const matrixData = [];
    for (let i = 0; i < correlationMatrix.length; i++) {
        for (let j = 0; j < correlationMatrix[i].length; j++) {
            matrixData.push({
                x: j,
                y: i,
                v: correlationMatrix[i][j]
            });
        }
    }

    // 创建热力图
    new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: '相关性',
                data: matrixData,
                width: ({ chart }) => (chart.chartArea || {}).width / features.length - 1,
                height: ({ chart }) => (chart.chartArea || {}).height / features.length - 1,
                backgroundColor: ({ dataset, dataIndex }) => {
                    const value = dataset.data[dataIndex].v;
                    return getCorrelationColor(value);
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    offset: true,
                    min: -0.5,
                    max: features.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: (value) => labels[value] || ''
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    offset: true,
                    min: -0.5,
                    max: features.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: (value) => labels[value] || ''
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: () => '',
                        label: (context) => {
                            const v = context.dataset.data[context.dataIndex].v;
                            const i = context.dataset.data[context.dataIndex].y;
                            const j = context.dataset.data[context.dataIndex].x;
                            return `${labels[i]} 与 ${labels[j]} 的相关系数: ${v.toFixed(2)}`;
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: '指标相关性矩阵',
                    font: {
                        size: 16
                    }
                }
            }
        }
    });

    // 添加颜色图例
    const legendContainer = document.createElement('div');
    legendContainer.className = 'correlation-legend mt-3';
    legendContainer.innerHTML = `
        <div class="correlation-scale">
            <div class="d-flex justify-content-between mb-2">
                <span class="legend-label">强负相关</span>
                <span class="legend-label">无相关</span>
                <span class="legend-label">强正相关</span>
            </div>
            <div class="correlation-gradient"></div>
            <div class="d-flex justify-content-between mt-2">
                <span class="legend-value">-1.0</span>
                <span class="legend-value">0.0</span>
                <span class="legend-value">1.0</span>
            </div>
        </div>
    `;
    
    ctx.parentNode.appendChild(legendContainer);
}

// 获取相关性颜色
function getCorrelationColor(value) {
    // 使用更细腻的颜色渐变
    const maxColor = [220, 20, 60];  // 正相关最大值颜色（深红色）
    const minColor = [30, 144, 255]; // 负相关最大值颜色（深蓝色）
    const midColor = [255, 255, 255]; // 无相关颜色（白色）
    
    let resultColor;
    if (value > 0) {
        // 正相关：从白色渐变到红色
        const factor = value;
        resultColor = midColor.map((mid, i) => 
            Math.round(mid + (maxColor[i] - mid) * factor)
        );
    } else {
        // 负相关：从白色渐变到蓝色
        const factor = -value;
        resultColor = midColor.map((mid, i) => 
            Math.round(mid + (minColor[i] - mid) * factor)
        );
    }
    
    return `rgba(${resultColor[0]}, ${resultColor[1]}, ${resultColor[2]}, 0.8)`;
}

// 计算相关性矩阵
function calculateCorrelationMatrix(data, features) {
    const matrix = [];
    
    // 提取数值数据
    const featureData = features.map(feature => 
        data.map(d => parseFloat(d[feature]) || 0)
    );
    
    // 计算每对特征之间的相关性
    for (let i = 0; i < features.length; i++) {
        matrix[i] = [];
        for (let j = 0; j < features.length; j++) {
            matrix[i][j] = calculateCorrelation(featureData[i], featureData[j]);
        }
    }
    
    return matrix;
}

// 计算两个数组之间的相关系数
function calculateCorrelation(array1, array2) {
    const n = array1.length;
    if (n !== array2.length || n === 0) return 0;
    
    const mean1 = array1.reduce((a, b) => a + b, 0) / n;
    const mean2 = array2.reduce((a, b) => a + b, 0) / n;
    
    const variance1 = array1.reduce((a, b) => a + Math.pow(b - mean1, 2), 0);
    const variance2 = array2.reduce((a, b) => a + Math.pow(b - mean2, 2), 0);
    
    if (variance1 === 0 || variance2 === 0) return 0;
    
    const covariance = array1.reduce((a, b, i) => 
        a + (b - mean1) * (array2[i] - mean2), 0
    );
    
    return covariance / Math.sqrt(variance1 * variance2);
}

// 获取特征显示名称
function getFeatureDisplayName(feature) {
    const nameMap = {
        'Close': '收盘价',
        'Volume': '成交量',
        'MA_10': '10日均线',
        'MA_50': '50日均线',
        'RSI': 'RSI指标',
        'Signal': '交易信号',
        'Historical_Volatility': '历史波动率'
    };
    return nameMap[feature] || feature;
}