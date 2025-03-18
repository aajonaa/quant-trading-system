document.addEventListener('DOMContentLoaded', function() {
    // 初始化日期选择器
    flatpickr(".datepicker", {
        dateFormat: "Y-m-d",
        locale: "zh",
        defaultDate: new Date()
    });

    // 初始化参数滑块值显示
    document.querySelectorAll('.form-range').forEach(function(range) {
        const valueDisplay = document.getElementById(range.id + 'Value');
        if (valueDisplay) {
            range.addEventListener('input', function() {
                valueDisplay.textContent = this.value + '%';
            });
        }
    });

    // 设置导航菜单切换
    document.querySelectorAll('[data-page]').forEach(function(element) {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            const targetPage = this.getAttribute('data-page');
            showPage(targetPage);

            // 更新导航栏活动状态
            document.querySelectorAll('.nav-link').forEach(function(navLink) {
                navLink.classList.remove('active');
                if (navLink.getAttribute('data-page') === targetPage) {
                    navLink.classList.add('active');
                }
            });
        });
    });

    // 设置表单提交事件
    document.getElementById('backtestForm').addEventListener('submit', function(e) {
        e.preventDefault();
        runBacktest();
    });

    document.getElementById('riskForm').addEventListener('submit', function(e) {
        e.preventDefault();
        runRiskAnalysis();
    });

    document.getElementById('optimizeForm').addEventListener('submit', function(e) {
        e.preventDefault();
        runOptimization();
    });

    // 设置登录和注册按钮事件
    document.getElementById('loginButton').addEventListener('click', function() {
        handleLogin();
    });

    document.getElementById('registerButton').addEventListener('click', function() {
        handleRegister();
    });

    // 设置客服聊天功能
    document.getElementById('csButton').addEventListener('click', function() {
        document.getElementById('csPanel').style.display = 'flex';
    });

    document.getElementById('csClose').addEventListener('click', function() {
        document.getElementById('csPanel').style.display = 'none';
    });

    document.getElementById('csSend').addEventListener('click', function() {
        sendChatMessage();
    });

    document.getElementById('csInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendChatMessage();
        }
    });
});

// 显示指定页面
function showPage(pageName) {
    // 隐藏所有页面
    document.querySelectorAll('.page-content').forEach(function(page) {
        page.style.display = 'none';
    });

    // 显示目标页面
    const targetPage = document.getElementById(pageName + 'Page');
    if (targetPage) {
        targetPage.style.display = 'block';
    }
}

// 运行回测
function runBacktest() {
    const currency = document.getElementById('currency').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    if (!currency || !startDate || !endDate) {
        alert('请填写完整的回测参数');
        return;
    }

    // 显示加载动画
    document.getElementById('loading').style.display = 'flex';

    // 准备表单数据
    const formData = new FormData();
    formData.append('currency', currency);
    formData.append('start_date', startDate);
    formData.append('end_date', endDate);

    // 发送请求
    fetch('/api/backtest', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';

        if (data.error) {
            alert('回测失败: ' + data.error);
            return;
        }

        // 显示回测结果
        displayBacktestResults(data);
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        alert('请求失败: ' + error);
    });
}

// 显示回测结果
function displayBacktestResults(data) {
    const resultsContainer = document.getElementById('backtestResults');
    resultsContainer.style.display = 'block';
    resultsContainer.innerHTML = '';

    // 创建权益曲线图
    const equityCurveCard = document.createElement('div');
    equityCurveCard.className = 'chart-container';
    equityCurveCard.innerHTML = `
        <h3 class="chart-title">${data.currency} 权益曲线</h3>
        <canvas id="equityCurveChart"></canvas>
    `;
    resultsContainer.appendChild(equityCurveCard);

    // 创建回测指标卡片
    const metricsCard = document.createElement('div');
    metricsCard.className = 'card mb-4';
    metricsCard.innerHTML = `
        <div class="card-body">
            <h3 class="card-title">回测指标</h3>
            <div class="row">
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">总收益率</div>
                        <div class="metric-value ${data.metrics.total_return >= 0 ? 'positive' : 'negative'}">
                            ${(data.metrics.total_return * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">年化收益率</div>
                        <div class="metric-value ${data.metrics.annual_return >= 0 ? 'positive' : 'negative'}">
                            ${(data.metrics.annual_return * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">最大回撤</div>
                        <div class="metric-value negative">
                            ${(data.metrics.max_drawdown * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">夏普比率</div>
                        <div class="metric-value ${data.metrics.sharpe_ratio >= 0 ? 'positive' : 'negative'}">
                            ${data.metrics.sharpe_ratio.toFixed(2)}
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">胜率</div>
                        <div class="metric-value">
                            ${(data.metrics.win_rate * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">盈亏比</div>
                        <div class="metric-value">
                            ${data.metrics.profit_factor.toFixed(2)}
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">平均收益</div>
                        <div class="metric-value ${data.metrics.avg_trade >= 0 ? 'positive' : 'negative'}">
                            ${(data.metrics.avg_trade * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-title">最大连续亏损</div>
                        <div class="metric-value">
                            ${data.metrics.max_consecutive_losses}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    resultsContainer.appendChild(metricsCard);

    // 创建交易信号图
    const signalChartCard = document.createElement('div');
    signalChartCard.className = 'chart-container';
    signalChartCard.innerHTML = `
        <h3 class="chart-title">${data.currency} 交易信号</h3>
        <canvas id="signalChart"></canvas>
    `;
    resultsContainer.appendChild(signalChartCard);

    // 初始化权益曲线图
    initEquityCurveChart(data.equity_curve);

    // 初始化交易信号图
    initSignalChart(data.price_data, data.signals);
}

// 初始化权益曲线图
function initEquityCurveChart(equityCurve) {
    const ctx = document.getElementById('equityCurveChart').getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityCurve.dates,
            datasets: [{
                label: '账户权益',
                data: equityCurve.values,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        displayFormats: {
                            month: 'yyyy-MM'
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
                        text: '账户权益'
                    }
                }
            }
        }
    });
}

// 初始化交易信号图
function initSignalChart(priceData, signals) {
    const ctx = document.getElementById('signalChart').getContext('2d');

    // 准备买入和卖出信号数据
    const buySignals = [];
    const sellSignals = [];

    for (let i = 0; i < signals.length; i++) {
        if (signals[i] === 1) {
            buySignals.push({
                x: priceData.dates[i],
                y: priceData.prices[i]
            });
            sellSignals.push(null);
        } else if (signals[i] === -1) {
            sellSignals.push({
                x: priceData.dates[i],
                y: priceData.prices[i]
            });
            buySignals.push(null);
        } else {
            buySignals.push(null);
            sellSignals.push(null);
        }
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: priceData.dates,
            datasets: [
                {
                    label: '价格',
                    data: priceData.prices,
                    borderColor: '#7f8c8d',
                    backgroundColor: 'rgba(127, 140, 141, 0.1)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: '买入信号',
                    data: buySignals,
                    borderColor: '#2ecc71',
                    backgroundColor: '#2ecc71',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false
                },
                {
                    label: '卖出信号',
                    data: sellSignals,
                    borderColor: '#e74c3c',
                    backgroundColor: '#e74c3c',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        displayFormats: {
                            month: 'yyyy-MM'
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
            }
        }
    });
}

// 运行风险分析
function runRiskAnalysis() {
    // 获取选中的货币对
    const selectedCurrencies = [];
    document.querySelectorAll('input[name="currencies"]:checked').forEach(function(checkbox) {
        selectedCurrencies.push(checkbox.value);
    });

    if (selectedCurrencies.length < 2) {
        alert('请至少选择两个货币对进行分析');
        return;
    }

    const analysisType = document.getElementById('analysisType').value;

    // 显示加载动画
    document.getElementById('loading').style.display = 'flex';

    // 准备表单数据
    const formData = new FormData();
    formData.append('currencies', JSON.stringify(selectedCurrencies));
    formData.append('analysis_type', analysisType);

    // 发送请求
    fetch('/api/risk_analysis', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';

        if (data.error) {
            alert('风险分析失败: ' + data.error);
            return;
        }

        // 显示风险分析结果
        displayRiskResults(data);
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        alert('请求失败: ' + error);
    });
}

// 显示风险分析结果
function displayRiskResults(data) {
    const resultsContainer = document.getElementById('riskResults');
    resultsContainer.style.display = 'block';
    resultsContainer.innerHTML = '';

    // 创建相关性热图
    const correlationCard = document.createElement('div');
    correlationCard.className = 'chart-container';
    correlationCard.innerHTML = `
        <h3 class="chart-title">货币对相关性热图</h3>
        <canvas id="correlationHeatmap"></canvas>
    `;
    resultsContainer.appendChild(correlationCard);

    // 创建风险指标表格
    const riskTableCard = document.createElement('div');
    riskTableCard.className = 'table-container';
    riskTableCard.innerHTML = `
        <h3 class="chart-title">货币对组合风险分析</h3>
        <div class="table-responsive">
            <table class="table">
                <thead>
                    <tr>
                        <th>货币对组合</th>
                        <th>相关系数</th>
                        <th>组合波动率</th>
                        <th>风险得分</th>
                        <th>风险等级</th>
                        <th>交易建议</th>
                    </tr>
                </thead>
                <tbody id="riskTableBody">
                </tbody>
            </table>
        </div>
    `;
    resultsContainer.appendChild(riskTableCard);

    // 填充风险表格数据
    const tableBody = document.getElementById('riskTableBody');
    data.pair_risks.forEach(risk => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${risk.pair}</td>
            <td>${risk.correlation.toFixed(4)}</td>
            <td>${risk.volatility.toFixed(4)}</td>
            <td>${risk.risk_score.toFixed(2)}</td>
            <td>${risk.risk_level}</td>
            <td>${risk.recommendation}</td>
        `;
        tableBody.appendChild(row);
    });

    // 初始化相关性热图
    initCorrelationHeatmap(data.correlation_matrix);
}

// 初始化相关性热图
function initCorrelationHeatmap(correlationMatrix) {
    const ctx = document.getElementById('correlationHeatmap').getContext('2d');

    // 准备热图数据
    const labels = Object.keys(correlationMatrix);
    const data = [];

    for (let i = 0; i < labels.length; i++) {
        for (let j = 0; j < labels.length; j++) {
            data.push({
                x: j,
                y: i,
                v: correlationMatrix[labels[i]][labels[j]]
            });
        }
    }

    new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: '相关系数',
                data: data,
                backgroundColor(context) {
                    const value = context.dataset.data[context.dataIndex].v;

                    if (value >= 0.7) return 'rgba(231, 76, 60, 0.8)';  // 高正相关 - 红色
                    if (value >= 0.3) return 'rgba(230, 126, 34, 0.8)';  // 中正相关 - 橙色
                    if (value >= 0) return 'rgba(241, 196, 15, 0.8)';    // 低正相关 - 黄色
                    if (value >= -0.3) return 'rgba(46, 204, 113, 0.8)';  // 低负相关 - 绿色
                    if (value >= -0.7) return 'rgba(52, 152, 219, 0.8)';  // 中负相关 - 蓝色
                    return 'rgba(155, 89, 182, 0.8)';                    // 高负相关 - 紫色
                },
                borderWidth: 1,
                borderColor: 'rgba(0, 0, 0, 0.1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        title() {
                            return '';
                        },
                        label(context) {
                            const v = context.dataset.data[context.dataIndex];
                            return [
                                `${labels[v.y]} - ${labels[v.x]}`,
                                `相关系数: ${v.v.toFixed(4)}`
                            ];
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: labels,
                    ticks: {
                        display: true
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'category',
                    labels: labels,
                    offset: true,
                    ticks: {
                        display: true
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// 运行策略优化
function runOptimization() {
    const currency = document.getElementById('optimizeCurrency').value;
    const method = document.getElementById('optimizeMethod').value;
    const target = document.getElementById('optimizeTarget').value;
    const stopLoss = document.getElementById('stopLoss').value;
    const takeProfit = document.getElementById('takeProfit').value;
    const positionSize = document.getElementById('positionSize').value;
    const trailingStop = document.getElementById('trailingStop').value;

    if (!currency) {
        alert('请选择货币对');
        return;
    }

    // 显示加载动画
    document.getElementById('loading').style.display = 'flex';

    // 准备表单数据
    const formData = new FormData();
    formData.append('currency', currency);
    formData.append('method', method);
    formData.append('target', target);
    formData.append('stop_loss', stopLoss / 100);
    formData.append('take_profit', takeProfit / 100);
    formData.append('position_size', positionSize / 100);
    formData.append('trailing_stop', trailingStop / 100);

    // 发送请求
    fetch('/api/optimize', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';

        if (data.error) {
            alert('策略优化失败: ' + data.error);
            return;
        }

        // 显示优化结果
        displayOptimizationResults(data);
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        alert('请求失败: ' + error);
    });
}

// 显示优化结果
function displayOptimizationResults(data) {
    const resultsContainer = document.getElementById('optimizeResults');
    resultsContainer.style.display = 'block';
    resultsContainer.innerHTML = '';

    // 创建优化结果卡片
    const optimizationCard = document.createElement('div');
    optimizationCard.className = 'card mb-4';
    optimizationCard.innerHTML = `
        <div class="card-body">
            <h3 class="card-title">优化结果</h3>
            <div class="row">
                <div class="col-md-6">
                    <h4>最优参数</h4>
                    <div class="table-responsive">
                        <table class="table">
                            <tbody>
                                <tr>
                                    <td>止损比例</td>
                                    <td>${(data.best_params.stop_loss * 100).toFixed(2)}%</td>
                                </tr>
                                <tr>
                                    <td>止盈比例</td>
                                    <td>${(data.best_params.take_profit * 100).toFixed(2)}%</td>
                                </tr>
                                <tr>
                                    <td>仓位大小</td>
                                    <td>${(data.best_params.position_size * 100).toFixed(2)}%</td>
                                </tr>
                                <tr>
                                    <td>追踪止损</td>
                                    <td>${(data.best_params.trailing_stop * 100).toFixed(2)}%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    <h4>优化前后对比</h4>
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>指标</th>
                                    <th>优化前</th>
                                    <th>优化后</th>
                                    <th>变化</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>总收益率</td>
                                    <td>${(data.before.total_return * 100).toFixed(2)}%</td>
                                    <td>${(data.after.total_return * 100).toFixed(2)}%</td>
                                    <td class="${data.after.total_return > data.before.total_return ? 'positive' : 'negative'}">
                                        ${((data.after.total_return - data.before.total_return) * 100).toFixed(2)}%
                                    </td>
                                </tr>
                                <tr>
                                    <td>夏普比率</td>
                                    <td>${data.before.sharpe_ratio.toFixed(2)}</td>
                                    <td>${data.after.sharpe_ratio.toFixed(2)}</td>
                                    <td class="${data.after.sharpe_ratio > data.before.sharpe_ratio ? 'positive' : 'negative'}">
                                        ${(data.after.sharpe_ratio - data.before.sharpe_ratio).toFixed(2)}
                                    </td>
                                </tr>
                                <tr>
                                    <td>最大回撤</td>
                                    <td>${(data.before.max_drawdown * 100).toFixed(2)}%</td>
                                    <td>${(data.after.max_drawdown * 100).toFixed(2)}%</td>
                                    <td class="${data.after.max_drawdown < data.before.max_drawdown ? 'positive' : 'negative'}">
                                        ${((data.after.max_drawdown - data.before.max_drawdown) * 100).toFixed(2)}%
                                    </td>
                                </tr>
                                <tr>
                                    <td>胜率</td>
                                    <td>${(data.before.win_rate * 100).toFixed(2)}%</td>
                                    <td>${(data.after.win_rate * 100).toFixed(2)}%</td>
                                    <td class="${data.after.win_rate > data.before.win_rate ? 'positive' : 'negative'}">
                                        ${((data.after.win_rate - data.before.win_rate) * 100).toFixed(2)}%
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
    resultsContainer.appendChild(optimizationCard);

    // 创建优化前后权益曲线对比图
    const equityCurveCard = document.createElement('div');
    equityCurveCard.className = 'chart-container';
    equityCurveCard.innerHTML = `
        <h3 class="chart-title">优化前后权益曲线对比</h3>
        <canvas id="optimizationChart"></canvas>
    `;
    resultsContainer.appendChild(equityCurveCard);

    // 初始化优化对比图
    initOptimizationChart(data.before_equity, data.after_equity);

    // 创建应用优化按钮
    const applyButtonContainer = document.createElement('div');
    applyButtonContainer.className = 'd-flex justify-content-center mt-4';
    applyButtonContainer.innerHTML = `
        <button class="btn btn-primary btn-lg" id="applyOptimizationButton">
            <i class="fas fa-check-circle"></i> 应用优化参数
        </button>
    `;
    resultsContainer.appendChild(applyButtonContainer);

    // 添加应用优化按钮事件
    document.getElementById('applyOptimizationButton').addEventListener('click', function() {
        applyOptimization(data.currency, data.best_params);
    });
}

// 初始化优化对比图
function initOptimizationChart(beforeEquity, afterEquity) {
    const ctx = document.getElementById('optimizationChart').getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: beforeEquity.dates,
            datasets: [
                {
                    label: '优化前',
                    data: beforeEquity.values,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: '优化后',
                    data: afterEquity.values,
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        displayFormats: {
                            month: 'yyyy-MM'
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
                        text: '账户权益'
                    }
                }
            }
        }
    });
}

// 应用优化参数
function applyOptimization(currency, params) {
    // 显示加载动画
    document.getElementById('loading').style.display = 'flex';

    // 准备表单数据
    const formData = new FormData();
    formData.append('currency', currency);
    formData.append('stop_loss', params.stop_loss);
    formData.append('take_profit', params.take_profit);
    formData.append('position_size', params.position_size);
    formData.append('trailing_stop', params.trailing_stop);

    // 发送请求
    fetch('/api/apply_optimization', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载动画
        document.getElementById('loading').style.display = 'none';

        if (data.error) {
            alert('应用优化参数失败: ' + data.error);
            return;
        }

        alert('优化参数已成功应用！');
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        alert('请求失败: ' + error);
    });
}

// 处理登录
function handleLogin() {
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    if (!username || !password) {
        alert('请填写用户名和密码');
        return;
    }

    // 准备表单数据
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    // 发送请求
    fetch('/api/login', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('登录失败: ' + data.error);
            return;
        }

        // 登录成功
        alert('登录成功！');

        // 关闭模态框
        const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
        loginModal.hide();

        // 更新UI显示用户已登录
        updateUIAfterLogin(data.username);
    })
    .catch(error => {
        alert('请求失败: ' + error);
    });
}

// 处理注册
function handleRegister() {
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const agreeTerms = document.getElementById('agreeTerms').checked;

    if (!username || !email || !password || !confirmPassword) {
        alert('请填写所有必填字段');
        return;
    }

    if (password !== confirmPassword) {
        alert('两次输入的密码不一致');
        return;
    }

    if (!agreeTerms) {
        alert('请同意服务条款和隐私政策');
        return;
    }

    // 准备表单数据
    const formData = new FormData();
    formData.append('username', username);
    formData.append('email', email);
    formData.append('password', password);

    // 发送请求
    fetch('/api/register', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('注册失败: ' + data.error);
            return;
        }

        // 注册成功
        alert('注册成功！请登录');

        // 关闭注册模态框
        const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
        registerModal.hide();

        // 打开登录模态框
        const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
        loginModal.show();
    })
    .catch(error => {
        alert('请求失败: ' + error);
    });
}

// 更新UI显示用户已登录
function updateUIAfterLogin(username) {
    const navbarRight = document.querySelector('.navbar .d-flex');
    navbarRight.innerHTML = `
        <div class="dropdown">
            <button class="btn btn-outline-light dropdown-toggle" type="button" id="userDropdown" data-bs-toggle="dropdown">
                <i class="fas fa-user"></i> ${username}
            </button>
            <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="userDropdown">
                <li><a class="dropdown-item" href="#"><i class="fas fa-cog"></i> 账户设置</a></li>
                <li><a class="dropdown-item" href="#"><i class="fas fa-history"></i> 历史记录</a></li>
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item" href="#" id="logoutButton"><i class="fas fa-sign-out-alt"></i> 退出登录</a></li>
            </ul>
        </div>
    `;

    // 添加退出登录按钮事件
    document.getElementById('logoutButton').addEventListener('click', function() {
        handleLogout();
    });
}

// 处理退出登录
function handleLogout() {
    // 发送请求
    fetch('/api/logout', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('退出登录失败: ' + data.error);
            return;
        }

        // 退出登录成功
        alert('已退出登录');

        // 恢复原始UI
        const navbarRight = document.querySelector('.navbar .d-flex');
        navbarRight.innerHTML = `
            <button class="btn btn-outline-light me-2" data-bs-toggle="modal" data-bs-target="#loginModal">
                <i class="fas fa-user"></i> 登录
            </button>
            <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#registerModal">
                <i class="fas fa-user-plus"></i> 注册
            </button>
        `;
    })
    .catch(error => {
        alert('请求失败: ' + error);
    });
}

// 发送客服聊天消息
function sendChatMessage() {
    const input = document.getElementById('csInput');
    const message = input.value.trim();

    if (!message) return;

    // 清空输入框
    input.value = '';

    // 添加用户消息
    const chatBody = document.getElementById('csBody');
    const messageElement = document.createElement('div');
    messageElement.className = 'cs-message cs-sent';
    messageElement.innerHTML = `
        <div class="cs-message-content">${message}</div>
        <div class="cs-message-time">刚刚</div>
    `;
    chatBody.appendChild(messageElement);

    // 滚动到底部
    chatBody.scrollTop = chatBody.scrollHeight;

    // 模拟客服回复
    setTimeout(() => {
        const replyElement = document.createElement('div');
        replyElement.className = 'cs-message cs-received';
        replyElement.innerHTML = `
            <div class="cs-message-content">感谢您的咨询，我们的客服人员将尽快回复您。</div>
            <div class="cs-message-time">刚刚</div>
        `;
        chatBody.appendChild(replyElement);

        // 滚动到底部
        chatBody.scrollTop = chatBody.scrollHeight;
    }, 1000);
}