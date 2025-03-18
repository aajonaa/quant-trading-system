document.addEventListener('DOMContentLoaded', function() {
    // 初始化日期选择器
    flatpickr(".datepicker", {
        dateFormat: "Y-m-d",
        locale: "zh",
        minDate: "2015-01-18",
        maxDate: "today",
        defaultDate: "2015-01-18"
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

    // 登录按钮点击事件
    const loginBtn = document.getElementById('loginBtn');
    const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
    const showRegisterLink = document.getElementById('showRegister');
    const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));

    loginBtn.addEventListener('click', function() {
        loginModal.show();
    });

    showRegisterLink.addEventListener('click', function(e) {
        e.preventDefault();
        loginModal.hide();
        registerModal.show();
    });

    // 处理登录表单提交
    document.getElementById('loginButton').addEventListener('click', function() {
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;

        fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loginModal.hide();
                updateUserInterface(data.username);
            } else {
                alert(data.error || '登录失败');
            }
        });
    });

    // 处理注册表单提交
    document.getElementById('registerButton').addEventListener('click', function() {
        const username = document.getElementById('registerUsername').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        if (password !== confirmPassword) {
            alert('两次输入的密码不一致');
            return;
        }

        fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `username=${encodeURIComponent(username)}&email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                registerModal.hide();
                loginModal.show();
                alert('注册成功，请登录');
            } else {
                alert(data.error || '注册失败');
            }
        });
    });

    // 客服功能
    initializeCustomerService();
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

// 回测功能
function runBacktest() {
    const currency = document.getElementById('backtestCurrency').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    // 验证日期范围
    const minDate = new Date('2015-01-18');
    const maxDate = new Date();
    const selectedStart = new Date(startDate);
    const selectedEnd = new Date(endDate);

    if (selectedStart < minDate || selectedEnd > maxDate) {
        alert('请选择2015-01-18至今天之间的日期范围');
        return;
    }

    document.getElementById('loading').style.display = 'flex';

    fetch('/api/backtest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `currency=${currency}&start_date=${startDate}&end_date=${endDate}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        // 显示回测结果
        const resultsContainer = document.getElementById('backtestResults');
        resultsContainer.innerHTML = `
            <div class="row">
                <div class="col-12">
                    <div class="card mb-4">
                        <div class="card-body">
                            <h5 class="card-title">回测结果 - ${currency}</h5>
                            <img src="/static/images/${currency}_analysis.png" class="img-fluid mb-3" alt="回测分析图">
                            <div class="row">
                                <div class="col-md-6">
                                    <p>总收益率: ${data.metrics.total_return}%</p>
                                    <p>Sharpe比率: ${data.metrics.sharpe_ratio}</p>
                                    <p>Sortino比率: ${data.metrics.sortino_ratio}</p>
                                    <p>最大回撤: ${data.metrics.max_drawdown}%</p>
                                </div>
                                <div class="col-md-6">
                                    <p>胜率: ${data.metrics.win_rate}%</p>
                                    <p>盈亏比: ${data.metrics.profit_factor}</p>
                                    <p>平均收益: ${data.metrics.avg_return}%</p>
                                    <p>最大连续亏损: ${data.metrics.max_consecutive_losses}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    })
    .catch(error => {
        alert('回测请求失败: ' + error);
    })
    .finally(() => {
        document.getElementById('loading').style.display = 'none';
    });
}

// 风险分析功能
function runRiskAnalysis() {
    const selectedPairs = Array.from(document.querySelectorAll('input[name="currencies"]:checked'))
        .map(input => input.value);
    
    if (selectedPairs.length < 2) {
        alert('请至少选择两个货币对进行分析');
        return;
    }

    const analysisType = document.querySelector('input[name="analysisType"]:checked').value;
    
    document.getElementById('loading').style.display = 'flex';

    fetch('/api/risk_analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `pairs=${selectedPairs.join(',')}&type=${analysisType}`
    })
    .then(response => response.json())
    .then(data => {
        const resultsContainer = document.getElementById('riskResults');
        let html = '';

        switch(analysisType) {
            case 'correlation':
                html = `
                    <div class="card mb-4">
                        <div class="card-body">
                            <h5 class="card-title">相关性分析</h5>
                            <img src="/static/images/correlation_heatmap.png" class="img-fluid" alt="相关性热图">
                        </div>
                    </div>
                `;
                break;
            case 'risk':
                html = `
                    <div class="card mb-4">
                        <div class="card-body">
                            <h5 class="card-title">风险评估</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <img src="/static/images/risk_heatmap.png" class="img-fluid mb-3" alt="风险热图">
                                </div>
                                <div class="col-md-6">
                                    <img src="/static/images/risk_timeseries.png" class="img-fluid mb-3" alt="风险时间序列">
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                break;
            case 'comprehensive':
                html = `
                    <div class="card mb-4">
                        <div class="card-body">
                            <h5 class="card-title">综合分析结果</h5>
                            <div class="table-responsive">
                                <table class="table table-striped">
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
                                    <tbody>
                                        ${data.results.filter(r => selectedPairs.includes(r.pair))
                                            .map(risk => `
                                            <tr>
                                                <td>${risk.pair}</td>
                                                <td>${risk.correlation}</td>
                                                <td>${risk.volatility}</td>
                                                <td>${risk.risk_score}</td>
                                                <td>${risk.risk_level}</td>
                                                <td>${risk.recommendation}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                `;
                break;
        }

        resultsContainer.innerHTML = html;
        resultsContainer.style.display = 'block';
    })
    .catch(error => {
        alert('风险分析请求失败: ' + error);
    })
    .finally(() => {
        document.getElementById('loading').style.display = 'none';
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

// 更新用户界面
function updateUserInterface(username) {
    const navbarRight = document.querySelector('.navbar .d-flex');
    navbarRight.innerHTML = `
        <div class="dropdown">
            <button class="btn btn-outline-light dropdown-toggle" type="button" data-bs-toggle="dropdown">
                <i class="fas fa-user"></i> ${username}
            </button>
            <div class="dropdown-menu dropdown-menu-end">
                <a class="dropdown-item" href="#"><i class="fas fa-cog"></i> 设置</a>
                <div class="dropdown-divider"></div>
                <a class="dropdown-item" href="#" id="logoutBtn">
                    <i class="fas fa-sign-out-alt"></i> 退出
                </a>
            </div>
        </div>
    `;

    // 添加退出事件
    document.getElementById('logoutBtn').addEventListener('click', function() {
        fetch('/api/logout', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                }
            });
    });
}

// 客服功能
function initializeCustomerService() {
    const csButton = document.getElementById('csButton');
    const csPanel = document.getElementById('csPanel');
    const csClose = document.getElementById('csClose');
    const csInput = document.getElementById('csInput');
    const csBody = document.getElementById('csBody');
    const csSend = document.getElementById('csSend');

    if (csButton && csPanel && csClose) {
        csButton.addEventListener('click', function() {
            csPanel.style.display = csPanel.style.display === 'none' ? 'flex' : 'none';
        });

        csClose.addEventListener('click', function() {
            csPanel.style.display = 'none';
        });

        csSend.addEventListener('click', sendMessage);
        csInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    function sendMessage() {
        const message = csInput.value.trim();
        if (!message) return;

        // 添加用户消息
        addMessage(message, 'sent');
        csInput.value = '';

        // 模拟客服回复
        setTimeout(() => {
            addMessage('感谢您的咨询，我们的客服人员将尽快回复您。', 'received');
        }, 1000);
    }

    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `cs-message cs-${type}`;
        messageDiv.innerHTML = `
            <div class="cs-message-content">${text}</div>
            <div class="cs-message-time">刚刚</div>
        `;
        csBody.appendChild(messageDiv);
        csBody.scrollTop = csBody.scrollHeight;
    }
}

// 单一货币对回测
function runSingleBacktest() {
    const currencyPair = document.getElementById('currency-pair-select').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    if (!startDate || !endDate) {
        alert('请选择日期范围');
        return;
    }

    showLoading('正在进行回测...');

    fetch('/api/single_backtest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            currency_pair: currencyPair,
            start_date: startDate,
            end_date: endDate
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            updateBacktestResults(data.results);
            drawEquityCurve(data.results.equity_curve);
        } else {
            alert('回测失败: ' + data.error);
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('回测过程中发生错误');
    });
}

// 更新回测结果显示
function updateBacktestResults(results) {
    document.getElementById('backtest-results').style.display = 'block';
    document.getElementById('total-return').textContent = 
        (results.total_return * 100).toFixed(2) + '%';
    // 更新其他指标...
}

// 绘制权益曲线
function drawEquityCurve(equityCurve) {
    const ctx = document.getElementById('equity-curve-chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: equityCurve.length}, (_, i) => i + 1),
            datasets: [{
                label: '权益曲线',
                data: equityCurve,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 多货币风险分析
function analyzeMultiCurrencyRisk() {
    const selectedPairs = Array.from(document.querySelectorAll('.currency-pairs-checkboxes input:checked'))
        .map(checkbox => checkbox.value);

    if (selectedPairs.length === 0) {
        alert('请至少选择一个货币对组合');
        return;
    }

    showLoading('正在分析风险...');

    fetch('/api/multi_currency_risk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            currency_pairs: selectedPairs
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            displayRiskAnalysis(data.risk_analysis);
        } else {
            alert('风险分析失败: ' + data.error);
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('风险分析过程中发生错误');
    });
}

// 显示风险分析结果
function displayRiskAnalysis(riskData) {
    const resultsDiv = document.getElementById('risk-analysis-results');
    resultsDiv.style.display = 'block';
    
    let html = '<table class="table">';
    html += `<thead>
        <tr>
            <th>货币对组合</th>
            <th>相关系数</th>
            <th>组合波动率</th>
            <th>信号一致性</th>
            <th>风险得分</th>
            <th>风险等级</th>
            <th>交易建议</th>
        </tr>
    </thead><tbody>`;

    riskData.forEach(item => {
        html += `<tr>
            <td>${item.货币对组合}</td>
            <td>${item.相关系数.toFixed(2)}</td>
            <td>${item.组合波动率.toFixed(2)}</td>
            <td>${item.信号一致性.toFixed(2)}</td>
            <td>${item.风险得分.toFixed(2)}</td>
            <td>${item.风险等级}</td>
            <td>${item.交易建议}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    resultsDiv.innerHTML = html;
}

// 工具函数
function showLoading(message) {
    // 显示加载提示
}

function hideLoading() {
    // 隐藏加载提示
}