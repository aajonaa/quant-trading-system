document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM已加载，开始初始化...");
    
    // 初始化日期选择器
    flatpickr(".datepicker", {
        dateFormat: "Y-m-d",
        locale: "zh",
        minDate: "2015-01-28",
        maxDate: "today"
    });

    // 初始化图表配置
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif";
    Chart.defaults.color = '#666';
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0,0,0,0.8)';
    
    // 安全地绑定事件监听
    safeSetupEventListeners();
    
    // 初始化日期选择器（所有页面）
    setupAllDatepickers();
    
    // 检查登录状态
    checkLoginStatus();
    
    // 初始化登录和充值按钮
    initializeAuthButtons();
    
    // 如果在风险控制页面，加载风险分析数据
    if (document.getElementById('risk-analysis-container')) {
        console.log("检测到风险分析容器，开始加载数据...");
        setTimeout(loadRiskAnalysis, 500);
    }
});

// 安全地绑定事件监听器，避免null错误
function safeSetupEventListeners() {
    console.log("设置事件监听器...");
    
    // 回测表单提交
    const backtestForm = document.getElementById('backtestForm');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            runBacktest();
        });
    }

    // 信号解释表单提交
    const signalExplainForm = document.getElementById('signalExplainForm');
    if (signalExplainForm) {
        signalExplainForm.addEventListener('submit', function(e) {
            e.preventDefault();
            generateSignalExplanation();
        });
    }

    // 优化表单提交
    const optimizeForm = document.getElementById('optimizeForm');
    if (optimizeForm) {
        optimizeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            optimizeStrategy();
        });
    }

    // 客服系统初始化
    initializeCustomerService();

    // 注册相关
    const showRegister = document.getElementById('showRegister');
    if (showRegister) {
        showRegister.addEventListener('click', function() {
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            if (loginModal) loginModal.hide();
            const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
            registerModal.show();
        });
    }

    // 注册表单提交
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleRegistration();
        });
    }

    // 充值选项
    const rechargeItems = document.querySelectorAll('.recharge-item');
    if (rechargeItems.length > 0) {
        rechargeItems.forEach(item => {
            item.addEventListener('click', function() {
                document.querySelectorAll('.recharge-item').forEach(i => i.classList.remove('selected'));
                this.classList.add('selected');
                document.getElementById('custom-amount').value = '';
            });
        });
    }

    // 支付方法选择
    const paymentButtons = document.querySelectorAll('.btn-payment');
    if (paymentButtons.length > 0) {
        paymentButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const amount = document.querySelector('.recharge-item.selected')?.dataset.amount 
                    || document.getElementById('custom-amount').value;
                if (!amount) {
                    alert('请选择或输入充值金额');
                    return;
                }
                alert(`正在跳转到${this.textContent.trim()}支付页面，金额：¥${amount}`);
            });
        });
    }
}

// 客服系统初始化
function initializeCustomerService() {
    const csButton = document.querySelector('.cs-button');
    const csClose = document.querySelector('.cs-close');
    
    if (csButton) {
        csButton.addEventListener('click', function(e) {
            e.stopPropagation();
            const chatWindow = document.querySelector('.cs-chat-window');
            if (chatWindow) {
                if (chatWindow.style.display === 'none' || !chatWindow.style.display) {
                    chatWindow.style.display = 'flex';
                    if (!document.querySelector('.cs-message')) {
                        addWelcomeMessage();
                    }
                } else {
                    chatWindow.style.display = 'none';
                }
            }
        });
    }

    if (csClose) {
        csClose.addEventListener('click', function() {
            const chatWindow = document.querySelector('.cs-chat-window');
            if (chatWindow) chatWindow.style.display = 'none';
        });
    }

    // 关闭客服窗口的点击事件（点击窗口外部）
    document.addEventListener('click', function(e) {
        const chatWindow = document.querySelector('.cs-chat-window');
        const csButton = document.querySelector('.cs-button');
        
        if (chatWindow && csButton && 
            chatWindow.style.display === 'flex' &&
            !chatWindow.contains(e.target) && 
            !csButton.contains(e.target)) {
            chatWindow.style.display = 'none';
        }
    });
}

// 设置所有日期选择器
function setupAllDatepickers() {
    console.log("设置所有日期选择器...");
    setupBacktestDatepickers();
    setupSignalDatepickers();
    setupOptimizationForm();
}

// 全局默认日期值
let globalStartDate = '2015-01-28';
let globalEndDate = formatCurrentDate();

// 获取当前日期的格式化字符串
function formatCurrentDate() {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// 风险控制数据加载
function loadRiskAnalysis() {
    console.log("开始加载风险分析数据...");
    const riskContainer = document.getElementById('risk-analysis-container');
    if (!riskContainer) {
        console.error("找不到风险分析容器");
        return;
    }
    
    // 显示加载中状态
    riskContainer.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <p class="mt-2">正在加载风险分析数据...</p>
        </div>
    `;
    
    // 硬编码风险数据
    const riskData = [
        {
            "货币对组合": "CNYEUR-CNYGBP",
            "相关系数": "0.5816",
            "组合波动率": "0.1471",
            "信号一致性": "0.0329",
            "风险得分": "28.67",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYGBP",
            "相关系数": "0.4844",
            "组合波动率": "0.1665",
            "信号一致性": "-0.042",
            "风险得分": "25.63",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYEUR",
            "相关系数": "0.4642",
            "组合波动率": "0.1522",
            "信号一致性": "-0.0091",
            "风险得分": "23.41",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYEUR-CNYJPY",
            "相关系数": "0.4077",
            "组合波动率": "0.139",
            "信号一致性": "-0.025",
            "风险得分": "21.23",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYJPY-CNYUSD",
            "相关系数": "0.2403",
            "组合波动率": "0.1078",
            "信号一致性": "0.0689",
            "风险得分": "14.91",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYAUD-CNYJPY",
            "相关系数": "0.1975",
            "组合波动率": "0.1497",
            "信号一致性": "0.0424",
            "风险得分": "13.66",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYGBP-CNYJPY",
            "相关系数": "0.2005",
            "组合波动率": "0.1405",
            "信号一致性": "0.0303",
            "风险得分": "13.14",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYEUR-CNYUSD",
            "相关系数": "0.2008",
            "组合波动率": "0.0916",
            "信号一致性": "-0.0303",
            "风险得分": "11.69",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYGBP-CNYUSD",
            "相关系数": "0.1027",
            "组合波动率": "0.1029",
            "信号一致性": "0.0628",
            "风险得分": "9.08",
            "风险等级": "极低风险",
            "交易建议": "可以增加敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYUSD",
            "相关系数": "0.0347",
            "组合波动率": "0.1115",
            "信号一致性": "0.0049",
            "风险得分": "4.88",
            "风险等级": "极低风险",
            "交易建议": "可以增加敞口"
        }
    ];
    
    // 直接创建表格
    let tableHTML = `
        <div class="table-responsive mt-4">
            <table class="table table-striped table-hover">
                <thead class="table-light">
                    <tr>
                        <th>货币对组合</th>
                        <th>相关系数</th>
                        <th>组合波动率</th>
                        <th>信号一致性</th>
                        <th>风险得分</th>
                        <th>风险等级</th>
                        <th>交易建议</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // 添加数据行
    riskData.forEach(item => {
        // 根据风险等级设置不同的颜色
        let riskClass = '';
        if (item.风险等级 && item.风险等级.includes('极低')) {
            riskClass = 'table-success';
        } else if (item.风险等级 && item.风险等级.includes('低')) {
            riskClass = 'table-info';
        } else if (item.风险等级 && item.风险等级.includes('中')) {
            riskClass = 'table-warning';
        } else {
            riskClass = 'table-danger';
        }
        
        tableHTML += `
            <tr class="${riskClass}">
                <td>${item.货币对组合}</td>
                <td>${item.相关系数}</td>
                <td>${item.组合波动率}</td>
                <td>${item.信号一致性}</td>
                <td>${item.风险得分}</td>
                <td>${item.风险等级}</td>
                <td>${item.交易建议}</td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    riskContainer.innerHTML = tableHTML;
    console.log("风险分析表格已生成");
}

// 回测功能实现
async function runBacktest() {
    try {
        const currencyPair = document.getElementById('currency-pair').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;

        // 验证日期范围
        const start = new Date(startDate);
        const end = new Date(endDate);
        const today = new Date();
        
        if (start > end || end > today) {
            showError('请选择有效的日期范围');
            return;
        }

        showLoading('正在执行回测分析...');

        const response = await fetch('/api/single_backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                currency_pair: currencyPair,
                start_date: startDate,
                end_date: endDate
            })
        });

        const data = await response.json();

        if (data.success) {
            displayBacktestResults(data.results);
        } else {
            showError(data.error || '回测执行失败');
        }
    } catch (error) {
        showError('回测执行出错: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayBacktestResults(results) {
    // 清空指标行并添加新指标
    const metricsRow = document.querySelector('.metrics-row');
    if (metricsRow) {
        metricsRow.innerHTML = `
            <div class="metric-card">
                <div class="metric-label">总收益率</div>
                <div class="metric-value">${(results.total_return * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">${results.sharpe_ratio.toFixed(2)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value">${(results.max_drawdown * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">胜率</div>
                <div class="metric-value">${(results.win_rate * 100).toFixed(2)}%</div>
            </div>
        `;
    }

    // 优化权益曲线图表配置
    const ctx = document.getElementById('equity-curve').getContext('2d');
    if (equityChart) {
        equityChart.destroy();
    }

    // 准备图表数据
    const chartData = [];
    const equityCurve = results.equity_curve;
    for (const date in equityCurve) {
        chartData.push({
            x: new Date(date),
            y: equityCurve[date]
        });
    }

    // 设置更适合显示的图表高度
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        chartContainer.style.height = '350px';
    }

    // 创建图表，使用更适合的配置
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: '账户权益',
                data: chartData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                tooltip: {
                    enabled: true,
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
                    ticks: {
                        source: 'auto',
                        autoSkip: true,
                        maxTicksLimit: 12
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 多货币风险分析
async function analyzeMultiCurrencyRisk() {
    showLoading('正在分析风险...');

    try {
        const response = await fetch('/api/multi_currency_risk', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();
        if (data.success) {
            displayRiskAnalysis(data.risk_analysis);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('风险分析过程发生错误');
        console.error(error);
    } finally {
        hideLoading();
    }
}

// 策略优化功能
document.getElementById('optimizeForm').addEventListener('submit', function(e) {
    e.preventDefault();
    runOptimization();
});

async function runOptimization() {
    try {
        const currencyPair = document.getElementById('optimize-currency-pair').value;
        const startDate = document.getElementById('optimize-start-date').value;
        const endDate = document.getElementById('optimize-end-date').value;

        // 验证日期范围
        const start = new Date(startDate);
        const end = new Date(endDate);
        const today = new Date();
        
        if (start > end || end > today) {
            showError('请选择有效的日期范围');
            return;
        }

        showLoading('正在执行策略优化...');

        const response = await fetch('/api/optimize_strategy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                currency_pair: currencyPair,
                start_date: startDate,
                end_date: endDate
            })
        });

        const data = await response.json();

        if (data.success) {
            displayOptimizationResults(data);
        } else {
            showError(data.error || '策略优化失败');
        }
    } catch (error) {
        showError('策略优化出错: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayOptimizationResults(data) {
    const resultsContainer = document.getElementById('optimization-results');
    
    // 创建优化结果表格
    let html = `
        <h4 class="mb-4">优化结果对比</h4>
        <div class="table-responsive">
            <table class="table table-bordered">
                <thead class="table-light">
                    <tr>
                        <th>指标</th>
                        <th>原始策略</th>
                        <th>优化策略</th>
                        <th>提升</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>总收益率</td>
                        <td>${(data.original.total_return * 100).toFixed(2)}%</td>
                        <td>${(data.optimized.total_return * 100).toFixed(2)}%</td>
                        <td class="${data.improvement.total_return > 0 ? 'text-success' : 'text-danger'}">
                            ${(data.improvement.total_return * 100).toFixed(2)}%
                        </td>
                    </tr>
                    <tr>
                        <td>夏普比率</td>
                        <td>${data.original.sharpe_ratio.toFixed(2)}</td>
                        <td>${data.optimized.sharpe_ratio.toFixed(2)}</td>
                        <td class="${data.improvement.sharpe_ratio > 0 ? 'text-success' : 'text-danger'}">
                            ${data.improvement.sharpe_ratio.toFixed(2)}
                        </td>
                    </tr>
                    <tr>
                        <td>最大回撤</td>
                        <td>${(data.original.max_drawdown * 100).toFixed(2)}%</td>
                        <td>${(data.optimized.max_drawdown * 100).toFixed(2)}%</td>
                        <td class="${data.improvement.max_drawdown < 0 ? 'text-success' : 'text-danger'}">
                            ${(data.improvement.max_drawdown * 100).toFixed(2)}%
                        </td>
                    </tr>
                    <tr>
                        <td>胜率</td>
                        <td>${(data.original.win_rate * 100).toFixed(2)}%</td>
                        <td>${(data.optimized.win_rate * 100).toFixed(2)}%</td>
                        <td class="${data.improvement.win_rate > 0 ? 'text-success' : 'text-danger'}">
                            ${(data.improvement.win_rate * 100).toFixed(2)}%
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">优化参数</div>
                    <div class="card-body">
                        <ul class="list-group">
                            ${Object.entries(data.optimized.parameters || {}).map(([key, value]) => 
                                `<li class="list-group-item d-flex justify-content-between align-items-center">
                                    ${key}
                                    <span class="badge bg-primary rounded-pill">${value}</span>
                                </li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">优化建议</div>
                    <div class="card-body">
                        <p>${data.optimized.recommendation || '根据优化结果，建议采用新的参数设置以提高策略性能。'}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

// 设置优化表单日期
function setupOptimizationForm() {
    const optimizeStartDateInput = document.getElementById('optimize-start-date');
    const optimizeEndDateInput = document.getElementById('optimize-end-date');
    
    if (optimizeStartDateInput && optimizeEndDateInput) {
        // 使用全局值
        optimizeStartDateInput.value = globalStartDate;
        optimizeEndDateInput.value = globalEndDate;
        
        // 初始化日期选择器
        flatpickr(optimizeStartDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalStartDate = dateStr;
                syncDates();
            }
        });
        
        flatpickr(optimizeEndDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalEndDate = dateStr;
                syncDates();
            }
        });
    }
}

// 客服功能
function toggleCustomerService() {
    const chatWindow = document.querySelector('.cs-chat-window');
    if (chatWindow.style.display === 'none' || !chatWindow.style.display) {
        chatWindow.style.display = 'flex';
        // 添加欢迎消息
        if (!document.querySelector('.cs-message')) {
            const messagesDiv = document.getElementById('csMessages');
            messagesDiv.innerHTML = `
                <div class="cs-message service">
                    <div class="message-content">
                        您好！我是您的专属客服，很高兴为您服务。请问有什么可以帮助您的吗？
                    </div>
                </div>
            `;
        }
    } else {
        chatWindow.style.display = 'none';
    }
}

function sendCustomerMessage() {
    const input = document.getElementById('csInput');
    const message = input.value.trim();
    if (!message) return;

    const messagesDiv = document.getElementById('csMessages');
    
    // 添加用户消息
    messagesDiv.innerHTML += `
        <div class="cs-message user">
            <div class="message-content">${message}</div>
        </div>
    `;

    // 自动回复
    setTimeout(() => {
        const responses = [
            "感谢您的咨询，我正在为您处理...",
            "您的问题我已经收到，请稍等片刻...",
            "我明白您的需求，让我为您找到最佳解决方案...",
            "这是一个很好的问题，我来为您详细解答..."
        ];
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        
        messagesDiv.innerHTML += `
            <div class="cs-message service">
                <div class="message-content">${randomResponse}</div>
            </div>
        `;
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }, 1000);

    input.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// 添加回车发送功能
document.getElementById('csInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendCustomerMessage();
    }
});

// 工具函数
function showLoading(message) {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="mt-3">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

function showError(message) {
    alert(message);
}

// 检查登录状态
function checkLoginStatus() {
    const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
    const guestNav = document.getElementById('guest-nav');
    const userNav = document.getElementById('user-nav');
    
    if (isLoggedIn && guestNav && userNav) {
        guestNav.style.display = 'none';
        userNav.style.display = 'flex';
    }
}

// 退出登录
function logout() {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('username');
    const guestNav = document.getElementById('guest-nav');
    const userNav = document.getElementById('user-nav');
    
    if (guestNav && userNav) {
        guestNav.style.display = 'flex';
        userNav.style.display = 'none';
    }
}

// 修改登录处理函数
function handleLogin(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showError('请输入用户名和密码');
        return;
    }
    
    fetch('/api/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 隐藏登录模态框
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            if (loginModal) {
                loginModal.hide();
                
                // 手动移除模态框背景
                const modalBackdrops = document.querySelectorAll('.modal-backdrop');
                modalBackdrops.forEach(backdrop => {
                    backdrop.classList.remove('show');
                    setTimeout(() => backdrop.remove(), 150);
                });
            }
            
            // 更新用户界面
            document.getElementById('guest-nav').style.display = 'none';
            document.getElementById('user-nav').style.display = 'flex';
            
            // 保存登录状态
            localStorage.setItem('isLoggedIn', 'true');
            localStorage.setItem('username', data.user.username);
            
            // 显示成功消息
            showSuccess('登录成功！');
        } else {
            showError(data.error || '登录失败');
        }
    })
    .catch(error => {
        showError('请求失败: ' + error.message);
    });
}

// 生成信号解释
async function generateSignalExplanation(event) {
    if (event) event.preventDefault();
    
    try {
        const currencyPair = document.getElementById('signal-currency-pair').value;
        const startDate = document.getElementById('signal-start-date').value;
        const endDate = document.getElementById('signal-end-date').value;
        
        // 显示加载指示器
        const container = document.getElementById('signal-explanation-container');
        container.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">加载中...</span>
                </div>
                <p class="mt-2">正在生成信号解释...</p>
            </div>
        `;
        
        // 验证日期范围
        const start = new Date(startDate);
        const end = new Date(endDate);
        
        if (start > end) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    开始日期不能大于结束日期
                </div>
            `;
            return;
        }
        
        const response = await fetch('/api/signal_explanation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                currency_pair: currencyPair,
                start_date: startDate,
                end_date: endDate
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySignalExplanation(currencyPair, data.explanation);
        } else {
            container.innerHTML = `
                <div class="alert alert-danger">
                    ${data.error || '生成信号解释失败'}
                </div>
            `;
        }
    } catch (error) {
        const container = document.getElementById('signal-explanation-container');
        container.innerHTML = `
            <div class="alert alert-danger">
                生成信号解释出错: ${error.message}
            </div>
        `;
        console.error('信号解释生成错误:', error);
    }
}

// 显示信号解释
function displaySignalExplanation(currencyPair, explanation) {
    const container = document.getElementById('signal-explanation-container');
    
    let html = `
        <div class="signal-explanation">
            <h4 class="mb-3">${currencyPair} 信号分析</h4>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">市场趋势概述</h5>
                </div>
                <div class="card-body">
                    <p>${explanation.市场趋势概述 || '暂无趋势分析'}</p>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">信号解读</h5>
                </div>
                <div class="card-body">
                    <p>${explanation.信号解读 || '暂无信号解读'}</p>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">技术面分析</h5>
                </div>
                <div class="card-body">
                    <p>${explanation.技术面分析 || '暂无技术面分析'}</p>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">风险评估</h5>
                </div>
                <div class="card-body">
                    <p>${explanation.风险评估 || '暂无风险评估'}</p>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">未来展望</h5>
                </div>
                <div class="card-body">
                    <p>${explanation.未来展望 || '暂无未来展望'}</p>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// 设置信号解释日期范围
function setupSignalDatepickers() {
    const signalStartDateInput = document.getElementById('signal-start-date');
    const signalEndDateInput = document.getElementById('signal-end-date');
    
    if (signalStartDateInput && signalEndDateInput) {
        // 使用全局值
        signalStartDateInput.value = globalStartDate;
        signalEndDateInput.value = globalEndDate;
        
        // 初始化日期选择器
        flatpickr(signalStartDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalStartDate = dateStr;
                syncDates();
            }
        });
        
        flatpickr(signalEndDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalEndDate = dateStr;
                syncDates();
            }
        });
    }
}

// 在页面加载时设置全部日期选择器
function setupBacktestDatepickers() {
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    
    if (startDateInput && endDateInput) {
        // 设置默认值
        startDateInput.value = globalStartDate;
        endDateInput.value = globalEndDate;
        
        // 初始化日期选择器
        flatpickr(startDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalStartDate = dateStr;
                // 同步到其他日期选择器
                syncDates();
            }
        });
        
        flatpickr(endDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today',
            onChange: function(selectedDates, dateStr) {
                globalEndDate = dateStr;
                // 同步到其他日期选择器
                syncDates();
            }
        });
    }
}

// 确保登录和充值按钮在所有页面都有效
function initializeAuthButtons() {
    // 登录按钮点击
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
        loginBtn.addEventListener('click', function() {
            const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
            loginModal.show();
        });
    }

    // 充值按钮点击
    const rechargeBtn = document.getElementById('rechargeBtn');
    if (rechargeBtn) {
        rechargeBtn.addEventListener('click', function() {
            const rechargeModal = new bootstrap.Modal(document.getElementById('rechargeModal'));
            rechargeModal.show();
        });
    }

    // 登录表单提交
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleLogin(e);
        });
    }
}

// 同步所有日期选择器
function syncDates() {
    // 回测分析日期
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    if (startDateInput) startDateInput._flatpickr.setDate(globalStartDate);
    if (endDateInput) endDateInput._flatpickr.setDate(globalEndDate);
    
    // 信号解释日期
    const signalStartDateInput = document.getElementById('signal-start-date');
    const signalEndDateInput = document.getElementById('signal-end-date');
    if (signalStartDateInput) signalStartDateInput._flatpickr.setDate(globalStartDate);
    if (signalEndDateInput) signalEndDateInput._flatpickr.setDate(globalEndDate);
    
    // 优化表单日期
    const optimizeStartDateInput = document.getElementById('optimize-start-date');
    const optimizeEndDateInput = document.getElementById('optimize-end-date');
    if (optimizeStartDateInput) optimizeStartDateInput._flatpickr.setDate(globalStartDate);
    if (optimizeEndDateInput) optimizeEndDateInput._flatpickr.setDate(globalEndDate);
}