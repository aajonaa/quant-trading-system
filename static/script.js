document.addEventListener('DOMContentLoaded', function() {
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
    
    // 绑定事件监听
    setupEventListeners();

    // 检查登录状态
    checkLoginStatus();

    // 如果在风险控制页面，加载风险分析数据
    if (document.getElementById('risk-analysis-container')) {
        loadRiskAnalysis();
    }

    // 设置优化表单
    setupOptimizationForm();
});

// 事件监听设置
function setupEventListeners() {
    // 回测表单提交
    document.getElementById('backtestForm').addEventListener('submit', function(e) {
        e.preventDefault();
        runBacktest();
    });

    // 客服系统点击事件
    document.addEventListener('click', function(e) {
        const chatWindow = document.querySelector('.cs-chat-window');
        const csButton = document.querySelector('.cs-button');
        
        // 如果点击的不是客服窗口内部元素且不是客服按钮
        if (chatWindow && chatWindow.style.display === 'flex' &&
            !chatWindow.contains(e.target) && 
            !csButton.contains(e.target)) {
            chatWindow.style.display = 'none';
        }
    });

    // 客服按钮点击
    const csButton = document.querySelector('.cs-button');
    const csClose = document.querySelector('.cs-close');
    
    if (csButton) {
        csButton.addEventListener('click', function(e) {
            e.stopPropagation();
            const chatWindow = document.querySelector('.cs-chat-window');
            if (chatWindow.style.display === 'none' || !chatWindow.style.display) {
                chatWindow.style.display = 'flex';
                if (!document.querySelector('.cs-message')) {
                    addWelcomeMessage();
                }
            } else {
                chatWindow.style.display = 'none';
            }
        });
    }

    // 添加关闭按钮事件
    if (csClose) {
        csClose.addEventListener('click', function() {
            const chatWindow = document.querySelector('.cs-chat-window');
            chatWindow.style.display = 'none';
        });
    }

    // 登录按钮点击
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
        loginBtn.addEventListener('click', () => {
            const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
            loginModal.show();
        });
    }

    // 注册相关
    document.getElementById('showRegister').addEventListener('click', function() {
        const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
        loginModal.hide();
        const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
        registerModal.show();
    });

    document.getElementById('registerForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const username = document.getElementById('reg-username').value;
        const password = document.getElementById('reg-password').value;
        const confirmPassword = document.getElementById('reg-confirm-password').value;
        const email = document.getElementById('reg-email').value;

        if (password !== confirmPassword) {
            alert('两次输入的密码不一致');
            return;
        }

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password, email })
            });
            
            const data = await response.json();
            if (data.success) {
                alert('注册成功！请登录');
                const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
                registerModal.hide();
                const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
                loginModal.show();
            } else {
                alert(data.error || '注册失败，请稍后重试');
            }
        } catch (error) {
            alert('注册失败，请稍后重试');
            console.error(error);
        }
    });

    // 充值相关
    document.getElementById('rechargeBtn').addEventListener('click', function() {
        const rechargeModal = new bootstrap.Modal(document.getElementById('rechargeModal'));
        rechargeModal.show();
    });

    // 选择充值金额
    document.querySelectorAll('.recharge-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.recharge-item').forEach(i => i.classList.remove('selected'));
            this.classList.add('selected');
            document.getElementById('custom-amount').value = '';
        });
    });

    // 支付方法选择
    document.querySelectorAll('.btn-payment').forEach(btn => {
        btn.addEventListener('click', function() {
            const amount = document.querySelector('.recharge-item.selected')?.dataset.amount 
                || document.getElementById('custom-amount').value;
            if (!amount) {
                alert('请选择或输入充值金额');
                return;
            }
            // 处理支付逻辑
            alert(`正在跳转到${this.textContent.trim()}支付页面，金额：¥${amount}`);
        });
    });

    // 设置默认日期范围
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    
    if (startDateInput && endDateInput) {
        // 设置开始日期为 2015-01-28
        startDateInput.value = '2015-01-28';
        
        // 设置结束日期为今天
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        endDateInput.value = `${year}-${month}-${day}`;
        
        // 初始化日期选择器
        flatpickr(startDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today'
        });
        
        flatpickr(endDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today'
        });
    }
}

function addWelcomeMessage() {
    const messagesDiv = document.getElementById('csMessages');
    messagesDiv.innerHTML = `
        <div class="cs-message service">
            <div class="message-content">
                您好！我是您的专属客服，很高兴为您服务。请问有什么可以帮助您的吗？
            </div>
        </div>
    `;
}

// 全局变量存储图表实例
let equityChart = null;

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
    // 更新回测指标显示
    const metricsRow = document.querySelector('.metrics-row');
    metricsRow.innerHTML = `
        <div class="metric-item">
            <h4>总收益</h4>
            <p>${(results.total_return * 100).toFixed(2)}%</p>
        </div>
        <div class="metric-item">
            <h4>夏普比率</h4>
            <p>${results.sharpe_ratio.toFixed(2)}</p>
        </div>
        <div class="metric-item">
            <h4>最大回撤</h4>
            <p>${(results.max_drawdown * 100).toFixed(2)}%</p>
        </div>
        <div class="metric-item">
            <h4>胜率</h4>
            <p>${(results.win_rate * 100).toFixed(2)}%</p>
        </div>
    `;

    // 转换权益曲线数据并过滤无效值
    const chartData = Object.entries(results.equity_curve)
        .map(([date, value]) => ({
            x: new Date(date),
            y: parseFloat(value)
        }))
        .filter(point => 
            !isNaN(point.y) && 
            point.y > 0 && 
            point.x >= new Date(document.getElementById('start-date').value) &&
            point.x <= new Date(document.getElementById('end-date').value)
        )
        .sort((a, b) => a.x - b.x);

    // 创建新的权益曲线图表
    const ctx = document.getElementById('equity-curve').getContext('2d');
    if (equityChart) {
        equityChart.destroy();
    }

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: '账户权益',
                data: chartData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: false,
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
                title: {
                    display: true,
                    text: '权益曲线'
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `账户权益: ${context.parsed.y.toLocaleString('zh-CN', {
                                style: 'currency',
                                currency: 'CNY'
                            })}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'yyyy-MM-dd'
                        },
                        tooltipFormat: 'yyyy-MM-dd'
                    },
                    grid: {
                        display: true,
                        drawOnChartArea: true
                    },
                    ticks: {
                        source: 'data',
                        autoSkip: true,
                        maxTicksLimit: 30,
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '账户权益 (CNY)'
                    },
                    grid: {
                        display: true,
                        drawOnChartArea: true
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString('zh-CN', {
                                style: 'currency',
                                currency: 'CNY'
                            });
                        }
                    }
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

// 设置优化表单的默认日期
function setupOptimizationForm() {
    const startDateInput = document.getElementById('optimize-start-date');
    const endDateInput = document.getElementById('optimize-end-date');
    
    if (startDateInput && endDateInput) {
        // 设置开始日期为 2015-01-28
        startDateInput.value = '2015-01-28';
        
        // 设置结束日期为今天
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        endDateInput.value = `${year}-${month}-${day}`;
        
        // 初始化日期选择器
        flatpickr(startDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today'
        });
        
        flatpickr(endDateInput, {
            dateFormat: 'Y-m-d',
            minDate: '2015-01-28',
            maxDate: 'today'
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

// 修改登录成功的处理函数
function handleLoginSuccess(data) {
    if (data.success) {
        // 隐藏登录模态框
        const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
        if (loginModal) {
            loginModal.hide();
        }
        
        // 更新界面显示
        const guestNav = document.getElementById('guest-nav');
        const userNav = document.getElementById('user-nav');
        
        if (guestNav && userNav) {
            guestNav.style.display = 'none';
            userNav.style.display = 'flex';
            localStorage.setItem('isLoggedIn', 'true');
            localStorage.setItem('username', data.user.username);
        }
    } else {
        showError(data.error || '登录失败');
    }
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

async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;

    if (!username || !password) {
        showError('请输入用户名和密码');
        return;
    }

    try {
        showLoading('登录中...');
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (data.success) {
            handleLoginSuccess(data);
        } else {
            showError(data.error || '登录失败');
            // 清空密码输入框
            document.getElementById('loginPassword').value = '';
        }
    } catch (error) {
        showError('登录请求失败');
    } finally {
        hideLoading();
    }
}

// 添加风险控制功能
async function loadRiskAnalysis() {
    try {
        showLoading('加载风险分析数据...');
        
        const response = await fetch('/api/risk_analysis');
        const data = await response.json();
        
        if (data.success) {
            displayRiskAnalysis(data.data);
        } else {
            showError(data.error || '加载风险分析数据失败');
        }
    } catch (error) {
        showError('加载风险分析数据出错: ' + error.message);
    } finally {
        hideLoading();
    }
}

// 显示风险分析结果
function displayRiskAnalysis(riskData) {
    const riskContainer = document.getElementById('risk-analysis-container');
    if (!riskContainer) return;
    
    // 创建表格
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
        if (item.风险等级.includes('极低')) {
            riskClass = 'table-success';
        } else if (item.风险等级.includes('低')) {
            riskClass = 'table-info';
        } else if (item.风险等级.includes('中')) {
            riskClass = 'table-warning';
        } else {
            riskClass = 'table-danger';
        }
        
        tableHTML += `
            <tr class="${riskClass}">
                <td>${item.货币对组合}</td>
                <td>${parseFloat(item.相关系数).toFixed(4)}</td>
                <td>${parseFloat(item.组合波动率).toFixed(4)}</td>
                <td>${parseFloat(item.信号一致性).toFixed(4)}</td>
                <td>${parseFloat(item.风险得分).toFixed(2)}</td>
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
}