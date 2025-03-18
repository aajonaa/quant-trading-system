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
});

// 事件监听设置
function setupEventListeners() {
    // 回测表单提交
    document.getElementById('backtestForm').addEventListener('submit', function(e) {
        e.preventDefault();
        runBacktest();
    });

    // 客服按钮点击
    const csButton = document.querySelector('.cs-button');
    if (csButton) {
        csButton.addEventListener('click', toggleCustomerService);
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
}

// 回测功能实现
async function runBacktest() {
    const currencyPair = document.getElementById('currency-pair').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    showLoading('正在进行回测分析...');

    try {
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
            updateBacktestResults(data.results);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('回测过程发生错误');
        console.error(error);
    } finally {
        hideLoading();
    }
}

// 更新回测结果
function updateBacktestResults(results) {
    // 更新指标卡片
    const metricsRow = document.querySelector('.metrics-row');
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

    // 绘制权益曲线
    drawEquityCurve(results.equity_curve);
}

// 绘制权益曲线
function drawEquityCurve(data) {
    const ctx = document.getElementById('equity-curve').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: '策略收益',
                data: data.values,
                borderColor: '#1890ff',
                backgroundColor: 'rgba(24,144,255,0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '策略收益曲线'
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '收益率(%)'
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

// 参数优化
async function optimizeStrategy() {
    const currencyPair = document.getElementById('currency-pair').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    showLoading('正在优化策略参数...');

    try {
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
            showError(data.error);
        }
    } catch (error) {
        showError('优化过程发生错误');
        console.error(error);
    } finally {
        hideLoading();
    }
}

// 显示风险分析结果
function displayRiskAnalysis(riskData) {
    const riskPanel = document.querySelector('.risk-analysis-panel');
    let html = '<div class="table-responsive"><table class="table table-striped">';
    html += `<thead>
        <tr>
            <th>货币对组合</th>
            <th>相关系数</th>
            <th>组合波动率</th>
            <th>信号一致性</th>
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
            <td>${item.风险等级}</td>
            <td>${item.交易建议}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    riskPanel.innerHTML = html;
}

// 显示优化结果
function displayOptimizationResults(data) {
    const optimizationPanel = document.querySelector('.optimization-panel');
    let html = '<div class="optimization-results">';
    
    // 显示最优参数
    html += `<div class="best-params">
        <h4>最优参数组合</h4>
        <div class="params-grid">
            <div class="param-item">
                <div class="param-label">短期均线</div>
                <div class="param-value">${data.best_solution.parameters.ma_short}</div>
            </div>
            <div class="param-item">
                <div class="param-label">长期均线</div>
                <div class="param-value">${data.best_solution.parameters.ma_long}</div>
            </div>
            <div class="param-item">
                <div class="param-label">RSI周期</div>
                <div class="param-value">${data.best_solution.parameters.rsi_period}</div>
            </div>
            <div class="param-item">
                <div class="param-label">止损比例</div>
                <div class="param-value">${(data.best_solution.parameters.stop_loss * 100).toFixed(2)}%</div>
            </div>
        </div>
    </div>`;

    // 显示优化效果对比
    html += `<div class="optimization-comparison mt-4">
        <h4>优化效果对比</h4>
        <div class="comparison-chart">
            <canvas id="comparisonChart"></canvas>
        </div>
    </div>`;

    optimizationPanel.innerHTML = html;

    // 绘制对比图表
    drawComparisonChart(data);
}

// 客服功能
function toggleCustomerService() {
    const chatWindow = document.querySelector('.cs-chat-window');
    chatWindow.style.display = chatWindow.style.display === 'none' ? 'block' : 'none';
}

function sendCustomerMessage() {
    const input = document.getElementById('csInput');
    const message = input.value.trim();
    if (!message) return;

    const messagesDiv = document.getElementById('csMessages');
    messagesDiv.innerHTML += `
        <div class="cs-message user">
            <div class="message-content">${message}</div>
        </div>
    `;

    // 模拟客服回复
    setTimeout(() => {
        messagesDiv.innerHTML += `
            <div class="cs-message service">
                <div class="message-content">感谢您的咨询，我们的客服人员将尽快回复您。</div>
            </div>
        `;
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }, 1000);

    input.value = '';
}

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