document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM已加载，开始初始化...");
    
    // 初始化基本UI功能
    setupBasicUI();
    
    // 自动加载风险分析（如果在风险控制页面）
    const riskContainer = document.getElementById('risk-analysis-container');
    if (riskContainer) {
        console.log("检测到风险分析容器，将在页面加载完成后加载数据");
        setTimeout(function() {
            console.log("开始加载风险分析数据");
            loadRiskAnalysis();
        }, 1000);
    }
});

// 设置基本UI功能
function setupBasicUI() {
    // 初始化日期选择器
    setupDatePickers();
    
    // 初始化登录功能
    setupLoginButton();
    
    // 回测表单初始化(如果存在)
    setupTestForms();
    
    // 客服系统初始化
    setupCustomerService();
    
    // 如果是风险控制页面，自动加载数据
    const riskContainer = document.getElementById('risk-analysis-container');
    if (riskContainer) {
        console.log("自动加载风险分析数据");
        setTimeout(loadRiskAnalysis, 500);
    }
}

// 修改日期选择器设置函数
function setupDatePickers() {
    // 获取当前日期作为结束日期
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    const endDateStr = `${year}-${month}-${day}`;
    const startDateStr = '2015-01-28';
    
    console.log("设置默认日期:", startDateStr, "到", endDateStr);
    
    // 单独初始化每个日期选择器，而不是一次性初始化所有的
    
    // 回测分析日期选择器
    const startDate = document.getElementById('start-date');
    const endDate = document.getElementById('end-date');
    if (startDate) {
        flatpickr(startDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28",
            maxDate: "today",
            defaultDate: startDateStr
        });
    }
    if (endDate) {
        flatpickr(endDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28",
            maxDate: "today",
            defaultDate: endDateStr
        });
    }
    
    // 信号解释日期选择器
    const signalStartDate = document.getElementById('signal-start-date');
    const signalEndDate = document.getElementById('signal-end-date');
    if (signalStartDate) {
        flatpickr(signalStartDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28",
            maxDate: "today",
            defaultDate: startDateStr
        });
    }
    if (signalEndDate) {
        flatpickr(signalEndDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28",
            maxDate: "today",
            defaultDate: endDateStr
        });
    }
    
    // 优化策略日期选择器
    const optimizeStartDate = document.getElementById('optimize-start-date');
    const optimizeEndDate = document.getElementById('optimize-end-date');
    if (optimizeStartDate) {
        flatpickr(optimizeStartDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28",
            maxDate: "today",
            defaultDate: startDateStr
        });
    }
    if (optimizeEndDate) {
        flatpickr(optimizeEndDate, {
            dateFormat: "Y-m-d",
            locale: "zh",
            minDate: "2015-01-28", 
            maxDate: "today",
            defaultDate: endDateStr
        });
    }
}

// 设置回测表单等
function setupTestForms() {
    // 回测表单
    const backtestForm = document.getElementById('backtestForm');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            runBacktest();
        });
    }
    
    // 信号解释表单
    const signalForm = document.getElementById('signalExplainForm');
    if (signalForm) {
        signalForm.addEventListener('submit', function(e) {
            e.preventDefault();
            generateSignalExplanation();
        });
    }
    
    // 优化表单
    const optimizeForm = document.getElementById('optimizeForm');
    if (optimizeForm) {
        optimizeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            optimizeStrategy();
        });
    }
}

// 设置客服系统
function setupCustomerService() {
    const csButton = document.querySelector('.cs-button');
    if (csButton) {
        csButton.addEventListener('click', function() {
            const chatWindow = document.querySelector('.cs-chat-window');
            if (chatWindow) {
                chatWindow.style.display = chatWindow.style.display === 'flex' ? 'none' : 'flex';
            }
        });
    }
    
    const csClose = document.querySelector('.cs-close');
    if (csClose) {
        csClose.addEventListener('click', function() {
            const chatWindow = document.querySelector('.cs-chat-window');
            if (chatWindow) {
                chatWindow.style.display = 'none';
            }
        });
    }
}

// 设置登录按钮和相关事件
function setupLoginButton() {
    // 登录按钮点击事件
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
        loginBtn.addEventListener('click', function() {
            const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
            loginModal.show();
        });
    }
    
    // 充值按钮点击事件
    const rechargeBtn = document.getElementById('rechargeBtn');
    if (rechargeBtn) {
        rechargeBtn.addEventListener('click', function() {
            const rechargeModal = new bootstrap.Modal(document.getElementById('rechargeModal'));
            rechargeModal.show();
        });
    }
    
    // 注册按钮点击事件
    const showRegisterBtn = document.getElementById('showRegister');
    if (showRegisterBtn) {
        showRegisterBtn.addEventListener('click', function() {
            // 关闭登录模态框
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            if (loginModal) loginModal.hide();
            
            // 显示注册模态框
            setTimeout(function() {
                const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
                registerModal.show();
            }, 300);
        });
    }
    
    // 登录表单提交事件
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // 取得用户名和密码
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                alert('请输入用户名和密码');
                return;
            }
            
            // 关闭登录模态框
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            if (loginModal) {
                loginModal.hide();
                
                // 自动移除模态框背景
                setTimeout(function() {
                    document.body.classList.remove('modal-open');
                    const backdrop = document.querySelector('.modal-backdrop');
                    if (backdrop) backdrop.remove();
                }, 200);
            }
            
            // 显示成功消息
            setTimeout(function() {
                alert('登录成功!');
            }, 300);
        });
    }
}

// 处理退出登录 - 保留此函数仅用于兼容性
function handleLogout() {
    alert('退出登录成功!');
}

// 兼容旧代码，避免undefined错误
function logout() {
    handleLogout();
}

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

// 风险控制数据加载函数 - 改为直接从CSV硬编码
function loadRiskAnalysis() {
    console.log("执行风险分析加载函数");
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
    
    // 硬编码风险数据（直接从CSV复制）
    const riskData = [
        {
            "货币对组合": "CNYEUR-CNYGBP",
            "相关系数": "0.5815",
            "组合波动率": "0.147",
            "信号一致性": "0.0729",
            "风险得分": "29.86",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYGBP",
            "相关系数": "0.484",
            "组合波动率": "0.1664",
            "信号一致性": "0.0514",
            "风险得分": "25.89",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYEUR",
            "相关系数": "0.4643",
            "组合波动率": "0.1522",
            "信号一致性": "0.0215",
            "风险得分": "23.79",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYEUR-CNYJPY",
            "相关系数": "0.4064",
            "组合波动率": "0.1389",
            "信号一致性": "0.0555",
            "风险得分": "22.09",
            "风险等级": "中风险",
            "交易建议": "建议减小敞口"
        },
        {
            "货币对组合": "CNYGBP-CNYJPY",
            "相关系数": "0.2001",
            "组合波动率": "0.1404",
            "信号一致性": "0.0393",
            "风险得分": "13.4",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYJPY-CNYUSD",
            "相关系数": "0.2404",
            "组合波动率": "0.1078",
            "信号一致性": "-0.0045",
            "风险得分": "12.99",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYAUD-CNYJPY",
            "相关系数": "0.1955",
            "组合波动率": "0.1496",
            "信号一致性": "0.0008",
            "风险得分": "12.33",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYEUR-CNYUSD",
            "相关系数": "0.2003",
            "组合波动率": "0.0916",
            "信号一致性": "-0.034",
            "风险得分": "11.78",
            "风险等级": "低风险",
            "交易建议": "建议持有"
        },
        {
            "货币对组合": "CNYGBP-CNYUSD",
            "相关系数": "0.1026",
            "组合波动率": "0.1028",
            "信号一致性": "0.0525",
            "风险得分": "8.76",
            "风险等级": "极低风险",
            "交易建议": "可以增加敞口"
        },
        {
            "货币对组合": "CNYAUD-CNYUSD",
            "相关系数": "0.0341",
            "组合波动率": "0.1114",
            "信号一致性": "-0.011",
            "风险得分": "5.04",
            "风险等级": "极低风险",
            "交易建议": "可以增加敞口"
        }
    ];
    
    console.log("直接使用硬编码风险数据");
    
    // 直接创建并显示风险数据表格
    displayRiskAnalysis(riskData);
}

// 显示风险分析结果
function displayRiskAnalysis(riskData) {
    console.log("执行显示风险分析函数");
    const riskContainer = document.getElementById('risk-analysis-container');
    if (!riskContainer) {
        console.error("找不到风险分析容器");
        return;
    }
    
    if (!Array.isArray(riskData) || riskData.length === 0) {
        console.error("风险数据无效:", riskData);
        riskContainer.innerHTML = '<div class="alert alert-warning">没有可用的风险分析数据</div>';
        return;
    }
    
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
                <td>${item.货币对组合 || '-'}</td>
                <td>${item.相关系数 || '-'}</td>
                <td>${item.组合波动率 || '-'}</td>
                <td>${item.信号一致性 || '-'}</td>
                <td>${item.风险得分 || '-'}</td>
                <td>${item.风险等级 || '-'}</td>
                <td>${item.交易建议 || '-'}</td>
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

// 添加全局图表变量
let equityChart = null;
let drawdownChart = null;
let monthlyReturnsChart = null;

// 回测分析执行
async function runBacktest() {
    try {
        const currencyPair = document.getElementById('currency-pair').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        // 显示加载状态
        showLoading();
        
        // 发送回测请求
        const response = await fetch('/api/backtest', {
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
        console.error('回测执行错误:', error);
        showError('回测执行出错: ' + error.message);
    } finally {
        hideLoading();
    }
}

// 修改回测分析结果显示函数
function displayBacktestResults(results) {
    try {
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

        // 找到图表容器并清除其中任何存在的元素
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer) {
            // 调整容器高度以适应单个图表
            chartContainer.style.height = '500px';
        }

        // 仅保留并优化权益曲线图表
        const equityCurveCtx = document.getElementById('equity-curve');
        if (equityCurveCtx) {
            // 销毁旧图表
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
            
            // 创建新图表 - 优化配置以更好地适应容器
            equityChart = new Chart(equityCurveCtx.getContext('2d'), {
                type: 'line',
                data: {
                    datasets: [{
                        label: '账户权益',
                        data: chartData,
                        borderColor: 'rgb(30, 58, 138)', // 使用深蓝色主题色
                        backgroundColor: 'rgba(30, 58, 138, 0.1)',
                        fill: true,
                        tension: 0.2,
                        pointRadius: 1,
                        pointHoverRadius: 5,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                font: {
                                    size: 14,
                                    weight: 'bold'
                                }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(15, 41, 77, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            },
                            padding: 12,
                            cornerRadius: 6,
                            displayColors: false
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
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 12,
                                font: {
                                    size: 12
                                }
                            }
                        },
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                                font: {
                                    size: 12
                                },
                                callback: function(value) {
                                    return value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }

        // 移除回撤曲线和月度收益图表的容器
        const drawdownContainer = document.getElementById('drawdown-curve');
        if (drawdownContainer) {
            drawdownContainer.parentElement.remove();
        }

        const monthlyReturnsContainer = document.getElementById('monthly-returns');
        if (monthlyReturnsContainer) {
            monthlyReturnsContainer.parentElement.remove();
        }

    } catch (error) {
        console.error('显示回测结果错误:', error);
        showError('显示回测结果错误: ' + error.message);
    }
}

// 显示加载状态
function showLoading() {
    const resultsCard = document.querySelector('.results-card');
    if (resultsCard) {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-overlay';
        loadingDiv.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <p class="mt-2">正在分析数据...</p>
        `;
        resultsCard.appendChild(loadingDiv);
    }
}

// 隐藏加载状态
function hideLoading() {
    const loadingOverlay = document.querySelector('.loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.remove();
    }
}

// 显示错误消息
function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alert, container.firstChild);
        
        // 5秒后自动关闭
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    }
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

// 初始化用户界面
function initializeUserInterface() {
    // 检查登录状态
    checkLoginStatus();
    
    // 绑定登录按钮事件
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
        loginBtn.addEventListener('click', function() {
            console.log("点击登录按钮");
            const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
            loginModal.show();
        });
    }
    
    // 绑定登录表单提交事件
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleLogin();
        });
    }
    
    // 绑定退出登录按钮事件
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            console.log("点击退出登录按钮");
            handleLogout();
        });
    }
}

// 检查登录状态
function checkLoginStatus() {
    console.log("更新登录状态...");
    const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
    const username = localStorage.getItem('username');
    
    const guestNav = document.getElementById('guest-nav');
    const userNav = document.getElementById('user-nav');
    
    if (!guestNav || !userNav) return;
    
    if (isLoggedIn && username) {
        console.log("用户已登录:", username);
        guestNav.style.display = 'none';
        userNav.style.display = 'block';
        
        // 更新用户名显示
        const usernameEl = userNav.querySelector('.username-text');
        if (usernameEl) {
            usernameEl.textContent = username;
        }
    } else {
        console.log("用户未登录");
        guestNav.style.display = 'block';
        userNav.style.display = 'none';
    }
}

// 处理登录
function handleLogin() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        alert('请输入用户名和密码');
        return;
    }
    
    console.log("处理登录...", username);
    
    // 模拟登录成功
    localStorage.setItem('isLoggedIn', 'true');
    localStorage.setItem('username', username);
    
    // 关闭登录窗口
    const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
    if (loginModal) {
        loginModal.hide();
        
        // 移除模态框背景
        document.body.classList.remove('modal-open');
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) backdrop.remove();
    }
    
    // 更新界面
    checkLoginStatus();
    
    alert('登录成功!');
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

// 添加显示成功消息的函数
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alert, container.firstChild);
        
        // 5秒后自动关闭
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    }
}

// 显示浮动消息
function showFloatingMessage(message, type = 'success') {
    // 创建消息容器
    const messageDiv = document.createElement('div');
    messageDiv.className = `floating-message ${type}`;
    messageDiv.innerHTML = message;
    
    // 添加到页面
    document.body.appendChild(messageDiv);
    
    // 显示动画
    setTimeout(() => {
        messageDiv.classList.add('show');
    }, 100);
    
    // 5秒后移除
    setTimeout(() => {
        messageDiv.classList.remove('show');
        setTimeout(() => {
            messageDiv.remove();
        }, 300);
    }, 5000);
}

// 绑定退出登录事件
function bindLogoutEvent() {
    const logoutLink = document.querySelector('.logout-link');
    if (logoutLink) {
        logoutLink.addEventListener('click', function(e) {
            e.preventDefault();
            handleLogout();
        });
    }
}