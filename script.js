document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const initialLoader = document.getElementById('initialLoader');
    const analysisLoader = document.getElementById('analysisLoader');
    const keywordInput = document.getElementById('keyword-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const results = document.getElementById('results');
    const errorPopup = document.getElementById('errorPopup');
    let sentimentChart;

    // Initial loading animation
    function simulateInitialLoading() {
        const progress = initialLoader.querySelector('.progress');
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    initialLoader.style.opacity = '0';
                    setTimeout(() => {
                        initialLoader.style.display = 'none';
                    }, 300);
                }, 500);
            } else {
                width += 2;
                progress.style.width = width + '%';
            }
        }, 30);
    }

    // Initialize Chart
    function initChart() {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        sentimentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'SENTIMENT',
                    data: [],
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 2,
                    pointStyle: 'rect',
                    pointRadius: 5,
                    pointBorderColor: '#00ff00',
                    pointBackgroundColor: '#000000',
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#000000',
                        titleColor: '#00ff00',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff00',
                        borderWidth: 1,
                        displayColors: false,
                        titleFont: {
                            family: 'Space Mono'
                        },
                        bodyFont: {
                            family: 'Space Mono'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        min: -1,
                        max: 1,
                        grid: {
                            color: 'rgba(0, 255, 0, 0.1)',
                            borderColor: '#00ff00'
                        },
                        ticks: {
                            color: '#00ff00',
                            font: {
                                family: 'Space Mono'
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#00ff00',
                            font: {
                                family: 'Space Mono'
                            },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    }

    // Error handling
    function showError(message) {
        errorPopup.style.display = 'flex';
        document.getElementById('errorMessage').textContent = message;
    }

    window.closeError = function() {
        errorPopup.style.display = 'none';
    }

    // Update UI with results
    function updateUI(data) {
        const scoreValue = document.querySelector('.score-value');
        const scoreLabel = document.querySelector('.score-label');
        
        // Update sentiment score and label
        scoreValue.textContent = data.current_score.toFixed(2);
        if (data.current_score > 0.2) {
            scoreLabel.textContent = 'POSITIVE';
            scoreValue.style.color = '#00ff00';
        } else if (data.current_score < -0.2) {
            scoreLabel.textContent = 'NEGATIVE';
            scoreValue.style.color = '#ff0000';
        } else {
            scoreLabel.textContent = 'NEUTRAL';
            scoreValue.style.color = '#00ff00';
        }

        // Update chart
        sentimentChart.data.labels = data.timestamps;
        sentimentChart.data.datasets[0].data = data.historical_scores;
        sentimentChart.update();
        
        // Animate score value
        let startScore = 0;
        const targetScore = data.current_score;
        const duration = 1000; // 1 second
        const startTime = Date.now();
        
        function updateScore() {
            const currentTime = Date.now();
            const elapsed = currentTime - startTime;
            
            if (elapsed < duration) {
                const progress = elapsed / duration;
                const currentScore = startScore + (targetScore - startScore) * progress;
                scoreValue.textContent = currentScore.toFixed(2);
                requestAnimationFrame(updateScore);
            } else {
                scoreValue.textContent = targetScore.toFixed(2);
            }
        }
        
        updateScore();
    }

    // Simulate analysis progress
    function simulateAnalysisProgress() {
        const progress = analysisLoader.querySelector('.loader-progress');
        progress.style.animation = 'loading 1s linear infinite';
    }

    // Analyze function
    async function analyzeSentiment(keyword) {
        try {
            // You may need to adjust the URL based on your local setup
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ keyword: keyword })
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            return data;
        } catch (error) {
            throw new Error(error.message || 'Failed to analyze sentiment');
        }
    }

    // Handle analyze button click
    analyzeBtn.addEventListener('click', async () => {
        const keyword = keywordInput.value.trim();
        
        if (!keyword) {
            showError('PLEASE ENTER A KEYWORD');
            return;
        }

        // Show loading state
        analysisLoader.style.display = 'block';
        results.style.display = 'none';
        analyzeBtn.disabled = true;
        simulateAnalysisProgress();

        try {
            const data = await analyzeSentiment(keyword);
            
            // Hide loader
            analysisLoader.style.display = 'none';
            
            // Show results
            results.style.display = 'block';
            updateUI(data);

        } catch (error) {
            showError(error.message);
            analysisLoader.style.display = 'none';
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    // Allow Enter key to trigger analysis
    keywordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !analyzeBtn.disabled) {
            analyzeBtn.click();
        }
    });

    // Initialize
    simulateInitialLoading();
    initChart();
});