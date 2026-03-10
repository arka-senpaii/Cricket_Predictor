document.addEventListener('DOMContentLoaded', () => {
    // ── Elements ──────────────────────────────────────────────────────────
    const t1Sel = document.getElementById('team1');
    const t2Sel = document.getElementById('team2');
    const tossWinSel = document.getElementById('toss_winner');
    const flagT1 = document.getElementById('flag-t1');
    const flagT2 = document.getElementById('flag-t2');
    const resultsSection = document.getElementById('results-section');
    const h2hSection = document.getElementById('h2h-section');

    // ── Chart instances ──────────────────────────────────────────────────
    let winProbChart = null;
    let scoresChart = null;
    let radarChart = null;

    // ── Flag helpers ────────────────────────────────────────────────────
    function getFlag(team) {
        return TEAM_FLAGS[team] || 'https://flagcdn.com/h40/un.png';
    }

    function updateFlagImg(imgEl, team) {
        imgEl.src = getFlag(team);
        imgEl.style.display = 'inline-block';
    }

    // ── Toss winner sync ───────────────────────────────────────────────
    function syncTossOptions() {
        const t1 = t1Sel.value, t2 = t2Sel.value;
        const cur = tossWinSel.value;
        tossWinSel.innerHTML = '';
        [t1, t2].forEach(t => {
            const opt = new Option(t, t);
            if (t === cur) opt.selected = true;
            tossWinSel.add(opt);
        });
    }

    t1Sel.addEventListener('change', () => {
        updateFlagImg(flagT1, t1Sel.value);
        syncTossOptions();
        fetchH2H();
    });
    t2Sel.addEventListener('change', () => {
        updateFlagImg(flagT2, t2Sel.value);
        syncTossOptions();
        fetchH2H();
    });

    // Init — ensure Team 2 defaults to a different team than Team 1
    if (t2Sel.options.length > 1) {
        t2Sel.selectedIndex = 1;
    }
    updateFlagImg(flagT1, t1Sel.value);
    updateFlagImg(flagT2, t2Sel.value);
    syncTossOptions();

    // ── Form submit ────────────────────────────────────────────────────
    const form = document.getElementById('predictor-form');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const payload = {
            team1: t1Sel.value,
            team2: t2Sel.value,
            venue: document.getElementById('venue').value,
            toss_winner: tossWinSel.value,
            toss_decision: document.getElementById('toss_decision').value
        };

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.error) { alert('Error: ' + data.error); return; }
            renderResults(data, payload.team1, payload.team2);
        } catch (err) {
            alert('Failed to get prediction. Is the server running?');
        } finally {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    // ── Render results ────────────────────────────────────────────────
    function renderResults(d, t1, t2) {
        // Winner
        document.getElementById('predicted-winner').textContent = d.winner;
        document.getElementById('winner-flag').src = d.winner_flag;
        document.getElementById('winner-flag').style.display = 'inline-block';

        // Probability bar
        const winnerIsT1 = (d.winner === t1);
        const t1Prob = winnerIsT1 ? d.probability : d.loser_probability;
        const t2Prob = winnerIsT1 ? d.loser_probability : d.probability;

        document.getElementById('t1-prob-label').textContent = t1;
        document.getElementById('t2-prob-label').textContent = t2;
        document.getElementById('prob-t1-value').textContent = t1Prob + '%';
        document.getElementById('prob-t2-value').textContent = t2Prob + '%';

        requestAnimationFrame(() => {
            document.getElementById('t1-fill').style.width = t1Prob + '%';
            document.getElementById('t2-fill').style.width = t2Prob + '%';
        });

        // Scorecards
        document.getElementById('sc-flag1').src = d.innings1.flag || getFlag(d.innings1.team);
        document.getElementById('sc-flag1').style.display = 'inline-block';
        document.getElementById('sc-t1').textContent = d.innings1.team;

        document.getElementById('sc-flag2').src = d.innings2.flag || getFlag(d.innings2.team);
        document.getElementById('sc-flag2').style.display = 'inline-block';
        document.getElementById('sc-t2').textContent = d.innings2.team;

        // Animate numbers — only if real data exists
        const t1RunsEl = document.getElementById('t1-runs');
        const t1WktsEl = document.getElementById('t1-wkts');
        const t2RunsEl = document.getElementById('t2-runs');
        const t2WktsEl = document.getElementById('t2-wkts');

        if (d.innings1.runs != null) {
            countUp(t1RunsEl, d.innings1.runs, 1400);
        } else {
            t1RunsEl.textContent = '—';
        }
        if (d.innings1.wickets != null) {
            countUp(t1WktsEl, d.innings1.wickets, 900);
        } else {
            t1WktsEl.textContent = '—';
        }
        if (d.innings2.runs != null) {
            countUp(t2RunsEl, d.innings2.runs, 1400);
        } else {
            t2RunsEl.textContent = '—';
        }
        if (d.innings2.wickets != null) {
            countUp(t2WktsEl, d.innings2.wickets, 900);
        } else {
            t2WktsEl.textContent = '—';
        }

        // Update sub-label with innings count
        const t1cnt = d.innings1.innings_count || 0;
        const t2cnt = d.innings2.innings_count || 0;
        document.getElementById('sc-t1-tag').textContent =
            t1cnt > 0 ? `Historical Avg · ${t1cnt} innings` : 'No data';
        document.getElementById('sc-t2-tag').textContent =
            t2cnt > 0 ? `Historical Avg · ${t2cnt} innings` : 'No data';

        resultsSection.classList.remove('hidden');
        resultsSection.style.animation = 'none';
        void resultsSection.offsetWidth;
        resultsSection.style.animation = '';

        // Render charts
        renderCharts(d, t1, t2, t1Prob, t2Prob);
    }

    // ── Charts ────────────────────────────────────────────────────────
    function renderCharts(d, t1, t2, t1Prob, t2Prob) {
        const chartFont = { family: "'Outfit', sans-serif" };
        const gridColor = 'rgba(255,255,255,0.06)';
        const tickColor = '#6b7280';

        // Destroy previous charts
        if (winProbChart) winProbChart.destroy();
        if (scoresChart) scoresChart.destroy();
        if (radarChart) radarChart.destroy();

        // 1. Win Probability Doughnut
        const ctxWin = document.getElementById('chart-win-prob').getContext('2d');
        winProbChart = new Chart(ctxWin, {
            type: 'doughnut',
            data: {
                labels: [t1, t2],
                datasets: [{
                    data: [t1Prob, t2Prob],
                    backgroundColor: [
                        'rgba(79, 142, 247, 0.85)',
                        'rgba(168, 85, 247, 0.85)',
                    ],
                    borderColor: [
                        'rgba(79, 142, 247, 1)',
                        'rgba(168, 85, 247, 1)',
                    ],
                    borderWidth: 2,
                    hoverOffset: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#e8eaf0', font: { ...chartFont, size: 12 }, padding: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${ctx.parsed}%`
                        }
                    }
                },
                animation: { animateRotate: true, duration: 1200 }
            }
        });

        // 2. Historical Avg Scores Bar Chart
        const ctxScores = document.getElementById('chart-scores').getContext('2d');
        scoresChart = new Chart(ctxScores, {
            type: 'bar',
            data: {
                labels: [t1, t2],
                datasets: [
                    {
                        label: 'Avg Runs',
                        data: [d.innings1.runs ?? 0, d.innings2.runs ?? 0],
                        backgroundColor: ['rgba(79, 142, 247, 0.7)', 'rgba(168, 85, 247, 0.7)'],
                        borderColor: ['rgba(79, 142, 247, 1)', 'rgba(168, 85, 247, 1)'],
                        borderWidth: 2,
                        borderRadius: 8,
                        barPercentage: 0.6,
                    },
                    {
                        label: 'Avg Wickets',
                        data: [d.innings1.wickets ?? 0, d.innings2.wickets ?? 0],
                        backgroundColor: ['rgba(239, 68, 68, 0.5)', 'rgba(245, 158, 11, 0.5)'],
                        borderColor: ['rgba(239, 68, 68, 1)', 'rgba(245, 158, 11, 1)'],
                        borderWidth: 2,
                        borderRadius: 8,
                        barPercentage: 0.6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e8eaf0', font: { ...chartFont, size: 12 } }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: tickColor, font: chartFont }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#e8eaf0', font: { ...chartFont, weight: 600 } }
                    }
                },
                animation: { duration: 1000 }
            }
        });

        // 3. Radar Chart — Team Comparison
        const stats = d.stats || {};
        const ctxRadar = document.getElementById('chart-radar').getContext('2d');
        radarChart = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['ELO Rating', 'Form (5)', 'Form (10)', 'Win Prob', 'H2H'],
                datasets: [
                    {
                        label: t1,
                        data: [
                            normalizeElo(stats.elo_team1 || 1500),
                            stats.form5_team1 || 50,
                            stats.form10_team1 || 50,
                            t1Prob,
                            stats.h2h_pct || 50,
                        ],
                        backgroundColor: 'rgba(79, 142, 247, 0.2)',
                        borderColor: 'rgba(79, 142, 247, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(79, 142, 247, 1)',
                        pointRadius: 4,
                    },
                    {
                        label: t2,
                        data: [
                            normalizeElo(stats.elo_team2 || 1500),
                            stats.form5_team2 || 50,
                            stats.form10_team2 || 50,
                            t2Prob,
                            100 - (stats.h2h_pct || 50),
                        ],
                        backgroundColor: 'rgba(168, 85, 247, 0.2)',
                        borderColor: 'rgba(168, 85, 247, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(168, 85, 247, 1)',
                        pointRadius: 4,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#e8eaf0', font: { ...chartFont, size: 12 }, padding: 16 }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { display: false, stepSize: 25 },
                        grid: { color: 'rgba(255,255,255,0.08)' },
                        angleLines: { color: 'rgba(255,255,255,0.08)' },
                        pointLabels: { color: '#9ca3af', font: { ...chartFont, size: 11, weight: 600 } }
                    }
                },
                animation: { duration: 1200 }
            }
        });
    }

    // Normalize ELO (typically 1200-1800) to 0-100 scale for radar
    function normalizeElo(elo) {
        return Math.min(100, Math.max(0, ((elo - 1200) / 600) * 100));
    }

    // ── Animated counter ──────────────────────────────────────────────
    function countUp(el, target, duration) {
        let start = null;
        const step = (ts) => {
            if (!start) start = ts;
            const progress = Math.min((ts - start) / duration, 1);
            el.textContent = Math.floor(easeOut(progress) * target);
            if (progress < 1) requestAnimationFrame(step);
            else el.textContent = target;
        };
        requestAnimationFrame(step);
    }
    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    // ── H2H history ──────────────────────────────────────────────────
    let h2hTimeout;
    function fetchH2H() {
        clearTimeout(h2hTimeout);
        h2hTimeout = setTimeout(async () => {
            const t1 = t1Sel.value, t2 = t2Sel.value;
            if (!t1 || !t2 || t1 === t2) return;

            try {
                const res = await fetch('/history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ team1: t1, team2: t2 })
                });
                const data = await res.json();
                renderH2H(data);
            } catch (e) { /* silent */ }
        }, 400);
    }

    function renderH2H(data) {
        const { matches, team1, team2 } = data;
        if (!matches || matches.length === 0) {
            h2hSection.classList.add('hidden');
            return;
        }

        // H2H Header flags
        document.getElementById('h2h-flag1').src = getFlag(team1);
        document.getElementById('h2h-flag1').style.display = 'inline-block';
        document.getElementById('h2h-t1-name').textContent = team1;

        document.getElementById('h2h-flag2').src = getFlag(team2);
        document.getElementById('h2h-flag2').style.display = 'inline-block';
        document.getElementById('h2h-t2-name').textContent = team2;

        // Win counts
        let t1wins = 0, t2wins = 0;
        matches.forEach(m => {
            if (m.winner === team1) t1wins++;
            else if (m.winner === team2) t2wins++;
        });
        document.getElementById('h2h-t1-wins').textContent = t1wins;
        document.getElementById('h2h-t2-wins').textContent = t2wins;

        // Match cards
        const list = document.getElementById('match-list');
        list.innerHTML = '';
        matches.forEach(m => {
            const card = document.createElement('div');
            card.className = 'match-card';
            card.innerHTML = `
        <div class="match-meta">
          <span class="match-date">${m.date}</span>
          <span class="tournament-tag">${m.tournament}</span>
        </div>
        <div class="match-body">
          <div class="match-score-row">
            <img src="${m.batting_first_flag}" alt="" class="mc-flag" onerror="this.style.display='none'"/>
            <span>${m.batting_first}</span>
            <span class="score-val">${m.score1}</span>
            <span style="color:var(--muted);font-size:0.78rem">vs</span>
            <img src="${m.chasing_flag}" alt="" class="mc-flag" onerror="this.style.display='none'"/>
            <span>${m.chasing_team}</span>
            <span class="score-val">${m.score2}</span>
          </div>
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <img src="${m.winner_flag}" alt="" class="mc-flag" onerror="this.style.display='none'"/>
            <span class="winner-badge">🏆 ${m.winner}</span>
            <span class="tournament-tag" style="flex:1;text-align:right;">${m.venue.slice(0, 35)}${m.venue.length > 35 ? '…' : ''}</span>
          </div>
        </div>`;
            list.appendChild(card);
        });

        h2hSection.classList.remove('hidden');
        h2hSection.style.animation = 'none';
        void h2hSection.offsetWidth;
        h2hSection.style.animation = '';
    }

    // Load H2H on startup
    fetchH2H();
});
