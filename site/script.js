// =====================================================================
// Caspian Risk - client-side CSV loader and renderer
// =====================================================================
// Reads predictions/YYYY-MM/{daily,monthly}.csv from the same origin,
// renders city cards with calendar grids. No build step required.
// =====================================================================

(function () {
  'use strict';

  const CITY_ORDER = ['Baku', 'Aktau', 'Anzali', 'Turkmenbashi', 'Makhachkala'];
  const MONTHS = ['January','February','March','April','May','June',
                  'July','August','September','October','November','December'];

  // ─── Tiny CSV parser (handles quoted fields with commas/newlines) ──
  function parseCSV(text) {
    const rows = [];
    let cur = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQuotes) {
        if (c === '"' && text[i+1] === '"') { field += '"'; i++; }
        else if (c === '"') { inQuotes = false; }
        else { field += c; }
      } else {
        if (c === '"') { inQuotes = true; }
        else if (c === ',') { cur.push(field); field = ''; }
        else if (c === '\n') { cur.push(field); rows.push(cur); cur = []; field = ''; }
        else if (c === '\r') { /* skip */ }
        else { field += c; }
      }
    }
    if (field.length > 0 || cur.length > 0) {
      cur.push(field);
      rows.push(cur);
    }
    if (rows.length === 0) return [];
    const headers = rows[0];
    return rows.slice(1).filter(r => r.length === headers.length).map(r => {
      const obj = {};
      headers.forEach((h, i) => obj[h.trim()] = r[i]);
      return obj;
    });
  }

  // ─── Helpers ──────────────────────────────────────────────────────
  function fmtMonth(ym) {
    const [y, m] = ym.split('-');
    return MONTHS[parseInt(m, 10) - 1] + ' ' + y;
  }

  function riskClass(p, source) {
    if (source === 'climatology') return 'risk-clim';
    if (p >= 0.5) return 'risk-high';
    if (p >= 0.25) return 'risk-med';
    return 'risk-low';
  }

  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') e.className = v;
      else if (k === 'style') e.style.cssText = v;
      else if (k === 'html') e.innerHTML = v;
      else e.setAttribute(k, v);
    });
    children.flat().forEach(c => {
      if (c == null) return;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return e;
  }

  // ─── Discover the latest available month ──────────────────────────
  // We try the current YYYY-MM first; if it fails, walk back month-by-month.
  // This works because Vercel returns 404 for missing files.
  async function findLatestMonth() {
    const now = new Date();
    let y = now.getUTCFullYear();
    let m = now.getUTCMonth() + 1;
    for (let i = 0; i < 6; i++) {
      const ym = `${y}-${String(m).padStart(2, '0')}`;
      try {
        const r = await fetch(`predictions/${ym}/daily.csv`, { method: 'HEAD' });
        if (r.ok) return ym;
      } catch (e) { /* CORS or network — ignore and continue */ }
      m -= 1;
      if (m === 0) { m = 12; y -= 1; }
    }
    return null;
  }

  // ─── Load and render the latest forecast ──────────────────────────
  async function loadLatest() {
    const eyebrow = document.getElementById('month-eyebrow');
    const titleEl = document.getElementById('month-title');
    const heroStats = document.getElementById('hero-stats');
    const cityGrid = document.getElementById('city-grid');

    const ym = await findLatestMonth();
    if (!ym) {
      cityGrid.appendChild(el('div', { className: 'notice error' },
        'No forecast data found. Predictions are generated monthly by GitHub Actions; check back after the 1st of the next month.'));
      return;
    }

    let dailyText, monthlyText;
    try {
      [dailyText, monthlyText] = await Promise.all([
        fetch(`predictions/${ym}/daily.csv`).then(r => r.text()),
        fetch(`predictions/${ym}/monthly.csv`).then(r => r.text()),
      ]);
    } catch (err) {
      cityGrid.appendChild(el('div', { className: 'notice error' },
        'Could not load prediction CSVs. Try refreshing.'));
      return;
    }

    const daily = parseCSV(dailyText);
    const monthly = parseCSV(monthlyText);

    // Update header
    eyebrow.textContent = `Forecast · ${fmtMonth(ym)}`;
    titleEl.textContent = `${fmtMonth(ym)} delay-risk forecast`;
    heroStats.innerHTML = '';
    renderHeroStats(heroStats, daily, monthly);

    // Render city cards
    cityGrid.innerHTML = '';
    CITY_ORDER.forEach(city => {
      const cityDays = daily.filter(d => d.city === city)
        .sort((a, b) => parseInt(a.day_of_month, 10) - parseInt(b.day_of_month, 10));
      const cityMonth = monthly.find(m => m.city === city);
      if (cityDays.length === 0) return;
      cityGrid.appendChild(renderCityCard(city, cityDays, cityMonth));
    });
  }

  function renderHeroStats(container, daily, monthly) {
    const totalDays = daily.length;
    const highRiskDays = daily.filter(d => parseFloat(d.probability) >= 0.5).length;
    const climDays = daily.filter(d => d.source === 'climatology').length;
    const highRiskCities = monthly.filter(
      m => parseFloat(m.high_risk_month_probability || 0) >= 0.5
    ).length;

    [
      ['High-risk days', highRiskDays, `of ${totalDays} city-days forecast`],
      ['High-risk months', highRiskCities, `of ${monthly.length} cities`],
      ['Forecast horizon', totalDays - climDays, 'days from real forecasts'],
      ['Beyond horizon',  climDays, 'days from climatology'],
    ].forEach(([label, value, detail]) => {
      container.appendChild(el('div', { className: 'stat' },
        el('p', { className: 'stat-label' }, label),
        el('p', { className: 'stat-value' }, String(value)),
        el('p', { className: 'stat-detail' }, detail),
      ));
    });
  }

  function renderCityCard(city, days, monthSummary) {
    const high = days.filter(d => parseFloat(d.probability) >= 0.5).length;
    const med  = days.filter(d => {
      const p = parseFloat(d.probability);
      return p >= 0.25 && p < 0.5;
    }).length;
    const monthProb = monthSummary
      ? Math.round(parseFloat(monthSummary.high_risk_month_probability) * 100)
      : null;

    const summary = el('div', { className: 'city-summary' },
      el('strong', {}, String(high)), ' high-risk · ',
      el('strong', {}, String(med)), ' elevated',
      monthProb != null
        ? el('span', {}, ` · month risk ${monthProb}%`)
        : null,
    );

    const calendar = el('div', { className: 'calendar' });
    days.forEach(d => {
      const p = parseFloat(d.probability);
      const cls = riskClass(p, d.source);
      const cell = el('div', {
        className: 'day-cell ' + cls,
        title: `Day ${d.day_of_month} · ${(p*100).toFixed(0)}% risk · ${d.source}`,
      },
        el('span', { className: 'day-num' }, d.day_of_month),
        el('span', { className: 'day-prob' }, `${Math.round(p*100)}%`),
      );
      calendar.appendChild(cell);
    });

    return el('article', { className: 'city-card' },
      el('div', { className: 'city-header' },
        el('h3', { className: 'city-name' }, city),
        summary,
      ),
      calendar,
    );
  }

  // ─── Page-specific entry point ────────────────────────────────────
  if (document.getElementById('city-grid')) {
    loadLatest();
  }

  // ─── Archive page ─────────────────────────────────────────────────
  if (document.getElementById('archive-list')) {
    loadArchive();
  }

  async function loadArchive() {
    const list = document.getElementById('archive-list');
    const found = [];
    const now = new Date();
    let y = now.getUTCFullYear();
    let m = now.getUTCMonth() + 1;
    for (let i = 0; i < 24; i++) {
      const ym = `${y}-${String(m).padStart(2, '0')}`;
      try {
        const r = await fetch(`predictions/${ym}/monthly.csv`, { method: 'HEAD' });
        if (r.ok) found.push(ym);
      } catch (e) { /* skip */ }
      m -= 1;
      if (m === 0) { m = 12; y -= 1; }
    }
    if (found.length === 0) {
      list.appendChild(el('li', { className: 'notice' },
        'No archived forecasts yet.'));
      return;
    }
    found.forEach(ym => {
      list.appendChild(el('li', { className: 'archive-item' },
        el('a', { href: `predictions/${ym}/monthly.csv` }, fmtMonth(ym)),
        el('span', { className: 'meta' }, ym),
      ));
    });
  }
})();
