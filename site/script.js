// =====================================================================
// Caspian Risk — client-side CSV loader and renderer
// =====================================================================

(function () {
  'use strict';

  // --- Constants & Config ---
  const CITY_ORDER = ['Baku', 'Aktau', 'Anzali', 'Turkmenbashi', 'Makhachkala'];
  const MONTHS = ['January','February','March','April','May','June',
                  'July','August','September','October','November','December'];

  // --- Path Mappings (Matched to your GitHub Action output) ---
  const PATHS = {
    latestJson: 'predictions/latest.json',
    dailyCsv: (ym) => `predictions/${ym}/daily.csv`,
    monthlyCsv: (ym) => `predictions/${ym}/monthly.csv`
  };

  // ─── Tiny CSV parser ──────────────────────────────────────────────
  function parseCSV(text) {
    const rows = [];
    let cur = [], field = '', inQuotes = false;
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
        else if (c === '\r') {}
        else { field += c; }
      }
    }
    if (field.length > 0 || cur.length > 0) { cur.push(field); rows.push(cur); }
    if (rows.length === 0) return [];
    const headers = rows[0];
    return rows.slice(1)
      .filter(r => r.length === headers.length && r.some(v => v.trim() !== ''))
      .map(r => {
        const obj = {};
        headers.forEach((h, i) => obj[h.trim()] = (r[i] || '').trim());
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
    if (p >= 0.5)  return 'risk-high';
    if (p >= 0.25) return 'risk-med';
    return 'risk-low';
  }

  function el(tag, attrs, ...children) {
    attrs = attrs || {};
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') e.className = v;
      else if (k === 'style') e.style.cssText = v;
      else e.setAttribute(k, v);
    });
    children.flat().forEach(c => {
      if (c == null) return;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return e;
  }

  // ─── Find the latest month ─────────────────────────────────────────
  async function findLatestMonth() {
    // Primary path: Read from the root predictions/latest.json
    try {
      const r = await fetch(PATHS.latestJson, { cache: 'no-store' });
      if (r.ok) {
        const data = await r.json();
        if (data && data.month && /^\d{4}-\d{2}$/.test(data.month)) {
          return data.month; // Expected output format: "2026-05"
        }
      }
    } catch (e) {
      console.warn("Could not load latest.json, falling back to manual probe.");
    }

    // Fallback: probe month-by-month if latest.json fails
    const now = new Date();
    const probe = [];

    let y = now.getUTCFullYear();
    let m = now.getUTCMonth() + 2; 
    if (m > 12) { m -= 12; y += 1; }
    probe.push(`${y}-${String(m).padStart(2, '0')}`);

    y = now.getUTCFullYear();
    m = now.getUTCMonth() + 1;
    for (let i = 0; i < 6; i++) {
      probe.push(`${y}-${String(m).padStart(2, '0')}`);
      m -= 1;
      if (m === 0) { m = 12; y -= 1; }
    }

    for (const ym of probe) {
      try {
        const r = await fetch(PATHS.dailyCsv(ym), { method: 'HEAD', cache: 'no-store' });
        if (r.ok) return ym;
      } catch (e) { }
    }
    return null;
  }

  // ─── Render the forecast ──────────────────────────────────────────
  async function loadLatest() {
    const eyebrow   = document.getElementById('month-eyebrow');
    const titleEl   = document.getElementById('month-title');
    const heroStats = document.getElementById('hero-stats');
    const cityGrid  = document.getElementById('city-grid');
    
    if (!cityGrid) return;

    const ym = await findLatestMonth();

    if (!ym) {
      cityGrid.innerHTML = '';
      cityGrid.appendChild(el('div', { className: 'notice error' },
        'No forecast data found. ',
        el('br', {}),
        'Make sure predictions/latest.json and predictions/YYYY-MM/daily.csv exist.'
      ));
      return;
    }

    // Fetch daily and monthly CSVs concurrently
    let dailyText, monthlyText;
    try {
      const [dr, mr] = await Promise.all([
        fetch(PATHS.dailyCsv(ym),   { cache: 'no-store' }),
        fetch(PATHS.monthlyCsv(ym), { cache: 'no-store' }),
      ]);
      if (!dr.ok) throw new Error(`daily.csv returned HTTP ${dr.status}`);
      if (!mr.ok) throw new Error(`monthly.csv returned HTTP ${mr.status}`);
      [dailyText, monthlyText] = await Promise.all([dr.text(), mr.text()]);
    } catch (err) {
      cityGrid.innerHTML = '';
      cityGrid.appendChild(el('div', { className: 'notice error' },
        `Could not load forecast for ${ym}: ${err.message}`));
      return;
    }

    const daily   = parseCSV(dailyText);
    const monthly = parseCSV(monthlyText);

    if (daily.length === 0) {
      cityGrid.innerHTML = '';
      cityGrid.appendChild(el('div', { className: 'notice error' },
        `Loaded daily.csv for ${ym} but it parsed to 0 rows. Check the file.`));
      return;
    }

    // Render Header
    if (eyebrow)   eyebrow.textContent  = `Forecast · ${fmtMonth(ym)}`;
    if (titleEl)   titleEl.textContent  = `${fmtMonth(ym)} delay-risk forecast`;
    if (heroStats) renderHeroStats(heroStats, daily, monthly);

    // Render City cards
    cityGrid.innerHTML = '';
    CITY_ORDER.forEach(city => {
      const cityDays = daily
        .filter(d => d.city === city)
        .sort((a, b) => parseInt(a.day_of_month, 10) - parseInt(b.day_of_month, 10));
      const cityMonth = monthly.find(row => row.city === city);
      
      if (cityDays.length === 0) return;
      cityGrid.appendChild(renderCityCard(city, cityDays, cityMonth));
    });
  }

  function renderHeroStats(container, daily, monthly) {
    container.innerHTML = '';
    const totalDays      = daily.length;
    const highRiskDays   = daily.filter(d => parseFloat(d.probability) >= 0.5).length;
    const climDays       = daily.filter(d => d.source === 'climatology').length;
    const highRiskCities = monthly.filter(
      m => parseFloat(m.high_risk_month_probability || 0) >= 0.5
    ).length;

    [
      ['High-risk days',   highRiskDays,         `of ${totalDays} city-days`],
      ['High-risk months', highRiskCities,         `of ${monthly.length} cities`],
      ['Model forecast',   totalDays - climDays, 'days with real weather data'],
      ['Climatology',      climDays,             'days from historical averages'],
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
      ? Math.round(parseFloat(monthSummary.high_risk_month_probability || 0) * 100)
      : null;

    const summary = el('div', { className: 'city-summary' });
    summary.innerHTML =
      `<strong>${high}</strong> high-risk &nbsp;·&nbsp; <strong>${med}</strong> elevated` +
      (monthProb != null ? ` &nbsp;·&nbsp; month risk ${monthProb}%` : '');

    const calendar = el('div', { className: 'calendar' });
    days.forEach(d => {
      const p   = parseFloat(d.probability);
      const cls = riskClass(p, d.source);
      calendar.appendChild(el('div', {
        className: `day-cell ${cls}`,
        title: `Day ${d.day_of_month} · ${Math.round(p*100)}% · ${d.source}`,
      },
        el('span', { className: 'day-num' }, String(d.day_of_month)),
        el('span', { className: 'day-prob' }, `${Math.round(p*100)}%`),
      ));
    });

    return el('article', { className: 'city-card' },
      el('div', { className: 'city-header' },
        el('h3', { className: 'city-name' }, city),
        summary,
      ),
      calendar,
    );
  }

  // ─── Archive page ──────────────────────────────────────────────────
  async function loadArchive() {
    const list = document.getElementById('archive-list');
    if (!list) return;

    let anchorMonth = null;
    try {
      const r = await fetch(PATHS.latestJson, { cache: 'no-store' });
      if (r.ok) {
        const data = await r.json();
        if (data && data.month) anchorMonth = data.month;
      }
    } catch (e) {}

    const found = [];
    const now = new Date();
    const probe = [];
    let y = now.getUTCFullYear();
    let m = now.getUTCMonth() + 2;
    if (m > 12) { m -= 12; y += 1; }
    probe.push(`${y}-${String(m).padStart(2, '0')}`);
    
    y = now.getUTCFullYear();
    m = now.getUTCMonth() + 1;
    for (let i = 0; i < 24; i++) {
      probe.push(`${y}-${String(m).padStart(2, '0')}`);
      m -= 1;
      if (m === 0) { m = 12; y -= 1; }
    }

    list.innerHTML = '<li class="notice">Scanning for forecasts…</li>';

    for (const ym of probe) {
      try {
        const r = await fetch(PATHS.monthlyCsv(ym), { method: 'HEAD', cache: 'no-store' });
        if (r.ok) found.push(ym);
      } catch (e) {}
    }

    list.innerHTML = '';
    if (found.length === 0) {
      list.appendChild(el('li', { className: 'notice' }, 'No archived forecasts yet.'));
      return;
    }
    
    found.forEach(ym => {
      const isCurrent = anchorMonth && ym === anchorMonth;
      const link = el('a', { href: PATHS.monthlyCsv(ym) },
        fmtMonth(ym), isCurrent ? ' (current)' : '');
      list.appendChild(el('li', { className: 'archive-item' },
        link,
        el('span', { className: 'meta' }, ym),
      ));
    });
  }

  // ─── Entry points ──────────────────────────────────────────────────
  if (document.getElementById('city-grid'))    loadLatest();
  if (document.getElementById('archive-list')) loadArchive();

})();