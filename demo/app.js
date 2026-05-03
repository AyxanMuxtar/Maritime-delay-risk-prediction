const PREDICTIONS_BASE = "../predictions";

const dailyView = document.getElementById("dailyView");
const targetMonth = document.getElementById("targetMonth");
const cityCount = document.getElementById("cityCount");
const highRiskDays = document.getElementById("highRiskDays");
const lastUpdated = document.getElementById("lastUpdated");
const riskModal = document.getElementById("riskModal");
const riskModalContent = document.getElementById("riskModalContent");
const riskModalClose = document.getElementById("riskModalClose");

let selectedDayCell = null;

const CITY_COUNTRIES = {
  Baku: "Azerbaijan",
  Anzali: "Iran",
  Aktau: "Kazakhstan",
  Makhachkala: "Russia",
  Turkmenbashi: "Turkmenistan",
};

// UI-only risk bands. These DO NOT change model calculations.
// They only interpret the existing probability column from predictions/latest/daily.csv.
const RISK_BANDS = [
  {
    max: 0.10,
    className: "low",
    label: "Low",
    action: "Proceed normally",
  },
  {
    max: 0.25,
    className: "moderate",
    label: "Moderate",
    action: "Monitor conditions",
  },
  {
    max: 0.50,
    className: "high",
    label: "High",
    action: "Plan with caution",
  },
  {
    max: 0.75,
    className: "very-high",
    label: "Very High",
    action: "Manual review advised",
  },
  {
    max: Infinity,
    className: "severe",
    label: "Severe",
    action: "Avoid/postpone if possible",
  },
];

function cityLabel(city) {
  const country = CITY_COUNTRIES[city];
  return country ? `${city}, ${country}` : city;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map(h => h.trim());

  return lines.slice(1).filter(Boolean).map(line => {
    const values = line.split(",").map(v => v.trim());
    const row = {};

    headers.forEach((h, i) => {
      row[h] = values[i];
    });

    return row;
  });
}

async function loadText(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.text();
}

async function loadJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}

function numberFrom(row, ...keys) {
  for (const key of keys) {
    const value = Number(row[key]);
    if (!Number.isNaN(value)) return value;
  }
  return 0;
}

function probability(row) {
  return numberFrom(row, "probability", "prob", "p");
}

function portWeatherProbability(row) {
  return numberFrom(row, "port_weather_probability", "base_probability", "probability");
}

function offshoreSeaProbability(row) {
  return numberFrom(row, "offshore_sea_probability");
}

function uncalibratedMaritimeProbability(row) {
  return numberFrom(row, "uncalibrated_maritime_probability", "raw_probability", "probability");
}

function riskBand(p) {
  const value = Number(p);
  return RISK_BANDS.find(band => value < band.max) ?? RISK_BANDS[RISK_BANDS.length - 1];
}

function getRiskClass(p) {
  return riskBand(p).className;
}

function riskLabel(rowOrProbability) {
  if (typeof rowOrProbability === "object") {
    return rowOrProbability.risk_level || riskBand(probability(rowOrProbability)).label;
  }
  return riskBand(rowOrProbability).label;
}

function riskAction(rowOrProbability) {
  if (typeof rowOrProbability === "object") {
    return rowOrProbability.recommended_action || riskBand(probability(rowOrProbability)).action;
  }
  return riskBand(rowOrProbability).action;
}

function formatPercent(p) {
  return `${Math.round(Number(p) * 100)}%`;
}

function formatNumber(value, digits = 2) {
  const n = Number(value);
  if (Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function sourceLabel(row) {
  const src = (row.source || "").toLowerCase();

  if (src.includes("climatology")) return "climatology";
  if (src.includes("short")) return "ML forecast";
  if (src.includes("forecast")) return "ML forecast";

  return row.source || "model";
}

function offshoreSourceLabel(row) {
  const src = (row.offshore_source || "").toLowerCase();

  if (src.includes("marine")) return "marine forecast";
  if (src.includes("wave_climatology")) return "wave climatology";
  if (src.includes("climatology")) return "wave climatology";

  return row.offshore_source || "not available";
}

function conditionLabel(row) {
  const raw = row.main_drivers || row.risk_reason || row.reason || "";

  if (!String(raw).trim()) {
    return "No major risk condition";
  }

  return String(raw)
    .replaceAll("_", " ")
    .replace(/wind/gi, "Wind")
    .replace(/gusts?/gi, "Gusts")
    .replace(/precipitation/gi, "Precipitation")
    .replace(/rain/gi, "Rain")
    .replace(/visibility/gi, "Visibility")
    .replace(/wave/gi, "Waves")
    .replace(/offshore/gi, "Offshore")
    .replace(/\s+/g, " ")
    .trim();
}

function toISODate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseISODateLocal(value) {
  return new Date(`${value}T00:00:00`);
}

function startOfToday() {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function addDays(date, days) {
  const d = new Date(date);
  d.setDate(d.getDate() + days);
  return d;
}

function formatWindow(start, end) {
  const dateOpts = { month: "short", day: "numeric" };
  const startText = start.toLocaleDateString("en-US", dateOpts);
  const endText = end.toLocaleDateString("en-US", dateOpts);
  const yearText = end.getFullYear();

  return `
    <span class="window-dates">${startText} → ${endText}</span>
    <span class="window-year">${yearText}</span>
  `;
}

function formatCalendarMonths(start, end) {
  const sameMonth =
    start.getFullYear() === end.getFullYear() &&
    start.getMonth() === end.getMonth();

  const startMonth = start.toLocaleString("en-US", { month: "long" });
  const endMonth = end.toLocaleString("en-US", { month: "long" });

  const startYear = start.getFullYear();
  const endYear = end.getFullYear();

  if (sameMonth) return `${startMonth} ${startYear}`;
  if (startYear === endYear) return `${startMonth}–${endMonth} ${startYear}`;
  return `${startMonth} ${startYear} – ${endMonth} ${endYear}`;
}

function formatCellDay(date) {
  return date.getDate();
}

function formatGeneratedAt(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderRiskModalContent(row) {
  const finalRisk = probability(row);
  const rawAdjusted = uncalibratedMaritimeProbability(row);
  const portWeather = portWeatherProbability(row);
  const offshore = offshoreSeaProbability(row);
  const band = riskBand(finalRisk);
  const wave = row.offshore_wave_height_m;
  const rawDiff = Math.abs(rawAdjusted - finalRisk);
  const showRawScore = rawDiff >= 0.01;

  riskModalContent.innerHTML = `
    <div class="detail-header">
      <div>
        <p class="eyebrow">Selected forecast day</p>
        <h2 id="riskModalTitle">${cityLabel(row.city)} · ${row.date}</h2>
      </div>
      <div class="detail-score ${band.className}">
        <span>${riskLabel(row)}</span>
        <strong>${formatPercent(finalRisk)}</strong>
      </div>
    </div>

    <section class="action-card ${band.className}">
      <span class="label">Recommended action</span>
      <strong>${riskAction(row)}</strong>
    </section>

    <div class="risk-components ${showRawScore ? "has-raw-score" : ""}">
      <div class="component-card primary-card">
        <span class="label">Final maritime risk</span>
        <strong>${formatPercent(finalRisk)}</strong>
        <p>${riskLabel(row)}</p>
      </div>

      <div class="component-card">
        <span class="label">Port-weather risk</span>
        <strong>${formatPercent(portWeather)}</strong>
        <p>${sourceLabel(row)}</p>
      </div>

      <div class="component-card">
        <span class="label">Offshore sea-state risk</span>
        <strong>${formatPercent(offshore)}</strong>
        <p>${offshoreSourceLabel(row)}</p>
      </div>

      <div class="component-card">
        <span class="label">Wave estimate</span>
        <strong>${formatNumber(wave)} m</strong>
        <p>${row.offshore_driver || "No major offshore condition"}</p>
      </div>

      ${showRawScore ? `
        <div class="component-card muted-card">
          <span class="label">Raw adjusted score</span>
          <strong>${formatPercent(rawAdjusted)}</strong>
          <p>before calibration</p>
        </div>
      ` : ""}
    </div>

    <p class="detail-reason">
      <strong>Key conditions:</strong> ${conditionLabel(row)}.
    </p>
  `;
}

function openRiskModal(row, cell = null) {
  renderRiskModalContent(row);

  if (selectedDayCell) {
    selectedDayCell.classList.remove("selected");
  }

  selectedDayCell = cell;
  if (selectedDayCell) {
    selectedDayCell.classList.add("selected");
  }

  riskModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  riskModalClose.focus();
}

function closeRiskModal() {
  riskModal.classList.add("hidden");
  document.body.classList.remove("modal-open");

  if (selectedDayCell) {
    selectedDayCell.classList.remove("selected");
    selectedDayCell = null;
  }
}

function buildCalendar(city, rows, startDate, endDate) {
  const firstDayIndex = (startDate.getDay() + 6) % 7;

  const rowByDate = new Map();
  rows.forEach(row => {
    rowByDate.set(row.date, row);
  });

  const card = document.createElement("article");
  card.className = "city-card";

  const highDays = rows.filter(r => probability(r) >= 0.25).length;
  const veryHighDays = rows.filter(r => probability(r) >= 0.50).length;
  const avgProb = rows.length
    ? rows.reduce((sum, r) => sum + probability(r), 0) / rows.length
    : 0;

  card.innerHTML = `
    <div class="city-header">
      <div class="city-title-row">
        <h2 class="city-name">${cityLabel(city)}</h2>
        <h2 class="city-months">${formatCalendarMonths(startDate, endDate)}</h2>
      </div>

      <div class="city-stats">
        <strong>${highDays}</strong> high+ days ·
        <strong>${veryHighDays}</strong> very-high+ ·
        avg risk ${formatPercent(avgProb)}
      </div>
    </div>
  `;

  const calendar = document.createElement("div");
  calendar.className = "calendar";

  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].forEach(day => {
    const el = document.createElement("div");
    el.className = "weekday";
    el.textContent = day;
    calendar.appendChild(el);
  });

  for (let i = 0; i < firstDayIndex; i++) {
    const empty = document.createElement("div");
    empty.className = "day empty";
    calendar.appendChild(empty);
  }

  for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
    const iso = toISODate(d);
    const row = rowByDate.get(iso);
    const cell = document.createElement("div");
    const displayDay = formatCellDay(d);

    if (!row) {
      cell.className = "day";
      cell.innerHTML = `
        <span class="day-number">${displayDay}</span>
        <div class="day-main">
          <span class="prob">—</span>
          <span class="source">no data</span>
        </div>
      `;
    } else {
      const p = probability(row);
      const band = riskBand(p);
      const isClim = (row.source || "").toLowerCase().includes("climatology");
      const offshore = offshoreSeaProbability(row);

      cell.className = `day ${band.className} ${isClim ? "climatology" : ""}`;
      cell.title = `${city} ${row.date}: ${band.label} ${formatPercent(p)} · port ${formatPercent(portWeatherProbability(row))} · offshore ${formatPercent(offshore)} — ${conditionLabel(row)}`;

      cell.innerHTML = `
        <span class="day-number">${displayDay}</span>
        <div class="day-main">
          <span class="prob">${formatPercent(p)}</span>
          <span class="risk-word">${band.label}</span>
          <span class="source">${sourceLabel(row)}</span>
        </div>
      `;

      cell.addEventListener("click", () => openRiskModal(row, cell));
      cell.tabIndex = 0;
      cell.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openRiskModal(row, cell);
        }
      });
    }

    calendar.appendChild(cell);
  }

  card.appendChild(calendar);
  return card;
}

function renderDaily(rows, startDate, endDate) {
  dailyView.innerHTML = "";

  if (!rows.length || !("date" in rows[0])) {
    dailyView.innerHTML = `
      <div class="city-card">
        <h2>No daily forecast found</h2>
        <p class="subtitle">The demo could not find daily.csv with a date column.</p>
      </div>
    `;
    return [];
  }

  const startISO = toISODate(startDate);
  const endISO = toISODate(endDate);

  const windowRows = rows.filter(row => {
    return row.date >= startISO && row.date <= endISO;
  });

  if (!windowRows.length) {
    dailyView.innerHTML = `
      <div class="city-card">
        <h2>No rows found for this forecast window</h2>
        <p class="subtitle">The daily prediction file loaded, but it does not contain dates from ${startISO} to ${endISO}.</p>
      </div>
    `;
    return [];
  }

  const cities = [...new Set(windowRows.map(r => r.city))].sort();

  cities.forEach(city => {
    const cityRows = windowRows
      .filter(r => r.city === city)
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    dailyView.appendChild(buildCalendar(city, cityRows, startDate, endDate));
  });

  return windowRows;
}

async function init() {
  try {
    const latest = await loadJSON(`${PREDICTIONS_BASE}/latest.json`);

    let startDate;
    let endDate;

    if (latest.window_start && latest.window_end) {
      startDate = parseISODateLocal(latest.window_start);
      endDate = parseISODateLocal(latest.window_end);
    } else {
      startDate = startOfToday();
      endDate = addDays(startDate, 30);
    }

    targetMonth.innerHTML = formatWindow(startDate, endDate);
    lastUpdated.textContent = formatGeneratedAt(latest.generated_at);

    const dailyText = await loadText(`${PREDICTIONS_BASE}/latest/daily.csv`);
    const dailyRows = parseCSV(dailyText);

    const visibleRows = renderDaily(dailyRows, startDate, endDate);

    const cities = new Set(visibleRows.map(r => r.city));
    const highDays = visibleRows.filter(r => probability(r) >= 0.25).length;

    cityCount.textContent = cities.size;
    highRiskDays.textContent = highDays;

  } catch (err) {
    console.error(err);

    dailyView.innerHTML = `
      <div class="city-card">
        <h2>Could not load prediction files</h2>
        <p class="subtitle">
          Make sure <code>predictions/latest.json</code> and
          <code>predictions/latest/daily.csv</code> exist.
        </p>
        <p class="subtitle">${err.message}</p>
      </div>
    `;
  }
}

riskModalClose.addEventListener("click", closeRiskModal);
riskModal.addEventListener("click", event => {
  if (event.target.dataset.modalClose !== undefined) {
    closeRiskModal();
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !riskModal.classList.contains("hidden")) {
    closeRiskModal();
  }
});

init();
