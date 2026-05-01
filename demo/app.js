const PREDICTIONS_BASE = "../predictions";

const dailyView = document.getElementById("dailyView");
const monthlyView = document.getElementById("monthlyView");
const monthlyTable = document.getElementById("monthlyTable");

const dailyBtn = document.getElementById("dailyBtn");
const monthlyBtn = document.getElementById("monthlyBtn");

dailyBtn.addEventListener("click", () => {
  dailyBtn.classList.add("active");
  monthlyBtn.classList.remove("active");
  dailyView.classList.remove("hidden");
  monthlyView.classList.add("hidden");
});

monthlyBtn.addEventListener("click", () => {
  monthlyBtn.classList.add("active");
  dailyBtn.classList.remove("active");
  monthlyView.classList.remove("hidden");
  dailyView.classList.add("hidden");
});

const CITY_COUNTRIES = {
  Baku: "Azerbaijan",
  Anzali: "Iran",
  Aktau: "Kazakhstan",
  Makhachkala: "Russia",
  Turkmenbashi: "Turkmenistan",
};

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

  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }

  return response.text();
}

async function loadJSON(path) {
  const response = await fetch(path, { cache: "no-store" });

  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }

  return response.json();
}

function probability(row) {
  return Number(row.probability ?? row.prob ?? row.p ?? 0);
}

function summaryProbability(row) {
  return Number(
    row.high_risk_window_probability ??
    row.high_risk_month_probability ??
    row.probability ??
    row.prob ??
    row.p ??
    0
  );
}

function getRiskClass(p) {
  if (p >= 0.25) return "high";
  if (p >= 0.10) return "elevated";
  return "low";
}

function formatPercent(p) {
  return `${Math.round(p * 100)}%`;
}

function sourceLabel(row) {
  const src = (row.source || "").toLowerCase();

  if (src.includes("climatology")) return "climatology";
  if (src.includes("short")) return "ML forecast";
  if (src.includes("forecast")) return "ML forecast";

  return row.source || "model";
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

  if (sameMonth) {
    return `${startMonth} ${startYear}`;
  }

  if (startYear === endYear) {
    return `${startMonth}–${endMonth} ${startYear}`;
  }

  return `${startMonth} ${startYear} – ${endMonth} ${endYear}`;
}

function formatCellDay(date) {
  return date.getDate();
}

function buildCalendar(city, rows, startDate, endDate) {
  const firstDayIndex = (startDate.getDay() + 6) % 7; // Monday first

  const rowByDate = new Map();

  rows.forEach(row => {
    rowByDate.set(row.date, row);
  });

  const card = document.createElement("article");
  card.className = "city-card";

  const highDays = rows.filter(r => probability(r) >= 0.25).length;
  const elevatedDays = rows.filter(r => probability(r) >= 0.10 && probability(r) < 0.25).length;
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
        <strong>${highDays}</strong> high-risk ·
        <strong>${elevatedDays}</strong> elevated ·
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
      const riskClass = getRiskClass(p);
      const isClim = (row.source || "").toLowerCase().includes("climatology");

      cell.className = `day ${riskClass} ${isClim ? "climatology" : ""}`;
      cell.title = `${city} ${row.date}: ${formatPercent(p)} risk`;

      cell.innerHTML = `
        <span class="day-number">${displayDay}</span>
        <div class="day-main">
          <span class="prob">${formatPercent(p)}</span>
          <span class="source">${sourceLabel(row)}</span>
        </div>
      `;
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
        <p class="subtitle">
          The demo could not find daily.csv with a date column.
        </p>
      </div>
    `;
    return;
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
        <p class="subtitle">
          The daily prediction file loaded, but it does not contain dates from
          ${startISO} to ${endISO}.
        </p>
      </div>
    `;
    return;
  }

  const cities = [...new Set(windowRows.map(r => r.city))].sort();

  cities.forEach(city => {
    const cityRows = windowRows
      .filter(r => r.city === city)
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    dailyView.appendChild(buildCalendar(city, cityRows, startDate, endDate));
  });
}

function renderMonthly(rows) {
  if (!rows.length) {
    monthlyTable.innerHTML = "<p>No window summary available.</p>";
    return;
  }

  const table = document.createElement("table");

  table.innerHTML = `
    <thead>
      <tr>
        <th>City</th>
        <th>Window</th>
        <th>Probability</th>
        <th>Risk days</th>
        <th>Forecast days</th>
        <th>Climatology days</th>
      </tr>
    </thead>
    <tbody>
      ${rows.map(row => {
        const windowLabel =
          row.window_start && row.window_end
            ? `${row.window_start} → ${row.window_end}`
            : row.target_month || row.month || "—";

        return `
          <tr>
            <td>${row.city}</td>
            <td>${windowLabel}</td>
            <td>${formatPercent(summaryProbability(row))}</td>
            <td>${row.risk_days_predicted ?? "—"}</td>
            <td>${row.n_short_horizon_days ?? "—"}</td>
            <td>${row.n_climatology_days ?? "—"}</td>
          </tr>
        `;
      }).join("")}
    </tbody>
  `;

  monthlyTable.innerHTML = "";
  monthlyTable.appendChild(table);
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

    document.getElementById("targetMonth").innerHTML = formatWindow(startDate, endDate);

    const dailyText = await loadText(`${PREDICTIONS_BASE}/latest/daily.csv`);
    const monthlyText = await loadText(`${PREDICTIONS_BASE}/latest/monthly.csv`);

    const dailyRows = parseCSV(dailyText);
    const monthlyRows = parseCSV(monthlyText);

    renderDaily(dailyRows, startDate, endDate);
    renderMonthly(monthlyRows);

    const startISO = toISODate(startDate);
    const endISO = toISODate(endDate);

    const visibleRows = dailyRows.filter(row => {
      return row.date >= startISO && row.date <= endISO;
    });

    const cities = new Set(visibleRows.map(r => r.city));
    const highRiskDays = visibleRows.filter(r => probability(r) >= 0.25).length;

    document.getElementById("cityCount").textContent = cities.size;
    document.getElementById("highRiskDays").textContent = highRiskDays;
  } catch (err) {
    console.error(err);

    dailyView.innerHTML = `
      <div class="city-card">
        <h2>Could not load prediction files</h2>
        <p class="subtitle">
          Make sure <code>predictions/latest.json</code>,
          <code>predictions/latest/daily.csv</code>, and
          <code>predictions/latest/monthly.csv</code> exist.
        </p>
        <p class="subtitle">${err.message}</p>
      </div>
    `;
  }
}

init();