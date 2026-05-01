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

function monthName(monthStr) {
  const [year, month] = monthStr.split("-").map(Number);
  const date = new Date(year, month - 1, 1);

  return date.toLocaleString("en-US", {
    month: "long",
    year: "numeric",
  });
}

function buildCalendar(city, rows, targetMonth) {
  const [year, month] = targetMonth.split("-").map(Number);
  const firstDate = new Date(year, month - 1, 1);
  const lastDate = new Date(year, month, 0);
  const daysInMonth = lastDate.getDate();

  const firstDayIndex = (firstDate.getDay() + 6) % 7; // Monday first

  const rowByDay = new Map();

  rows.forEach(row => {
    const date = new Date(row.date);
    const day = date.getDate();
    rowByDay.set(day, row);
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
      <h2>${city}</h2>
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

  for (let day = 1; day <= daysInMonth; day++) {
    const row = rowByDay.get(day);
    const cell = document.createElement("div");

    if (!row) {
      cell.className = "day";
      cell.innerHTML = `
        <span class="day-number">${day}</span>
        <span class="prob">—</span>
        <span class="source">no data</span>
      `;
    } else {
      const p = probability(row);
      const riskClass = getRiskClass(p);
      const isClim = (row.source || "").toLowerCase().includes("climatology");

      cell.className = `day ${riskClass} ${isClim ? "climatology" : ""}`;
      cell.title = `${city} ${row.date}: ${formatPercent(p)} risk`;

      cell.innerHTML = `
        <span class="day-number">${day}</span>
        <span class="prob">${formatPercent(p)}</span>
        <span class="source">${sourceLabel(row)}</span>
      `;
    }

    calendar.appendChild(cell);
  }

  card.appendChild(calendar);
  return card;
}

function renderDaily(rows, targetMonth) {
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

  const cities = [...new Set(rows.map(r => r.city))].sort();

  cities.forEach(city => {
    const cityRows = rows
      .filter(r => r.city === city)
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    dailyView.appendChild(buildCalendar(city, cityRows, targetMonth));
  });
}

function renderMonthly(rows) {
  if (!rows.length) {
    monthlyTable.innerHTML = "<p>No monthly summary available.</p>";
    return;
  }

  const table = document.createElement("table");

  table.innerHTML = `
    <thead>
      <tr>
        <th>City</th>
        <th>Month</th>
        <th>Probability</th>
        <th>Prediction</th>
      </tr>
    </thead>
    <tbody>
      ${rows.map(row => `
        <tr>
          <td>${row.city}</td>
          <td>${row.target_month || row.month || "—"}</td>
          <td>${formatPercent(probability(row))}</td>
          <td>${Number(row.prediction) === 1 ? "Risk" : "No risk"}</td>
        </tr>
      `).join("")}
    </tbody>
  `;

  monthlyTable.innerHTML = "";
  monthlyTable.appendChild(table);
}

async function init() {
  try {
    const latest = await loadJSON(`${PREDICTIONS_BASE}/latest.json`);
    const targetMonth = latest.month || latest.target_month;

    document.getElementById("targetMonth").textContent = monthName(targetMonth);

    const dailyText = await loadText(`${PREDICTIONS_BASE}/${targetMonth}/daily.csv`);
    const monthlyText = await loadText(`${PREDICTIONS_BASE}/${targetMonth}/monthly.csv`);

    const dailyRows = parseCSV(dailyText);
    const monthlyRows = parseCSV(monthlyText);

    renderDaily(dailyRows, targetMonth);
    renderMonthly(monthlyRows);

    const cities = new Set(dailyRows.map(r => r.city));
    const highRiskDays = dailyRows.filter(r => probability(r) >= 0.25).length;

    document.getElementById("cityCount").textContent = cities.size;
    document.getElementById("highRiskDays").textContent = highRiskDays;
  } catch (err) {
    console.error(err);

    dailyView.innerHTML = `
      <div class="city-card">
        <h2>Could not load prediction files</h2>
        <p class="subtitle">
          Make sure <code>predictions/latest.json</code> and
          <code>predictions/YYYY-MM/daily.csv</code> exist.
        </p>
        <p class="subtitle">${err.message}</p>
      </div>
    `;
  }
}

init();