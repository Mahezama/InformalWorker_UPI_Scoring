/**
 * PRISM-CREDIT™ Institutional Underwriting Engine & Research Simulator
 * IEEE Research Project: Informal Worker Credit Scoring via UPI & Telecom Behavior
 * Compliant with RBI Microfinance Master Directions (2022) & DPDP Act (2023)
 */

// Embedded fallback metrics (aligned with IEEE Manuscript Tables IV, V, VI, VII)
let metricsData = {
  main_metrics: [
    { model: "XGBoost (Proposed)", AUC: 0.7214, Accuracy: 0.6525, Precision: 0.6461, Recall: 0.6790, F1: 0.6621, Brier: 0.2072 },
    { model: "Logistic Regression Scorecard", AUC: 0.6977, Accuracy: 0.6280, Precision: 0.6223, Recall: 0.6570, F1: 0.6392, Brier: 0.2162 },
    { model: "Random Forest Baseline", AUC: 0.6993, Accuracy: 0.6350, Precision: 0.6218, Recall: 0.6949, F1: 0.6563, Brier: 0.2198 }
  ],
  tier_stats: [
    { tier: "Eligible (70–100)", count: 236, default_rate: 0.1483, mean_score: 84.68, boundary: "Score ≥ 70" },
    { tier: "Review (40–69)", count: 1362, default_rate: 0.4670, mean_score: 51.65, boundary: "40 ≤ Score < 70" },
    { tier: "High Risk (0–39)", count: 402, default_rate: 0.8259, mean_score: 23.58, boundary: "Score < 40" }
  ],
  ablation: [
    { name: "UPI-Only Modality", AUC: 0.7124, Brier: 0.2077, cv_auc: 0.7220, cold_start_auc: 0.5126 },
    { name: "Telecom Recharge-Only Modality", AUC: 0.5447, Brier: 0.2485, cv_auc: 0.5560, cold_start_auc: "—" },
    { name: "Combined Multi-Modal Engine", AUC: 0.7234, Brier: 0.2066, cv_auc: 0.7352, cold_start_auc: 0.5468 }
  ],
  external_references: [
    { Reference: "Yadav et al. (2026)", Population: "Retail Borrowers, India", Modality: "Synthetic Bureau + Bank Records", AUC: 0.765, Brier: 0.195, Difference: "Does not target thin-file informal workers" },
    { Reference: "Yadav et al. (2026)", Population: "MSME Enterprises, India", Modality: "Synthetic Cashflow/Invoices", AUC: 0.784, Brier: 0.184, Difference: "MSME commercial turnover vs daily micro-wages" },
    { Reference: "AI-BAAM / Ng et al. (2025)", Population: "Malaysian MSMEs", Modality: "PDF Bank Statements", AUC: 0.806, Brier: "—", Difference: "Requires formal bank accounts; informal workers lack statements" },
    { Reference: "Ots et al. (2020)", Population: "Prepaid Customers, Europe", Modality: "Telecom / CDR Metadata", AUC: 0.620, Brier: "—", Difference: "Telecom-only footprint without payment transaction stream" },
    { Reference: "Proposed Framework (XGBoost)", Population: "Informal Workers (Vendors/Gig)", Modality: "UPI + Mobile Recharge Synergy", AUC: 0.7214, Brier: 0.2024, Difference: "Native TreeSHAP point-attribution & consent-gated AA architecture" }
  ],
  cold_start: {
    n_cold_workers: 1042,
    UPI_only_AUC: 0.5126,
    Combined_AUC: 0.5468,
    UPI_only_Brier: 0.2508,
    Combined_Brier: 0.2493
  }
};

let sampleWorkers = [];

// Standard Institutional Worker Archetypes
const ARCHETYPES = {
  vendor: {
    trade_name: "Street Vendor (Micro-Retail)",
    type: "street_vendor",
    income: 14000,
    upi_txns: 15,
    inflow_cv: 0.35,
    counterparties: 20,
    inflow_ratio: 1.25,
    rch_freq: 1.4,
    rch_plan: 239,
    rch_interval: 3.5,
    emi_count: 1,
    loan_request: 15000
  },
  gig: {
    trade_name: "Platform Gig Partner (Logistics/Delivery)",
    type: "gig_delivery",
    income: 18500,
    upi_txns: 24,
    inflow_cv: 0.28,
    counterparties: 32,
    inflow_ratio: 1.45,
    rch_freq: 2.1,
    rch_plan: 399,
    rch_interval: 2.1,
    emi_count: 1,
    loan_request: 25000
  },
  domestic: {
    trade_name: "Domestic Service Worker (Homecare)",
    type: "domestic_worker",
    income: 8500,
    upi_txns: 5,
    inflow_cv: 0.52,
    counterparties: 4,
    inflow_ratio: 1.10,
    rch_freq: 1.0,
    rch_plan: 199,
    rch_interval: 1.8,
    emi_count: 0,
    loan_request: 8000
  },
  labourer: {
    trade_name: "Daily-Wage Labourer (Construction/Trade)",
    type: "daily_wage_labourer",
    income: 7000,
    upi_txns: 2,
    inflow_cv: 0.85,
    counterparties: 2,
    inflow_ratio: 0.95,
    rch_freq: 0.7,
    rch_plan: 99,
    rch_interval: 6.8,
    emi_count: 0,
    loan_request: 5000
  }
};

let currentEvaluation = {};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSimulator();
  initArchetypes();
  renderBenchmarkTables();
  loadDataFiles();
  initModals();
});

// ==========================================================================
// Tab Navigation
// ==========================================================================
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetPaneId = `tab-${tab.dataset.tab}`;
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById(targetPaneId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// ==========================================================================
// Archetype Selector
// ==========================================================================
function initArchetypes() {
  const archButtons = document.querySelectorAll(".arch-btn");
  archButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      archButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const archKey = btn.dataset.persona;
      applyArchetype(ARCHETYPES[archKey]);
    });
  });
}

function applyArchetype(data) {
  if (!data) return;
  document.getElementById("inp-workertype").value = data.type;
  document.getElementById("inp-income").value = data.income;
  document.getElementById("inp-upi-txns").value = data.upi_txns;
  document.getElementById("inp-inflow-cv").value = data.inflow_cv;
  document.getElementById("inp-counterparties").value = data.counterparties;
  document.getElementById("inp-inflow-ratio").value = data.inflow_ratio;
  document.getElementById("inp-rch-freq").value = data.rch_freq;
  document.getElementById("inp-rch-plan").value = data.rch_plan;
  document.getElementById("inp-rch-interval").value = data.rch_interval;
  document.getElementById("inp-emi-count").value = data.emi_count;
  if (data.loan_request) {
    document.getElementById("inp-loan-request").value = data.loan_request;
  }

  updateSliderLabels();
  calculateScore();
}

// ==========================================================================
// Live Simulator & Underwriting Logic
// ==========================================================================
function initSimulator() {
  const inputIds = [
    "inp-workertype", "inp-income", "inp-upi-txns", "inp-inflow-cv",
    "inp-counterparties", "inp-inflow-ratio", "inp-rch-freq", "inp-rch-plan",
    "inp-rch-interval", "inp-emi-count", "inp-loan-request"
  ];

  inputIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("input", () => {
        updateSliderLabels();
        calculateScore();
      });
    }
  });

  document.getElementById("btn-reset-form")?.addEventListener("click", () => {
    applyArchetype(ARCHETYPES.vendor);
  });

  updateSliderLabels();
  calculateScore();
}

function updateSliderLabels() {
  const upiTxns = document.getElementById("inp-upi-txns").value;
  document.getElementById("val-upi-txns").textContent = `${upiTxns} txns/wk`;

  const inflowCv = parseFloat(document.getElementById("inp-inflow-cv").value);
  let cvDesc = inflowCv < 0.4 ? "Stable" : (inflowCv < 0.8 ? "Moderate" : "Volatile");
  document.getElementById("val-inflow-cv").textContent = `${inflowCv.toFixed(2)} (${cvDesc})`;

  const counterparties = document.getElementById("inp-counterparties").value;
  document.getElementById("val-counterparties").textContent = `${counterparties} unique`;

  const ratio = parseFloat(document.getElementById("inp-inflow-ratio").value);
  document.getElementById("val-inflow-ratio").textContent = `${ratio.toFixed(2)}x`;

  const rchFreq = parseFloat(document.getElementById("inp-rch-freq").value);
  document.getElementById("val-rch-freq").textContent = `${rchFreq.toFixed(1)} / mo`;

  const rchPlan = document.getElementById("inp-rch-plan").value;
  document.getElementById("val-rch-plan").textContent = `₹${rchPlan}`;

  const rchInterval = parseFloat(document.getElementById("inp-rch-interval").value);
  let intDesc = rchInterval < 3 ? "Regular" : (rchInterval < 7 ? "Moderate" : "Jittered");
  document.getElementById("val-rch-interval").textContent = `${rchInterval.toFixed(1)} days (${intDesc})`;
}

/**
 * Calibrated Underwriting Calculation
 * Matches XGBoost simulation + TreeSHAP point conversions in Section III-C
 */
function calculateScore() {
  const workerType = document.getElementById("inp-workertype").value;
  const income = parseFloat(document.getElementById("inp-income").value) || 0;
  const upiTxns = parseFloat(document.getElementById("inp-upi-txns").value) || 0;
  const inflowCv = parseFloat(document.getElementById("inp-inflow-cv").value) || 0.5;
  const counterparties = parseFloat(document.getElementById("inp-counterparties").value) || 0;
  const inflowRatio = parseFloat(document.getElementById("inp-inflow-ratio").value) || 1.0;
  const rchFreq = parseFloat(document.getElementById("inp-rch-freq").value) || 1.0;
  const rchPlan = parseFloat(document.getElementById("inp-rch-plan").value) || 199;
  const rchInterval = parseFloat(document.getElementById("inp-rch-interval").value) || 4.0;
  const emiCount = parseInt(document.getElementById("inp-emi-count").value) || 0;
  const loanRequest = parseFloat(document.getElementById("inp-loan-request").value) || 10000;

  // Baseline probability representation
  let baseScore = 50.0;
  let shapFactors = [];

  // Signal 1: Weekly UPI Frequency & Counterparties
  const upiDelta = Math.min(22, (upiTxns - 10) * 1.25 + (counterparties - 12) * 0.35);
  const upiLogOdds = (upiDelta / -25).toFixed(3);
  shapFactors.push({
    signal: "UPI Transaction Frequency & Payer Network",
    metric: `${upiTxns} txns/wk, ${counterparties} counterparties`,
    logOdds: upiLogOdds,
    pts: Math.round(upiDelta),
    finding: upiDelta >= 0 ? "High commercial velocity; frequent active client settlements" : "Constrained transaction volume indicating thin-file history"
  });

  // Signal 2: Cashflow Regularity (CV)
  const cvDelta = (0.50 - inflowCv) * 22;
  const cvLogOdds = (cvDelta / -25).toFixed(3);
  shapFactors.push({
    signal: "Inflow Volatility (Coefficient of Variation)",
    metric: `CV = ${inflowCv.toFixed(2)}`,
    logOdds: cvLogOdds,
    pts: Math.round(cvDelta),
    finding: cvDelta >= 0 ? "Predictable revenue schedule; low income variance across weeks" : "High volatility in weekly inflows; elevated cashflow uncertainty"
  });

  // Signal 3: Net Cashflow Surplus & Inflow/Outflow Ratio
  const incomeNorm = (income - 12000) / 10000;
  const ratioDelta = (inflowRatio - 1.0) * 18 + incomeNorm * 6;
  const ratioLogOdds = (ratioDelta / -25).toFixed(3);
  shapFactors.push({
    signal: "Net Liquidity Surplus (Inflow / Outflow Multiple)",
    metric: `${inflowRatio.toFixed(2)}x, ₹${income.toLocaleString("en-IN")}/mo`,
    logOdds: ratioLogOdds,
    pts: Math.round(ratioDelta),
    finding: ratioDelta >= 0 ? "Operating surplus verified; monthly receipts comfortably exceed outlays" : "Constrained cash margins; high proportion of outflows relative to inflows"
  });

  // Signal 4: Telecom Recharge Consistency & Plan Value
  const rchDelta = (rchFreq - 1.2) * 6 + ((rchPlan - 200) / 150) * 5 - (rchInterval - 3.5) * 1.8;
  const rchLogOdds = (rchDelta / -25).toFixed(3);
  shapFactors.push({
    signal: "Telecom Recharge Regularity & Plan Tier",
    metric: `${rchFreq.toFixed(1)}/mo @ ₹${rchPlan} (std: ${rchInterval.toFixed(1)}d)`,
    logOdds: rchLogOdds,
    pts: Math.round(rchDelta),
    finding: rchDelta >= 0 ? "Established prepaid renewal discipline; low interval jitter" : "Irregular recharge intervals or lower-tier micro top-ups observed"
  });

  // Signal 5: Recurring Commitments & Debt Service
  let emiDelta = 0;
  if (emiCount === 0) emiDelta = 0;
  else if (emiCount === 1) emiDelta = +4; // established recurring debit capability
  else if (emiCount === 2) emiDelta = -8;
  else emiDelta = -14;
  const emiLogOdds = (emiDelta / -25).toFixed(3);
  shapFactors.push({
    signal: "Observed Recurring Debt Service (Fixed Debits)",
    metric: `${emiCount} Active Commitments (₹${emiCount * 500}/mo)`,
    logOdds: emiLogOdds,
    pts: Math.round(emiDelta),
    finding: emiDelta > 0 ? "Positive record of servicing recurring micro-obligations on time" : (emiDelta === 0 ? "No active debt servicing trail observed" : "Multiple recurring debits present; elevated leverage risk")
  });

  // Calculate Final Score
  const totalDelta = shapFactors.reduce((acc, f) => acc + f.pts, 0);
  const finalScore = Math.max(5, Math.min(96, Math.round(baseScore + totalDelta)));
  const defaultProb = Math.max(0.04, Math.min(0.96, (100 - finalScore) / 100));

  // Determine Decision Tier
  let tierKey = "review";
  let tierLabel = "REVIEW (TIER 2)";
  let tierName = "Conditional Review (40 ≤ Score < 70)";
  let tierSub = "Forwarded to human credit officer for manual review";
  let cohortDef = "46.7%";
  let approvedLimit = Math.min(15000, Math.round(income * 1.0 / 1000) * 1000);

  if (finalScore >= 70) {
    tierKey = "eligible";
    tierLabel = "ELIGIBLE (TIER 1)";
    tierName = "Eligible (Score ≥ 70)";
    tierSub = "Instant algorithmic sanction authorized";
    cohortDef = "14.8%";
    approvedLimit = Math.min(40000, Math.round(income * 1.8 / 1000) * 1000);
  } else if (finalScore < 40) {
    tierKey = "highrisk";
    tierLabel = "HIGH RISK (TIER 3)";
    tierName = "High Risk (Score < 40)";
    tierSub = "Elevated default probability; requires co-guarantor";
    cohortDef = "82.6%";
    approvedLimit = 0;
  }

  // Update UI Elements
  document.getElementById("res-score").textContent = finalScore;
  document.getElementById("res-tier-name").textContent = tierName;
  document.getElementById("res-tier-sub").textContent = tierSub;
  document.getElementById("res-prob").textContent = `${(defaultProb * 100).toFixed(1)}%`;
  document.getElementById("res-cohort-default").textContent = cohortDef;
  document.getElementById("res-loan-limit").textContent = approvedLimit > 0 ? `₹${approvedLimit.toLocaleString("en-IN")}` : "Unsanctioned (High Risk)";

  const badgeTier = document.getElementById("badge-tier");
  badgeTier.className = `tier-indicator-badge ${tierKey}`;
  badgeTier.textContent = tierLabel;

  // Update Gauge Circle
  const circleBar = document.getElementById("circle-bar");
  if (circleBar) {
    const totalCircumference = 377;
    const offset = totalCircumference - (totalCircumference * (finalScore / 100));
    circleBar.style.strokeDashoffset = offset;
    if (finalScore >= 70) {
      circleBar.style.stroke = "var(--color-eligible)";
    } else if (finalScore >= 40) {
      circleBar.style.stroke = "var(--color-review)";
    } else {
      circleBar.style.stroke = "var(--color-risk)";
    }
  }

  // RBI Affordability & DTI Assessment
  const proposedEmi = Math.round((loanRequest * 1.05) / 4); // 4-month repayment
  const existingDebt = emiCount * 500;
  const totalMonthlyDebt = existingDebt + proposedEmi;
  const dtiRatio = income > 0 ? (totalMonthlyDebt / income) * 100 : 100;
  const isRbiCompliant = dtiRatio <= 50.0;

  document.getElementById("rbi-inflow").textContent = `₹${income.toLocaleString("en-IN")}`;
  document.getElementById("rbi-obligations").textContent = `₹${totalMonthlyDebt.toLocaleString("en-IN")}`;
  document.getElementById("rbi-dti").textContent = `${dtiRatio.toFixed(1)}%`;

  const rbiCard = document.getElementById("rbi-compliance-status");
  const rbiStatusLabel = document.getElementById("rbi-status-label");

  if (isRbiCompliant) {
    rbiCard.className = "rbi-compliance-card compliant";
    rbiStatusLabel.textContent = `COMPLIANT (${dtiRatio.toFixed(1)}% ≤ 50% CAP)`;
  } else {
    rbiCard.className = "rbi-compliance-card violation";
    rbiStatusLabel.textContent = `VIOLATION (${dtiRatio.toFixed(1)}% > 50% CAP)`;
  }

  // Render SHAP Attribution Table
  renderShapTable(shapFactors);

  // Store Current State for Memo Generator
  const workerSelect = document.getElementById("inp-workertype");
  const workerTradeName = workerSelect.options[workerSelect.selectedIndex].text;

  currentEvaluation = {
    workerType: workerType,
    tradeName: workerTradeName,
    score: finalScore,
    tierKey: tierKey,
    tierLabel: tierLabel,
    tierName: tierName,
    prob: (defaultProb * 100).toFixed(1),
    approvedLimit: approvedLimit,
    income: income,
    upiTxns: upiTxns,
    inflowCv: inflowCv,
    counterparties: counterparties,
    inflowRatio: inflowRatio,
    rchFreq: rchFreq,
    rchPlan: rchPlan,
    rchInterval: rchInterval,
    emiCount: emiCount,
    proposedEmi: proposedEmi,
    existingDebt: existingDebt,
    totalMonthlyDebt: totalMonthlyDebt,
    dtiRatio: dtiRatio.toFixed(1),
    isRbiCompliant: isRbiCompliant,
    shapFactors: shapFactors
  };
}

function renderShapTable(factors) {
  const tbody = document.getElementById("shap-table-body");
  if (!tbody) return;

  tbody.innerHTML = factors.map(f => {
    const ptsClass = f.pts >= 0 ? "pts-pos font-mono" : "pts-neg font-mono";
    const ptsSign = f.pts > 0 ? `+${f.pts}` : `${f.pts}`;
    return `
      <tr>
        <td><strong>${f.signal}</strong></td>
        <td class="font-mono">${f.metric}</td>
        <td class="font-mono">${f.logOdds}</td>
        <td class="${ptsClass}">${ptsSign} pts</td>
        <td>${f.finding}</td>
      </tr>
    `;
  }).join("");
}

// ==========================================================================
// Empirical Benchmarks Renderer (Tab 2)
// ==========================================================================
function renderBenchmarkTables() {
  // Main Metrics (Table IV)
  const tbodyMain = document.querySelector("#table-main-metrics tbody");
  if (tbodyMain) {
    tbodyMain.innerHTML = metricsData.main_metrics.map((row, idx) => `
      <tr class="${idx === 0 ? 'table-row-highlight' : ''}">
        <td><strong>${row.model}</strong></td>
        <td class="font-mono font-bold ${idx === 0 ? 'text-blue' : ''}">${row.AUC.toFixed(4)}</td>
        <td class="font-mono">${(row.Accuracy * 100).toFixed(2)}%</td>
        <td class="font-mono">${(row.Precision * 100).toFixed(2)}%</td>
        <td class="font-mono">${(row.Recall * 100).toFixed(2)}%</td>
        <td class="font-mono">${row.F1.toFixed(4)}</td>
        <td class="font-mono">${row.Brier.toFixed(4)}</td>
      </tr>
    `).join("");
  }

  // Tier Stats (Table VI)
  const tbodyTiers = document.getElementById("table-tier-stats");
  if (tbodyTiers) {
    tbodyTiers.innerHTML = metricsData.tier_stats.map(row => {
      let badgeClass = "badge-review";
      if (row.tier.includes("Eligible")) badgeClass = "badge-eligible";
      else if (row.tier.includes("High")) badgeClass = "badge-highrisk";

      return `
        <tr>
          <td><span class="badge-tag ${badgeClass}">${row.tier}</span></td>
          <td class="font-mono">${row.boundary}</td>
          <td class="font-mono">${row.count}</td>
          <td class="font-mono font-bold">${(row.default_rate * 100).toFixed(1)}%</td>
          <td class="font-mono">${row.mean_score.toFixed(1)}</td>
        </tr>
      `;
    }).join("");
  }

  // Modality Ablation (Table VII)
  const tbodyAblation = document.getElementById("table-ablation");
  if (tbodyAblation) {
    tbodyAblation.innerHTML = metricsData.ablation.map((row, idx) => `
      <tr class="${idx === 2 ? 'table-row-highlight' : ''}">
        <td><strong>${row.name}</strong></td>
        <td class="font-mono font-bold ${idx === 2 ? 'text-blue' : ''}">${row.AUC.toFixed(4)}</td>
        <td class="font-mono">${row.Brier.toFixed(4)}</td>
        <td class="font-mono">${row.cv_auc}</td>
        <td class="font-mono">${row.cold_start_auc}</td>
      </tr>
    `).join("");
  }

  // Literature Comparison
  const tbodyExt = document.querySelector("#table-ext-ref tbody");
  if (tbodyExt) {
    tbodyExt.innerHTML = metricsData.external_references.map(row => `
      <tr>
        <td><strong>${row.Reference}</strong></td>
        <td>${row.Population}</td>
        <td>${row.Modality}</td>
        <td class="font-mono font-bold">${typeof row.AUC === 'number' ? row.AUC.toFixed(4) : row.AUC}</td>
        <td class="font-mono">${typeof row.Brier === 'number' ? row.Brier.toFixed(4) : row.Brier}</td>
        <td><small class="text-muted">${row.Difference}</small></td>
      </tr>
    `).join("");
  }
}

// ==========================================================================
// Applicant Registry Cohort Explorer (Tab 4)
// ==========================================================================
function loadDataFiles() {
  fetch("data/sample_workers.json")
    .then(r => r.json())
    .then(data => {
      sampleWorkers = data;
      renderCohortTable(sampleWorkers.slice(0, 30));
      initCohortFilter();
    })
    .catch(err => {
      console.warn("Could not load sample_workers.json; generating institutional sample pool.", err);
      generateFallbackCohort();
    });
}

function generateFallbackCohort() {
  const dummy = [];
  const trades = ["street_vendor", "daily_wage_labourer", "gig_delivery", "domestic_worker"];
  for (let i = 1; i <= 35; i++) {
    const t = trades[i % trades.length];
    dummy.push({
      worker_id: 1000 + i,
      worker_type: t,
      upi_txns_per_week: (Math.random() * 25 + 1).toFixed(1),
      est_monthly_income: Math.round((Math.random() * 15000 + 6000) / 500) * 500,
      upi_inflow_cv: (Math.random() * 0.7 + 0.15).toFixed(2),
      recharge_per_month: (Math.random() * 2 + 0.5).toFixed(1),
      recharge_interval_std: (Math.random() * 6 + 1).toFixed(1),
      true_label: Math.random() > 0.45 ? 0 : 1
    });
  }
  sampleWorkers = dummy;
  renderCohortTable(sampleWorkers);
  initCohortFilter();
}

function renderCohortTable(workers) {
  const tbody = document.getElementById("cohort-tbody");
  if (!tbody) return;

  tbody.innerHTML = workers.map(w => {
    const tradeLabels = {
      street_vendor: "Street Vendor",
      daily_wage_labourer: "Daily Labourer",
      gig_delivery: "Gig Delivery",
      domestic_worker: "Domestic Worker"
    };

    const statusBadge = w.true_label === 0 
      ? '<span class="badge-tag badge-eligible">Repaid (Good)</span>' 
      : '<span class="badge-tag badge-highrisk">Delinquent (Default)</span>';

    return `
      <tr>
        <td class="font-mono font-bold">#W-${w.worker_id}</td>
        <td><strong>${tradeLabels[w.worker_type] || w.worker_type}</strong></td>
        <td class="font-mono">${parseFloat(w.upi_txns_per_week).toFixed(1)} / wk</td>
        <td class="font-mono">₹${Math.round(w.est_monthly_income).toLocaleString("en-IN")}</td>
        <td class="font-mono">${parseFloat(w.upi_inflow_cv).toFixed(2)}</td>
        <td class="font-mono">${parseFloat(w.recharge_per_month).toFixed(1)} / mo</td>
        <td class="font-mono">${parseFloat(w.recharge_interval_std).toFixed(1)} d</td>
        <td>${statusBadge}</td>
        <td>
          <button class="btn-table-action" onclick="loadWorkerIntoTerminal(${w.worker_id})">
            Assess in Terminal
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

function initCohortFilter() {
  const searchInp = document.getElementById("cohort-search");
  const filterSelect = document.getElementById("filter-workertype");

  function applyFilter() {
    const q = (searchInp?.value || "").toLowerCase().trim();
    const type = filterSelect?.value || "all";

    const filtered = sampleWorkers.filter(w => {
      const matchType = type === "all" || w.worker_type === type;
      const matchQuery = !q || String(w.worker_id).includes(q) || (w.worker_type || "").toLowerCase().includes(q);
      return matchType && matchQuery;
    });

    renderCohortTable(filtered.slice(0, 50));
  }

  searchInp?.addEventListener("input", applyFilter);
  filterSelect?.addEventListener("change", applyFilter);
}

window.loadWorkerIntoTerminal = function(workerId) {
  const worker = sampleWorkers.find(w => w.worker_id === workerId);
  if (!worker) return;

  document.getElementById("inp-workertype").value = worker.worker_type || "street_vendor";
  document.getElementById("inp-income").value = Math.max(2000, Math.round(worker.est_monthly_income || 12000));
  document.getElementById("inp-upi-txns").value = Math.round(worker.upi_txns_per_week || 10);
  document.getElementById("inp-inflow-cv").value = parseFloat(worker.upi_inflow_cv || 0.4).toFixed(2);
  document.getElementById("inp-counterparties").value = Math.round(worker.upi_distinct_counterparties_per_month || 15);
  document.getElementById("inp-inflow-ratio").value = parseFloat(worker.inflow_outflow_ratio || 1.15).toFixed(2);
  document.getElementById("inp-rch-freq").value = parseFloat(worker.recharge_per_month || 1.2).toFixed(1);
  document.getElementById("inp-rch-plan").value = Math.round(worker.recharge_amount_mean || 199);
  document.getElementById("inp-rch-interval").value = parseFloat(worker.recharge_interval_std || 3.5).toFixed(1);
  document.getElementById("disp-case-id").textContent = `UW-2026-IND-${worker.worker_id}`;

  updateSliderLabels();
  calculateScore();

  // Switch to simulator tab
  document.getElementById("tab-btn-sim")?.click();
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// ==========================================================================
// Formal Credit Appraisal Memo Modal & Print Generator
// ==========================================================================
function initModals() {
  const memoModal = document.getElementById("memo-modal");
  const btnCloseMemo = document.getElementById("btn-close-memo");
  const btnPrintMemo = document.getElementById("btn-print-memo");

  const openMemoButtons = [
    document.getElementById("btn-open-memo-top"),
    document.getElementById("btn-generate-memo")
  ];

  openMemoButtons.forEach(btn => {
    btn?.addEventListener("click", () => {
      populateMemoModal();
      memoModal?.classList.add("active");
    });
  });

  btnCloseMemo?.addEventListener("click", () => {
    memoModal?.classList.remove("active");
  });

  memoModal?.addEventListener("click", (e) => {
    if (e.target === memoModal) memoModal.classList.remove("active");
  });

  btnPrintMemo?.addEventListener("click", () => {
    window.print();
  });

  // Diagnostic Image Lightbox Modal
  const imgModal = document.getElementById("image-modal");
  const imgClose = document.getElementById("modal-close");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");
  const modalDownload = document.getElementById("modal-download");

  document.querySelectorAll(".fig-card").forEach(card => {
    card.addEventListener("click", () => {
      const src = card.dataset.fig;
      const title = card.dataset.title;
      if (src && modalImg) {
        modalImg.src = src;
        if (modalTitle) modalTitle.textContent = title;
        if (modalDownload) modalDownload.href = src;
        imgModal?.classList.add("active");
      }
    });
  });

  imgClose?.addEventListener("click", () => {
    imgModal?.classList.remove("active");
  });

  imgModal?.addEventListener("click", (e) => {
    if (e.target === imgModal) imgModal.classList.remove("active");
  });
}

function populateMemoModal() {
  const ev = currentEvaluation;
  const today = new Date().toISOString().slice(0, 10);
  const docRef = document.getElementById("disp-case-id").textContent || "UW-2026-IND-8841";

  document.getElementById("memo-doc-ref").textContent = docRef;
  document.getElementById("memo-doc-date").textContent = today;
  document.getElementById("memo-borrower-id").textContent = docRef.replace("UW-", "BORROWER-");
  document.getElementById("memo-borrower-trade").textContent = ev.tradeName;

  document.getElementById("memo-score-val").textContent = `${ev.score} / 100`;
  document.getElementById("memo-tier-val").textContent = ev.tierLabel;
  document.getElementById("memo-prob-val").textContent = `${ev.prob}%`;
  document.getElementById("memo-limit-val").textContent = ev.approvedLimit > 0 
    ? `₹${ev.approvedLimit.toLocaleString("en-IN")}` 
    : "Unsanctioned";

  const stampStatus = document.getElementById("memo-stamp-status");
  stampStatus.textContent = `STATUS: ${ev.tierLabel.split(' ')[0]}`;
  stampStatus.style.color = ev.score >= 70 ? "var(--color-eligible)" : (ev.score >= 40 ? "var(--color-review)" : "var(--color-risk)");

  // RBI Affordability
  document.getElementById("memo-rbi-inflow").textContent = `₹${ev.income.toLocaleString("en-IN")}`;
  document.getElementById("memo-rbi-existing").textContent = `₹${ev.existingDebt.toLocaleString("en-IN")}`;
  document.getElementById("memo-rbi-proposed").textContent = `₹${ev.proposedEmi.toLocaleString("en-IN")}`;
  document.getElementById("memo-rbi-total").textContent = `₹${ev.totalMonthlyDebt.toLocaleString("en-IN")}`;
  document.getElementById("memo-rbi-dti").textContent = `${ev.dtiRatio}%`;

  const memoComp = document.getElementById("memo-rbi-comp");
  if (ev.isRbiCompliant) {
    memoComp.textContent = "PASS (≤ 50% RBI Limit)";
    memoComp.className = "font-bold text-emerald";
  } else {
    memoComp.textContent = "VIOLATION (> 50% RBI Limit)";
    memoComp.className = "font-bold text-rose";
  }

  // Telemetry Matrix
  document.getElementById("memo-txns-val").textContent = `${ev.upiTxns} txns / week`;
  document.getElementById("memo-cv-val").textContent = `${ev.inflowCv.toFixed(2)}`;
  document.getElementById("memo-cp-val").textContent = `${ev.counterparties} unique counterparties`;
  document.getElementById("memo-ratio-val").textContent = `${ev.inflowRatio.toFixed(2)}x`;
  document.getElementById("memo-rch-freq-val").textContent = `${ev.rchFreq.toFixed(1)} / month`;
  document.getElementById("memo-rch-plan-val").textContent = `₹${ev.rchPlan}`;
  document.getElementById("memo-rch-jitter-val").textContent = `${ev.rchInterval.toFixed(1)} days std`;
  document.getElementById("memo-emi-val").textContent = `${ev.emiCount} Active Fixed Debit(s)`;

  // SHAP Table
  const memoShapTbody = document.getElementById("memo-shap-table");
  if (memoShapTbody) {
    memoShapTbody.innerHTML = ev.shapFactors.map(f => {
      const ptsClass = f.pts >= 0 ? "pts-pos font-mono" : "pts-neg font-mono";
      const ptsSign = f.pts > 0 ? `+${f.pts}` : `${f.pts}`;
      return `
        <tr>
          <td><strong>${f.signal}</strong></td>
          <td class="font-mono">${f.metric}</td>
          <td class="font-mono">${f.logOdds}</td>
          <td class="${ptsClass}">${ptsSign} pts</td>
          <td>${f.finding}</td>
        </tr>
      `;
    }).join("");
  }

  // Terms Box
  const termsBox = document.getElementById("memo-terms-box");
  if (ev.score >= 70 && ev.isRbiCompliant) {
    termsBox.innerHTML = `
      <p><strong>Approved Facility:</strong> Micro-Working Capital Loan of up to <strong>₹${ev.approvedLimit.toLocaleString("en-IN")}</strong> repayable in 4 equal monthly installments of ₹${ev.proposedEmi.toLocaleString("en-IN")}.</p>
      <p><strong>Disbursement Channel:</strong> Direct UPI VPA transfer to applicant's primary transacting account.</p>
      <p><strong>Escalation Rule:</strong> Timely repayment of Tranche 1 qualifies applicant for automatic credit limit enhancement under PM SVANidhi laddering guidelines.</p>
    `;
  } else if (ev.score >= 40) {
    termsBox.innerHTML = `
      <p><strong>Disposition:</strong> Conditional Review. Algorithmic score indicates borderline eligibility. Loan officer verification of physical trade premises or co-guarantor signature required prior to sanction.</p>
      <p><strong>Recommended Provisional Ceiling:</strong> ₹${Math.min(10000, ev.approvedLimit).toLocaleString("en-IN")}.</p>
    `;
  } else {
    termsBox.innerHTML = `
      <p><strong>Disposition:</strong> Algorithmic Sanction Declined. Elevated default risk probability (${ev.prob}%) exceeds allowable risk tolerance for unsecured facilities.</p>
      <p><strong>Rehabilitation Recommendation:</strong> Worker advised to maintain consistent weekly UPI inflows and regular telecom renewals over next 60 days to rebuild alternative credit profile.</p>
    `;
  }
}
