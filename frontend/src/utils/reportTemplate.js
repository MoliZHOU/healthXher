export const buildReportHtml = ({
  patientName,
  reportDate,
  reportId,
  probability,
  ringAngle,
  reportGroups,
  reportFieldLabels,
  reportValues,
  joints,
  jointScores,
}) => {
  const ringStyle = `conic-gradient(#f97316 0 ${ringAngle}deg, rgba(15,23,42,0.10) ${ringAngle}deg 360deg)`;

  const renderReportRows = (fields) => fields.map((field) => (
    `<tr><td class="k">${reportFieldLabels[field]}</td><td class="v">${reportValues[field] ?? 'N/A'}</td></tr>`
  )).join('');

  const renderJointColumn = (sideLabel) => {
    const sideJoints = joints.filter((joint) => joint.label.startsWith(sideLabel));
    return sideJoints.map((joint) => {
      const pain = jointScores[joint.id]?.pain ?? 0;
      const swelling = jointScores[joint.id]?.swelling ?? 0;
      return `
        <div class="joint-item">
          <div class="joint-top">
            <p class="joint-name">${joint.label}</p>
          </div>
          <div class="bars">
            <div class="bar-wrap">
              <div class="bar-label"><span>Pain</span></div>
              <div class="bar"><div class="fill" style="width:${pain}%"></div></div>
            </div>
            <div class="bar-wrap">
              <div class="bar-label"><span>Swelling</span></div>
              <div class="bar"><div class="fill orange" style="width:${swelling}%"></div></div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  };

  const renderJointItems = () => `
    <div class="joint-grid">
      <div class="joint-column">
        <h3 class="joint-col-title">Left Side</h3>
        ${renderJointColumn('Left')}
      </div>
      <div class="joint-column">
        <h3 class="joint-col-title">Right Side</h3>
        ${renderJointColumn('Right')}
      </div>
    </div>
  `;

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>High Risk Clinical Report</title>
  <style>
    @page { size: A4; margin: 14mm; }
    html, body {
      padding: 0;
      margin: 0;
      background: #0b1220;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
                   Arial, "Noto Sans", "Liberation Sans", "PingFang SC", "Hiragino Sans GB",
                   "Microsoft YaHei", sans-serif;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    .page { background: #0b1220; padding: 0; }
    .sheet {
      background: #ffffff;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 24px 60px rgba(2, 6, 23, 0.35);
    }
    .header {
      position: relative;
      padding: 22px 22px 18px 22px;
      color: #e2e8f0;
      background: radial-gradient(1100px 380px at 10% 5%, rgba(56,189,248,0.35) 0%, rgba(56,189,248,0) 55%),
                  radial-gradient(900px 360px at 80% 0%, rgba(167,139,250,0.30) 0%, rgba(167,139,250,0) 60%),
                  linear-gradient(135deg, #0b1220 0%, #111b33 55%, #0b1220 100%);
      border-bottom: 1px solid rgba(226, 232, 240, 0.12);
    }
    .brand-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .logo {
      width: 44px;
      height: 44px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(56,189,248,0.95), rgba(167,139,250,0.95));
      box-shadow: 0 10px 28px rgba(56,189,248,0.18);
      position: relative;
      flex: 0 0 auto;
    }
    .logo:after {
      content: "";
      position: absolute;
      inset: 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.18);
      backdrop-filter: blur(6px);
    }
    .title-wrap { min-width: 0; }
    .title {
      font-size: 20px;
      letter-spacing: 0.2px;
      margin: 0;
      font-weight: 750;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .subtitle {
      margin: 4px 0 0 0;
      font-size: 12.5px;
      color: rgba(226,232,240,0.80);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(148,163,184,0.14);
      border: 1px solid rgba(226,232,240,0.14);
      color: rgba(226,232,240,0.92);
      font-size: 12px;
      white-space: nowrap;
    }
    .chip strong { color: #ffffff; font-weight: 700; }
    .content {
      padding: 18px 18px 20px 18px;
      background:
        radial-gradient(900px 300px at 5% 0%, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0) 52%),
        radial-gradient(800px 260px at 98% 6%, rgba(167,139,250,0.10) 0%, rgba(167,139,250,0) 55%),
        #ffffff;
    }
    .grid {
      display: grid;
      grid-template-columns: 260px 1fr;
      column-gap: 12px;
      row-gap: 12px;
      align-items: stretch;
    }
    .card {
      border-radius: 16px;
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(255,255,255,0.92);
      box-shadow: 0 14px 36px rgba(2, 6, 23, 0.06);
      overflow: hidden;
    }
    .card-head {
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid rgba(15,23,42,0.08);
      background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(248,250,252,0.75));
    }
    .card-title {
      margin: 0;
      font-size: 13.5px;
      font-weight: 780;
      color: #0f172a;
      letter-spacing: 0.15px;
    }
    .card-body { padding: 12px 14px 14px 14px; }
    .risk-card .card-body { padding: 10px 12px 12px 12px; }
    .risk-card { width: auto; }
    .risk-wrap {
      display: grid;
      gap: 12px;
      justify-items: center;
      text-align: center;
    }
    .ring {
      width: 80px;
      height: 80px;
      border-radius: 999px;
      position: relative;
      background: ${ringStyle};
      box-shadow: 0 18px 42px rgba(249,115,22,0.12);
    }
    .ring-inner {
      position: absolute;
      inset: 7px;
      border-radius: 999px;
      background:
        radial-gradient(120px 120px at 30% 25%, rgba(56,189,248,0.20) 0%, rgba(56,189,248,0) 55%),
        linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid rgba(15,23,42,0.10);
      display: grid;
      place-items: center;
      text-align: center;
      padding: 6px;
    }
    .ring-kpi {
      font-size: 16px;
      font-weight: 820;
      color: #0f172a;
      line-height: 1.05;
      margin: 0;
    }
    .ring-sub {
      font-size: 10px;
      color: rgba(15,23,42,0.62);
      margin: 4px 0 0 0;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .t { border: 1px solid rgba(15,23,42,0.10); border-radius: 14px; overflow: hidden; }
    .t thead th {
      text-align: left;
      font-size: 11.5px;
      letter-spacing: 0.15px;
      color: rgba(15,23,42,0.70);
      background: #f8fafc;
      padding: 10px 10px;
      border-bottom: 1px solid rgba(15,23,42,0.08);
    }
    .t tbody td {
      padding: 9px 10px;
      border-bottom: 1px solid rgba(15,23,42,0.06);
      vertical-align: top;
    }
    .t tbody tr:last-child td { border-bottom: 0; }
    .k { color: rgba(15,23,42,0.68); width: 48%; }
    .v { font-weight: 720; color: #0f172a; }
    .section { margin-top: 12px; display: grid; gap: 12px; }
    .callout {
      border-radius: 16px;
      border: 1px solid rgba(56,189,248,0.22);
      background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(167,139,250,0.10));
      padding: 14px 14px;
      box-shadow: 0 14px 34px rgba(2,6,23,0.05);
    }
    .callout.neutral {
      border-color: rgba(15,23,42,0.12);
      background: #ffffff;
      box-shadow: 0 12px 28px rgba(2,6,23,0.05);
    }
    .callout h3 { margin: 0 0 8px 0; font-size: 13.5px; font-weight: 820; color: #0f172a; }
    .callout p { margin: 0; font-size: 12.5px; line-height: 1.55; color: rgba(15,23,42,0.74); }
    .joint-list {
      display: grid;
      gap: 10px;
    }
    .joint-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .joint-column {
      display: grid;
      gap: 10px;
    }
    .joint-col-title {
      margin: 0 0 2px 0;
      font-size: 12px;
      font-weight: 800;
      color: rgba(15,23,42,0.6);
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .joint-item {
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 14px;
      padding: 10px 12px;
      background: linear-gradient(180deg, rgba(248,250,252,0.88), rgba(248,250,252,0.60));
    }
    .joint-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .joint-name {
      margin: 0;
      font-size: 12.5px;
      font-weight: 780;
      color: #0f172a;
    }
    .bars {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      align-items: center;
    }
    .bar-wrap { display: grid; gap: 6px; }
    .bar-label {
      font-size: 11px;
      color: rgba(15,23,42,0.60);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(15,23,42,0.08);
      overflow: hidden;
      position: relative;
    }
    .fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(56,189,248,0.95), rgba(167,139,250,0.95));
    }
    .fill.orange {
      background: linear-gradient(90deg, rgba(249,115,22,0.95), rgba(244,63,94,0.92));
    }
    .list { display: grid; gap: 10px; }
    .li {
      border-radius: 16px;
      border: 1px solid rgba(15,23,42,0.10);
      background: #ffffff;
      padding: 12px 12px;
      box-shadow: 0 12px 28px rgba(2,6,23,0.05);
    }
    .li h4 { margin: 0 0 6px 0; font-size: 12.8px; font-weight: 820; color: #0f172a; }
    .li p { margin: 0; font-size: 12.4px; line-height: 1.55; color: rgba(15,23,42,0.74); }
    .footer {
      padding: 12px 18px 16px 18px;
      background: #0b1220;
      color: rgba(226,232,240,0.84);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 11.5px;
      border-top: 1px solid rgba(226,232,240,0.12);
    }
    .footer .muted { color: rgba(226,232,240,0.72); }
    .avoid-break { break-inside: avoid; page-break-inside: avoid; }
    .tight { margin-top: 10px; }
  </style>
</head>
<body>
  <div class="page">
    <div class="sheet">
      <div class="header">
        <div class="brand-row">
          <div class="brand">
            <div class="logo" aria-hidden="true"></div>
            <div class="title-wrap">
              <h1 class="title">High Risk Clinical Report</h1>
              <div class="subtitle">
                <span class="chip">Patient: <strong>${patientName}</strong></span>
                <span class="chip">Date: <strong>${reportDate}</strong></span>
              </div>
            </div>
          </div>
          <div class="chip" title="Document ID">
            Report ID <strong>${reportId}</strong>
          </div>
        </div>
      </div>

      <div class="content">
        <div class="grid">
          <div class="card avoid-break risk-card">
            <div class="card-head">
              <h2 class="card-title">Risk Snapshot</h2>
              <p class="card-note">Model output</p> 
            </div>
            <div class="card-body">
              <div class="risk-wrap">
                <div class="ring" aria-label="Risk probability ring">
                  <div class="ring-inner">
                    <p class="ring-kpi">${probability}%</p>
                    <p class="ring-sub">Risk Probability</p>
                  </div>
                </div>
              </div>

              <div class="callout tight">
                <h3>Summary</h3>
                <p>
                  Based on your clinical profile, our system has identified a significant risk score.
                  While this is not a final diagnosis, your symptoms align with the early-onset patterns of
                  Rheumatoid Arthritis (RA).
                </p>
              </div>
            </div>
          </div>

          <div class="card avoid-break">
            <div class="card-head">
              <h2 class="card-title">Patient Data</h2>
              <p class="card-note">Structured inputs</p>
            </div>
            <div class="card-body">
              <div class="t">
                <table>
                  <thead>
                    <tr><th colspan="2">Demographics</th></tr>
                  </thead>
                  <tbody>
                    ${renderReportRows(reportGroups[0].fields)}
                  </tbody>
                </table>
              </div>

              <div style="height:10px;"></div>

              <div class="t">
                <table>
                  <thead>
                    <tr><th colspan="2">Biometrics</th></tr>
                  </thead>
                  <tbody>
                    ${renderReportRows(reportGroups[1].fields)}
                  </tbody>
                </table>
              </div>

              <div style="height:10px;"></div>

              <div class="t">
                <table>
                  <thead>
                    <tr><th colspan="2">Lifestyle &amp; Health</th></tr>
                  </thead>
                  <tbody>
                    ${renderReportRows(reportGroups[2].fields)}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <div class="card section avoid-break">
          <div class="card-head">
            <h2 class="card-title">Joint Scores</h2>
            <p class="card-note">Pain &amp; Swelling (0–100)</p>
          </div>
          <div class="card-body">
            <div class="joint-list">
              ${renderJointItems()}
            </div>
          </div>
        </div>

        <div class="section" style="grid-template-columns: 1fr 1fr; display:grid;">
          <div class="card avoid-break">
            <div class="card-head">
              <h2 class="card-title">Suggested Tests</h2>
              <p class="card-note">Next steps</p>
            </div>
            <div class="card-body">
              <div class="list">
                <div class="li">
                  <h4>1) ACPA (Anti-Cyclic Citrullinated Peptide Antibody)</h4>
                  <p>
                    ACPA is the most specific biomarker for RA. A positive result often appears years before
                    irreversible joint damage occurs. Testing for ACPA helps us determine if your immune system
                    has begun targeting joint tissues specifically.
                  </p>
                </div>
                <div class="li">
                  <h4>2) RF (Rheumatoid Factor) &amp; ESR/CRP</h4>
                  <p>
                    RF assesses the intensity of systemic inflammation. We also recommend checking ESR
                    (Erythrocyte Sedimentation Rate) and CRP (C-reactive Protein) to quantify the current fire
                    of inflammation in your body.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div class="card avoid-break">
            <div class="card-head">
              <h2 class="card-title">Clinical Recommendation</h2>
              <p class="card-note">Care pathway</p>
            </div>
            <div class="card-body">
              <div class="callout neutral">
                <h3>Consultation</h3>
                <p>
                  We recommend a consultation with a Board-Certified Rheumatologist within the next 14 days.
                  If a diagnosis is confirmed, a monthly Treat-to-Target (T2T) follow-up is standard until your
                  inflammation markers normalize.
                </p>
              </div>

              <div style="height:12px;"></div>

              <div class="callout neutral">
                <h3>Follow-Up</h3>
                <p>
                  Once you receive your lab results, please upload your ACPA/RF titers here. This allows our system
                  to adjust your risk trajectory and provide personalized management tips for your specific RA subtype.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div class="card section avoid-break">
          <div class="card-head">
            <h2 class="card-title">Lifestyle Recommendations</h2>
            <p class="card-note">Supportive measures</p>
          </div>
          <div class="card-body">
            <div class="list">
              <div class="li">
                <h4>The Anti-Inflammatory Kitchen</h4>
                <p>
                  Shift to a strictly anti-inflammatory diet. Prioritize Omega-3 fatty acids (found in fatty fish or flaxseeds)
                  and antioxidants (berries, leafy greens). These act as natural modulators to dampen cytokine storms.
                </p>
              </div>
              <div class="li">
                <h4>Joint Protection &amp; Pacing</h4>
                <p>
                  Avoid high-impact stress. If you are experiencing morning stiffness, perform range of motion exercises in warm water.
                  Do not push through sharp pain; instead, practice pacing, balancing activity with rest to prevent flare-ups.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="footer">
        <div>High Risk Clinical Report · <span class="muted">Generated ${reportDate}</span></div>
        <div class="muted">For clinical decision support only</div>
      </div>
    </div>
  </div>
  <script>
    window.onload = () => {
      setTimeout(() => {
        window.print();
      }, 300);
    };
  </script>
</body>
</html>`;
};
