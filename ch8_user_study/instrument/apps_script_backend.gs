// ═══════════════════════════════════════════════════════════════════
// UCLDP demo — Google Apps Script backend  (v3.0 — HCI full instrument)
// ═══════════════════════════════════════════════════════════════════
// Updated to match survey_version "v3.0_hci_full" — the comprehensive
// instrument with Parts A through G, SUS, Trust, Cognitive Load, plus
// session timing. 71 columns total (including raw_json).
//
// Setup:
//   1. Open your Google Sheet
//   2. Extensions → Apps Script
//   3. Replace Code.gs with this file
//   4. Save (Ctrl+S)
//   5. From the dropdown at the top, choose initSheet → Run (▶)
//      → this writes the 68 column headers into row 1
//   6. Deploy → Manage deployments → ✏ edit current deployment →
//      Version: New version → Deploy
//      (The URL stays the same; only re-deploy publishes the new code.)
//
// IMPORTANT — about your existing data:
//   The schema is significantly different from v2.0. If you have existing
//   v2 responses in the sheet, the new column layout will NOT line up
//   with them. Recommended path:
//     (a) Duplicate the existing tab as an archive (right-click → Duplicate),
//     (b) On the original tab: select all → delete every row including the
//         header row,
//     (c) Run initSheet (Step 5 above) to write the new headers,
//     (d) Re-deploy.
//   The raw_json column at the end preserves the full submission as a
//   complete JSON record, so you always have a fallback for parsing.
// ═══════════════════════════════════════════════════════════════════

function initSheet() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  if (sheet.getLastRow() > 0) {
    Logger.log("Sheet is not empty. Aborting initSheet to avoid overwriting headers. Clear the sheet manually first.");
    return;
  }
  sheet.appendRow([
    // ── metadata ──
    'submitted_at_server', 'session_id', 'completion_code', 'recruitment_source', 'survey_version',
    'consented_at', 'time_to_complete_seconds', 'user_agent', 'locale',

    // ── Part A: pre-demo measures (6 items) ──
    'A1_iuipc_collection', 'A2_iuipc_awareness', 'A3_prior_eps_comprehension',
    'A4_age_range', 'A5_technical_background', 'A6_privacy_tool_experience',

    // ── Core UCLDP interactions ──
    'slider_LP', 'slider_LA', 'slider_CR',
    'topsis_eps', 'mic_dp_alpha', 'step4_explored_eps', 'highlighted_feature',
    'mae_standard_ldp', 'mae_ucldp', 'mae_improvement_pct',

    // ── PrivAgent extension ──
    'pa_completed', 'pa_LP', 'pa_LA', 'pa_CR',
    'pa_r1', 'pa_r2', 'pa_r3', 'pa_r4', 'pa_r5', 'pa_r6',

    // ── Part B: post-demo measures (5 items) ──
    'B1_post_demo_eps_comprehension', 'B2_demo_tradeoff_clarity',
    'B3_linkage_attack_comprehension',
    'B4_meter', 'B4_identity', 'B4_ecg',
    'B5_self_identified_archetype',

    // ── Part C: ULDP preference elicitation (5 explicit items) ──
    'C2_forced_eps_choice', 'C3_topsis_satisfaction',
    'C4_explanation_quality', 'C5_perceived_control', 'C6_archetype_match',

    // ── Part D: System Usability Scale (10 items + composite) ──
    'SUS1', 'SUS2', 'SUS3', 'SUS4', 'SUS5',
    'SUS6', 'SUS7', 'SUS8', 'SUS9', 'SUS10',
    'SUS_composite_0_100',

    // ── Part E: Trust (4 items) ──
    'T1_trust_recommendation', 'T2_transparency',
    'T3_stability', 'T4_adoption_intention',

    // ── Part F: Cognitive Load (3 items + mean) ──
    'CL1_mental_demand', 'CL2_effort', 'CL3_frustration', 'CL_mean',

    // ── Part G: Open-ended (4 items) ──
    'OE1_elicitation_preference',
    'OE2_confusing_elements', 'OE3_improvement_suggestions', 'OE4_missing_context',

    // ── full record ──
    'raw_json'
  ]);
  Logger.log("Headers written successfully (71 columns).");
}

function doPost(e) {
  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const r = JSON.parse(e.postData.contents);

    // Defensive accessors — every nested block falls back to {} so missing
    // fields don't throw, they just write blank cells.
    const a    = r.pre_demo_part_a || {};
    const core = r.core_uldp || {};
    const sl   = core.slider_weights || {};
    const mae  = core.mae_at_eps || {};
    const pa   = r.privagent_extension || {};
    const paw  = pa.inferred_weights || {};
    const par  = pa.responses || [];
    const b    = r.survey_part_b || {};
    const b4   = b.B4_per_domain_eps || {};
    const c    = r.survey_part_c || {};
    const d    = r.survey_part_d_sus || {};
    const dI   = d.items || {};
    const eT   = r.survey_part_e_trust || {};
    const f    = r.survey_part_f_cognitive_load || {};
    const g    = r.survey_part_g_open || {};

    sheet.appendRow([
      // metadata
      new Date(),
      r.session_id || '',
      r.completion_code || '',
      r.recruitment_source || '',
      r.survey_version || '',
      r.consented_at || '',
      r.time_to_complete_seconds || '',
      r.user_agent || '',
      r.locale || '',

      // Part A
      a.A1_iuipc_collection,
      a.A2_iuipc_awareness,
      a.A3_prior_eps_comprehension,
      a.A4_age_range,
      a.A5_technical_background,
      a.A6_privacy_tool_experience,

      // Core UCLDP
      sl.LP, sl.LA, sl.CR,
      core.topsis_eps, core.mic_dp_alpha, core.step4_explored_eps, core.highlighted_feature,
      mae.standard_ldp, mae.ucldp_micdp, mae.improvement_pct,

      // PrivAgent
      pa.completed === true,
      paw.LP, paw.LA, paw.CR,
      par[0] || '', par[1] || '', par[2] || '', par[3] || '', par[4] || '', par[5] || '',

      // Part B
      b.B1_post_demo_eps_comprehension,
      b.B2_demo_tradeoff_clarity,
      b.B3_linkage_attack_comprehension,
      b4.smart_meter, b4.identity, b4.ecg,
      b.B5_self_identified_archetype,

      // Part C
      c.C2_forced_eps_choice,
      c.C3_topsis_satisfaction,
      c.C4_explanation_quality,
      c.C5_perceived_control,
      c.C6_archetype_match,

      // Part D — SUS items + composite
      dI.SUS1, dI.SUS2, dI.SUS3, dI.SUS4, dI.SUS5,
      dI.SUS6, dI.SUS7, dI.SUS8, dI.SUS9, dI.SUS10,
      d.composite_score,

      // Part E — Trust
      eT.T1_trust_recommendation,
      eT.T2_transparency,
      eT.T3_stability,
      eT.T4_adoption_intention,

      // Part F — Cognitive Load
      f.CL1_mental_demand, f.CL2_effort, f.CL3_frustration, f.mean_cl,

      // Part G — Open-ended
      g.OE1_elicitation_preference,
      g.OE2_confusing_elements,
      g.OE3_improvement_suggestions,
      g.OE4_missing_context,

      // full record
      JSON.stringify(r)
    ]);

    return ContentService
      .createTextOutput(JSON.stringify({ok: true, session_id: r.session_id}))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    Logger.log("doPost error: " + err);
    return ContentService
      .createTextOutput(JSON.stringify({ok: false, error: String(err)}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

// Quick health check — open the /exec URL directly in a browser to verify
function doGet() {
  return ContentService
    .createTextOutput(JSON.stringify({
      ok: true,
      status: 'UCLDP endpoint v3.0 (HCI full instrument) is alive',
      expected_survey_version: 'v3.0_hci_full',
      column_count: 71
    }))
    .setMimeType(ContentService.MimeType.JSON);
}
