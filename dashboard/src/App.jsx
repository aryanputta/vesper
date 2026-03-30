import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';

// ─── Color Constants ──────────────────────────────────────────────────────────
const COLORS = {
  URLLC: '#4a9eff',
  eMBB:  '#50c878',
  mMTC:  '#ffd700',
  bg:    '#0a0e1a',
  card:  '#1a2035',
  border:'#2a3555',
  text:  '#e0e6f0',
  dim:   '#7a8aaa',
};

const STATE_COLORS = {
  NORMAL:    '#50c878',
  CAUTION:   '#ffd700',
  CRITICAL:  '#ff8c42',
  EMERGENCY: '#ff4466',
  RECOVERY:  '#4a9eff',
};

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ─── Utility: fetch with fallback ─────────────────────────────────────────────
async function safeFetch(url, fallback) {
  try {
    const res = await axios.get(url, { timeout: 800 });
    return res.data;
  } catch {
    return fallback;
  }
}

// ─── Styles (inline, no build dependencies) ───────────────────────────────────
const S = {
  app: {
    minHeight: '100vh',
    backgroundColor: COLORS.bg,
    color: COLORS.text,
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    padding: '0 0 24px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 24px',
    backgroundColor: COLORS.card,
    borderBottom: `1px solid ${COLORS.border}`,
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 16 },
  logo: {
    fontSize: 24,
    fontWeight: 800,
    letterSpacing: 4,
    color: COLORS.URLLC,
    textShadow: `0 0 20px ${COLORS.URLLC}66`,
  },
  statusBadge: (online) => ({
    padding: '3px 10px',
    borderRadius: 12,
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1,
    backgroundColor: online ? '#50c87822' : '#ff446622',
    color: online ? COLORS.eMBB : '#ff4466',
    border: `1px solid ${online ? COLORS.eMBB : '#ff4466'}`,
  }),
  scenarioTag: {
    fontSize: 13,
    color: COLORS.dim,
    backgroundColor: '#ffffff0a',
    padding: '4px 12px',
    borderRadius: 8,
    border: `1px solid ${COLORS.border}`,
  },
  uptime: { fontSize: 13, color: COLORS.dim, fontVariantNumeric: 'tabular-nums' },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gridTemplateRows: 'auto',
    gap: 16,
    padding: '16px 24px',
    maxWidth: 1800,
    margin: '0 auto',
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: 12,
    border: `1px solid ${COLORS.border}`,
    padding: 16,
    overflow: 'hidden',
  },
  cardTitle: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 2,
    textTransform: 'uppercase',
    color: COLORS.dim,
    marginBottom: 14,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  dot: (color) => ({
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: color,
    boxShadow: `0 0 6px ${color}`,
  }),
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function UtilBar({ value, color }) {
  const pct = Math.min(100, Math.max(0, value));
  const bg = value > 90 ? '#ff4466' : value > 70 ? '#ffd700' : color;
  return (
    <div style={{ background: '#ffffff10', borderRadius: 4, height: 8, width: '100%', overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', backgroundColor: bg, borderRadius: 4, transition: 'width 0.4s ease' }} />
    </div>
  );
}

function SliceGauge({ name, color, data }) {
  const util = data?.utilization_pct ?? 0;
  const latency = data?.latency_ms ?? '--';
  const loss = data?.packet_loss_pct ?? '--';
  const utilColor = util > 90 ? '#ff4466' : util > 70 ? '#ffd700' : color;

  return (
    <div style={{ flex: 1, backgroundColor: '#ffffff06', borderRadius: 10, padding: 14, border: `1px solid ${COLORS.border}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontWeight: 700, fontSize: 13, color }}>{name}</span>
        <span style={{ fontSize: 20, fontWeight: 800, color: utilColor, fontVariantNumeric: 'tabular-nums' }}>
          {util.toFixed(1)}%
        </span>
      </div>
      <UtilBar value={util} color={color} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 12, color: COLORS.dim }}>
        <span>Latency <b style={{ color: COLORS.text }}>{typeof latency === 'number' ? latency.toFixed(2) : latency}ms</b></span>
        <span>Loss <b style={{ color: COLORS.text }}>{typeof loss === 'number' ? loss.toFixed(3) : loss}%</b></span>
      </div>
    </div>
  );
}

function EVCard({ ev }) {
  const stateColor = STATE_COLORS[ev.state] || COLORS.dim;
  const urgency = ev.urgency_score ?? 0;

  return (
    <div style={{
      backgroundColor: '#ffffff05',
      borderRadius: 8,
      padding: '10px 12px',
      border: `1px solid ${stateColor}44`,
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
      minWidth: 0,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {ev.ev_id}
        </span>
        <span style={{
          fontSize: 10,
          fontWeight: 700,
          padding: '2px 7px',
          borderRadius: 8,
          backgroundColor: `${stateColor}22`,
          color: stateColor,
          border: `1px solid ${stateColor}66`,
          whiteSpace: 'nowrap',
          marginLeft: 4,
        }}>{ev.state}</span>
      </div>
      <div style={{ fontSize: 11, color: COLORS.dim }}>
        Slice: <span style={{ color: COLORS[ev.active_slice] || COLORS.text, fontWeight: 600 }}>{ev.active_slice || '—'}</span>
      </div>
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: COLORS.dim, marginBottom: 2 }}>
          <span>Urgency</span><span style={{ color: COLORS.text }}>{urgency.toFixed(3)}</span>
        </div>
        <UtilBar value={urgency * 100} color={urgency > 0.8 ? '#ff4466' : urgency > 0.5 ? '#ffd700' : COLORS.eMBB} />
      </div>
    </div>
  );
}

function DecisionRow({ d }) {
  const sliceColor = COLORS[d.assigned_slice] || COLORS.dim;
  const rowBg = d.confidence > 0.85
    ? '#50c87810'
    : d.rule_triggered
    ? '#ffd70010'
    : 'transparent';

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '70px 90px 1fr 80px 60px 1fr',
      gap: 8,
      padding: '6px 10px',
      borderRadius: 6,
      backgroundColor: rowBg,
      borderBottom: `1px solid ${COLORS.border}`,
      fontSize: 11,
      alignItems: 'center',
    }}>
      <span style={{ color: COLORS.dim, fontVariantNumeric: 'tabular-nums' }}>
        {new Date(d.timestamp).toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
      </span>
      <span style={{ color: COLORS.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.ev_id}</span>
      <span style={{ color: COLORS.dim, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.event_type}</span>
      <span style={{
        padding: '2px 8px',
        borderRadius: 8,
        backgroundColor: `${sliceColor}22`,
        color: sliceColor,
        fontWeight: 700,
        textAlign: 'center',
        border: `1px solid ${sliceColor}44`,
      }}>{d.assigned_slice}</span>
      <span style={{ color: d.confidence > 0.85 ? COLORS.eMBB : COLORS.dim, fontVariantNumeric: 'tabular-nums', textAlign: 'right' }}>
        {(d.confidence * 100).toFixed(0)}%
      </span>
      <span style={{ color: d.rule_triggered ? '#ffd700' : COLORS.dim, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {d.rule_triggered || '—'}
      </span>
    </div>
  );
}

function EventDot({ event, onClick }) {
  const severityColor = {
    critical: '#ff4466',
    high:     '#ff8c42',
    medium:   '#ffd700',
    low:      '#50c878',
    info:     '#4a9eff',
  }[event.severity] || COLORS.dim;

  return (
    <div
      onClick={() => onClick(event)}
      title={`${event.event_type} — ${event.severity}`}
      style={{
        width: 12,
        height: 12,
        borderRadius: '50%',
        backgroundColor: severityColor,
        border: `2px solid ${severityColor}`,
        boxShadow: `0 0 8px ${severityColor}88`,
        cursor: 'pointer',
        position: 'absolute',
        top: '50%',
        transform: 'translateY(-50%)',
        left: `${event.relativePos * 100}%`,
        transition: 'transform 0.2s',
        zIndex: 2,
      }}
    />
  );
}

function EventPopup({ event, onClose }) {
  if (!event) return null;
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: '#00000088',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 999,
    }} onClick={onClose}>
      <div style={{
        backgroundColor: COLORS.card,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 14,
        padding: 24,
        minWidth: 360,
        maxWidth: 500,
      }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>{event.event_type}</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: COLORS.dim, cursor: 'pointer', fontSize: 18 }}>×</button>
        </div>
        <div style={{ fontSize: 12, color: COLORS.dim, marginBottom: 12 }}>
          <div>EV: <span style={{ color: COLORS.text }}>{event.ev_id}</span></div>
          <div>Time: <span style={{ color: COLORS.text }}>{new Date(event.timestamp).toLocaleString()}</span></div>
          <div>Severity: <span style={{ color: COLORS.text }}>{event.severity?.toUpperCase()}</span></div>
          <div>Slice: <span style={{ color: COLORS[event.assigned_slice] || COLORS.text }}>{event.assigned_slice}</span></div>
        </div>
        {event.shap_features?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1, color: COLORS.dim, marginBottom: 8 }}>
              TOP SHAP FEATURES
            </div>
            {event.shap_features.map((f, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '4px 0', borderBottom: `1px solid ${COLORS.border}` }}>
                <span style={{ color: COLORS.text }}>{f.feature}</span>
                <span style={{ color: f.value > 0 ? '#ff4466' : COLORS.eMBB, fontVariantNumeric: 'tabular-nums' }}>
                  {f.value > 0 ? '+' : ''}{f.value.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  // System state
  const [online, setOnline]         = useState(false);
  const [scenario, setScenario]     = useState('Loading…');
  const [uptimeSec, setUptimeSec]   = useState(0);
  const startTime                   = useRef(Date.now());

  // Panel data
  const [slices, setSlices]         = useState({});
  const [fleet, setFleet]           = useState([]);
  const [latencyHistory, setLatency]= useState([]);
  const [decisions, setDecisions]   = useState([]);
  const [alerts, setAlerts]         = useState([]);
  const [modelStats, setModelStats] = useState(null);

  // UI state
  const [selectedEvent, setSelectedEvent] = useState(null);
  const decisionRef = useRef(null);

  // ── Uptime ticker ────────────────────────────────────────────────────────────
  useEffect(() => {
    const t = setInterval(() => setUptimeSec(Math.floor((Date.now() - startTime.current) / 1000)), 1000);
    return () => clearInterval(t);
  }, []);

  const fmtUptime = (s) => {
    const h = Math.floor(s / 3600).toString().padStart(2, '0');
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${h}:${m}:${sec}`;
  };

  // ── Slice status poll (1s) ───────────────────────────────────────────────────
  const pollSlices = useCallback(async () => {
    const data = await safeFetch(`${API_BASE}/slices/status`, null);
    if (data) {
      setSlices(data);
      setOnline(true);
    } else {
      setOnline(false);
    }
  }, []);

  // ── Fleet poll (1s) ──────────────────────────────────────────────────────────
  const pollFleet = useCallback(async () => {
    const data = await safeFetch(`${API_BASE}/telemetry/fleet/summary`, null);
    if (data?.evs) {
      setFleet(data.evs.slice(0, 20));
      setScenario(data.active_scenario || 'No scenario active');
    }
  }, []);

  // ── Latency history poll (1s) ────────────────────────────────────────────────
  const pollLatency = useCallback(async () => {
    const [urllc, embb, mmtc] = await Promise.all([
      safeFetch(`${API_BASE}/slices/URLLC/metrics`, null),
      safeFetch(`${API_BASE}/slices/eMBB/metrics`, null),
      safeFetch(`${API_BASE}/slices/mMTC/metrics`, null),
    ]);

    setLatency((prev) => {
      const next = [
        ...prev,
        {
          t: prev.length,
          URLLC: urllc?.latency_ms ?? null,
          eMBB:  embb?.latency_ms  ?? null,
          mMTC:  mmtc?.latency_ms  ?? null,
        },
      ].slice(-60);
      return next;
    });
  }, []);

  // ── Decision feed poll (500ms) ───────────────────────────────────────────────
  const pollDecisions = useCallback(async () => {
    const data = await safeFetch(`${API_BASE}/decisions/recent`, null);
    if (data?.decisions) {
      setDecisions(data.decisions.slice(0, 20));
      // Auto-scroll to top of decision list
      if (decisionRef.current) decisionRef.current.scrollTop = 0;
    }
  }, []);

  // ── Alerts poll (2s) ─────────────────────────────────────────────────────────
  const pollAlerts = useCallback(async () => {
    const data = await safeFetch(`${API_BASE}/alerts/history?minutes=30`, null);
    if (data?.alerts) {
      const now = Date.now();
      const horizon = 30 * 60 * 1000;
      const enriched = data.alerts.map((a) => ({
        ...a,
        relativePos: Math.max(0, Math.min(1, 1 - (now - new Date(a.timestamp).getTime()) / horizon)),
      }));
      setAlerts(enriched);
    }
  }, []);

  // ── Model stats poll (2s) ────────────────────────────────────────────────────
  const pollModelStats = useCallback(async () => {
    const data = await safeFetch(`${API_BASE}/ml/stats`, null);
    if (data) setModelStats(data);
  }, []);

  // ── Wire up polling intervals ─────────────────────────────────────────────────
  useEffect(() => {
    pollSlices(); pollFleet(); pollLatency(); pollDecisions(); pollAlerts(); pollModelStats();

    const t1 = setInterval(pollSlices,    1000);
    const t2 = setInterval(pollFleet,     1000);
    const t3 = setInterval(pollLatency,   1000);
    const t4 = setInterval(pollDecisions,  500);
    const t5 = setInterval(pollAlerts,    2000);
    const t6 = setInterval(pollModelStats,2000);

    return () => [t1, t2, t3, t4, t5, t6].forEach(clearInterval);
  }, [pollSlices, pollFleet, pollLatency, pollDecisions, pollAlerts, pollModelStats]);

  // ── Confidence distribution bucketing ────────────────────────────────────────
  const confidenceBuckets = React.useMemo(() => {
    const buckets = [
      { label: '0–0.5',      count: 0, color: '#ff4466' },
      { label: '0.5–0.7',    count: 0, color: '#ffd700' },
      { label: '0.7–0.85',   count: 0, color: '#ff8c42' },
      { label: '0.85–1.0',   count: 0, color: COLORS.eMBB },
    ];
    decisions.forEach((d) => {
      const c = d.confidence ?? 0;
      if (c < 0.5)       buckets[0].count++;
      else if (c < 0.7)  buckets[1].count++;
      else if (c < 0.85) buckets[2].count++;
      else               buckets[3].count++;
    });
    return buckets;
  }, [decisions]);

  // ── Anomaly count ─────────────────────────────────────────────────────────────
  const anomalyCount = alerts.filter(
    (a) => a.severity === 'critical' || a.severity === 'high'
  ).length;

  // ─────────────────────────────────────────────────────────────────────────────
  //  RENDER
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div style={S.app}>

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <header style={S.header}>
        <div style={S.headerLeft}>
          <span style={S.logo}>VESPER</span>
          <span style={S.statusBadge(online)}>{online ? 'ONLINE' : 'OFFLINE'}</span>
          <span style={S.scenarioTag}>{scenario}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{ fontSize: 11, color: COLORS.dim }}>
            UPTIME <span style={{ color: COLORS.text, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{fmtUptime(uptimeSec)}</span>
          </span>
          <span style={{ fontSize: 11, color: COLORS.dim }}>5G EV SLICE ORCHESTRATOR</span>
        </div>
      </header>

      {/* ── MAIN GRID ──────────────────────────────────────────────────────── */}
      <div style={S.grid}>

        {/* Panel 1: Slice Utilization */}
        <div style={{ ...S.card, gridColumn: '1 / 2' }}>
          <div style={S.cardTitle}>
            <div style={S.dot(COLORS.URLLC)} />
            Slice Utilization
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            {['URLLC', 'eMBB', 'mMTC'].map((name) => (
              <SliceGauge key={name} name={name} color={COLORS[name]} data={slices[name]} />
            ))}
          </div>
        </div>

        {/* Panel 3: Latency Chart */}
        <div style={{ ...S.card, gridColumn: '2 / 4' }}>
          <div style={S.cardTitle}>
            <div style={S.dot('#a78bfa')} />
            Real-time Latency (ms) — Last 60s
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={latencyHistory} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: COLORS.dim }} tickLine={false} axisLine={false} label={{ value: 'seconds ago', position: 'insideBottom', fill: COLORS.dim, fontSize: 10, offset: -2 }} />
              <YAxis
                scale="log"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: COLORS.dim }}
                tickLine={false}
                axisLine={false}
                width={36}
                tickFormatter={(v) => `${v}`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: COLORS.dim }}
                formatter={(value, name) => [`${value?.toFixed(2)}ms`, name]}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
              <Line type="monotone" dataKey="URLLC" stroke={COLORS.URLLC} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="eMBB"  stroke={COLORS.eMBB}  dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="mMTC"  stroke={COLORS.mMTC}  dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Panel 2: EV Fleet Grid */}
        <div style={{ ...S.card, gridColumn: '1 / 3' }}>
          <div style={S.cardTitle}>
            <div style={S.dot(COLORS.eMBB)} />
            EV Fleet State ({fleet.length} vehicles)
          </div>
          {fleet.length === 0 ? (
            <div style={{ color: COLORS.dim, fontSize: 12, textAlign: 'center', padding: '24px 0' }}>
              Waiting for fleet telemetry…
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
              gap: 8,
              maxHeight: 340,
              overflowY: 'auto',
            }}>
              {fleet.map((ev) => <EVCard key={ev.ev_id} ev={ev} />)}
            </div>
          )}
        </div>

        {/* Panel 6: Model Confidence + Anomaly */}
        <div style={{ ...S.card, gridColumn: '3 / 4' }}>
          <div style={S.cardTitle}>
            <div style={S.dot('#a78bfa')} />
            Model Confidence + Anomalies
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={confidenceBuckets} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="label" tick={{ fontSize: 10, fill: COLORS.dim }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10, fill: COLORS.dim }} tickLine={false} axisLine={false} width={24} />
              <Tooltip contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 11 }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {confidenceBuckets.map((b, i) => <Cell key={i} fill={b.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, gap: 8 }}>
            <div style={{ flex: 1, backgroundColor: '#ffffff06', borderRadius: 8, padding: '10px 12px', textAlign: 'center', border: `1px solid ${COLORS.border}` }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: anomalyCount > 0 ? '#ff4466' : COLORS.eMBB }}>{anomalyCount}</div>
              <div style={{ fontSize: 10, color: COLORS.dim, marginTop: 2 }}>ANOMALIES (30min)</div>
            </div>
            <div style={{ flex: 1, backgroundColor: '#ffffff06', borderRadius: 8, padding: '10px 12px', textAlign: 'center', border: `1px solid ${COLORS.border}` }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: modelStats?.inference_p99_ms < 2 ? COLORS.eMBB : '#ff4466' }}>
                {modelStats?.inference_p99_ms != null ? `${modelStats.inference_p99_ms.toFixed(2)}ms` : '—'}
              </div>
              <div style={{ fontSize: 10, color: COLORS.dim, marginTop: 2 }}>ML INFERENCE P99</div>
            </div>
          </div>
        </div>

        {/* Panel 4: Decision Feed */}
        <div style={{ ...S.card, gridColumn: '1 / 3' }}>
          <div style={S.cardTitle}>
            <div style={S.dot(COLORS.mMTC)} />
            Live Decision Feed
            <span style={{ marginLeft: 'auto', fontSize: 10, fontWeight: 400, color: COLORS.dim }}>polled 500ms</span>
          </div>
          <div style={{ fontSize: 10, color: COLORS.dim, display: 'grid', gridTemplateColumns: '70px 90px 1fr 80px 60px 1fr', gap: 8, padding: '0 10px 6px', borderBottom: `1px solid ${COLORS.border}` }}>
            <span>TIME</span><span>EV ID</span><span>EVENT</span><span>SLICE</span><span>CONF</span><span>RULE</span>
          </div>
          <div ref={decisionRef} style={{ maxHeight: 260, overflowY: 'auto' }}>
            {decisions.length === 0
              ? <div style={{ color: COLORS.dim, fontSize: 12, textAlign: 'center', padding: '24px 0' }}>Waiting for decisions…</div>
              : decisions.map((d, i) => <DecisionRow key={i} d={d} />)
            }
          </div>
        </div>

        {/* Panel 5: Safety Event Timeline */}
        <div style={{ ...S.card, gridColumn: '3 / 4' }}>
          <div style={S.cardTitle}>
            <div style={S.dot('#ff4466')} />
            Safety Event Timeline (30min)
          </div>
          <div style={{ position: 'relative', height: 24, backgroundColor: '#ffffff06', borderRadius: 4, margin: '8px 0 16px' }}>
            {/* Timeline track */}
            <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: 2, backgroundColor: COLORS.border, transform: 'translateY(-50%)' }} />
            {alerts.map((evt, i) => (
              <EventDot key={i} event={evt} onClick={setSelectedEvent} />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: COLORS.dim, marginBottom: 12 }}>
            <span>30 min ago</span>
            <span>now</span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {[['critical','#ff4466'],['high','#ff8c42'],['medium','#ffd700'],['low','#50c878'],['info','#4a9eff']].map(([sev, col]) => (
              <span key={sev} style={{ fontSize: 10, color: col, display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: col, display: 'inline-block' }} />
                {sev}
              </span>
            ))}
          </div>
          <div style={{ marginTop: 12, maxHeight: 100, overflowY: 'auto' }}>
            {alerts.slice(0, 5).map((a, i) => (
              <div key={i} onClick={() => setSelectedEvent(a)} style={{ fontSize: 11, padding: '4px 0', borderBottom: `1px solid ${COLORS.border}`, cursor: 'pointer', color: COLORS.dim, display: 'flex', gap: 8 }}>
                <span style={{ color: { critical:'#ff4466', high:'#ff8c42', medium:'#ffd700', low:'#50c878', info:'#4a9eff' }[a.severity] || COLORS.dim }}>
                  ●
                </span>
                <span style={{ color: COLORS.text }}>{a.ev_id}</span>
                <span>{a.event_type}</span>
              </div>
            ))}
            {alerts.length === 0 && (
              <div style={{ color: COLORS.dim, fontSize: 12, textAlign: 'center', padding: '12px 0' }}>No events in last 30 min</div>
            )}
          </div>
        </div>

      </div>

      {/* Event popup */}
      {selectedEvent && <EventPopup event={selectedEvent} onClose={() => setSelectedEvent(null)} />}

    </div>
  );
}
