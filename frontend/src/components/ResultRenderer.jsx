import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line,
} from 'recharts';

// Regex that matches a standard UUID v4 string
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Returns true when a column key or its sample value looks like an internal ID.
 * Used to skip UUID / surrogate key columns as chart X-axis candidates.
 */
function _isIdLike(key, sampleValue) {
  const lk = key.toLowerCase();
  // Key contains "id" as a whole word or suffix  
  if (/\bid\b|_id$|id_/.test(lk)) return true;
  // Sample value is a UUID
  if (typeof sampleValue === 'string' && UUID_RE.test(sampleValue)) return true;
  return false;
}

/**
 * Truncate a long string (e.g. a UUID) to a readable label.
 * Used as a fallback when we have no better column.
 */
function _truncateLabel(val, maxLen = 14) {
  const s = String(val ?? '');
  return s.length > maxLen ? s.slice(0, maxLen) + '…' : s;
}

/**
 * Pick the best X-axis key from a data row.
 * Priority: category/name/label column > any non-numeric non-id column > first column.
 */
function _pickXAxisKey(keys, firstRow) {
  // Preferred: column whose name contains "category", "name", "label", "type", "state", "city"
  const preferred = keys.find(k => {
    const lk = k.toLowerCase();
    return (
      lk.includes('category') ||
      lk.includes('name') ||
      lk.includes('label') ||
      lk.includes('type') ||
      lk.includes('state') ||
      lk.includes('city')
    );
  });
  if (preferred) return preferred;

  // Second choice: any string column that isn't ID-like
  const stringNonId = keys.find(k => {
    const v = firstRow[k];
    return typeof v === 'string' && !_isIdLike(k, v);
  });
  if (stringNonId) return stringNonId;

  // Last resort: first column (may be a UUID — we'll truncate the label)
  return keys[0];
}

function ResultRenderer({ text, sqlResult, route, isStreaming }) {
  const renderChart = () => {
    if (!sqlResult || !sqlResult.results_json) return null;
    const data = sqlResult.results_json;

    // Need at least 2 rows to make a chart meaningful
    if (data.length < 2) return null;

    const keys = Object.keys(data[0]);
    if (keys.length < 2) return null;

    const xAxisKey = _pickXAxisKey(keys, data[0]);
    const yAxisKey = keys.find(k => k !== xAxisKey && typeof data[0][k] === 'number');
    if (!yAxisKey) return null;

    // If the X values are UUID-like, transform the data to use truncated labels
    const xSample = data[0][xAxisKey];
    const needsTruncation =
      typeof xSample === 'string' &&
      (_isIdLike(xAxisKey, xSample) || String(xSample).length > 20);

    const chartData = needsTruncation
      ? data.map(row => ({
          ...row,
          __xLabel: _truncateLabel(row[xAxisKey]),
        }))
      : data;
    const displayKey = needsTruncation ? '__xLabel' : xAxisKey;

    // Date columns → line chart; everything else → bar chart
    const isDate =
      xAxisKey.toLowerCase().includes('date') ||
      xAxisKey.toLowerCase().includes('month') ||
      xAxisKey.toLowerCase().includes('year') ||
      xAxisKey.toLowerCase().includes('time');

    const maxLabelLength = Math.max(
      ...chartData.map(d => String(d[displayKey] || '').length)
    );
    const dynamicAxisHeight = Math.min(Math.max(Math.ceil(maxLabelLength * 5) + 10, 40), 160);
    const chartHeight = 250 + dynamicAxisHeight;

    const tooltipStyle = {
      background: 'var(--chat-bot-bg)',
      border: '1px solid var(--border-color)',
      color: 'var(--text-primary)',
    };

    return (
      <div className="chart-container" style={{ height: chartHeight, width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          {isDate ? (
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis
                dataKey={displayKey}
                stroke="var(--text-primary)"
                angle={-40}
                textAnchor="end"
                height={dynamicAxisHeight}
                tick={{ fontSize: 11 }}
              />
              <YAxis stroke="var(--text-primary)" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line
                type="monotone"
                dataKey={yAxisKey}
                stroke="var(--accent)"
                strokeWidth={2}
                dot={{ r: 3, fill: 'var(--accent)' }}
              />
            </LineChart>
          ) : (
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
              <XAxis
                dataKey={displayKey}
                stroke="var(--text-primary)"
                angle={-40}
                textAnchor="end"
                height={dynamicAxisHeight}
                tick={{ fontSize: 11 }}
              />
              <YAxis stroke="var(--text-primary)" tick={{ fontSize: 11 }} />
              <Tooltip cursor={{ fill: 'var(--hover-bg)' }} contentStyle={tooltipStyle} />
              <Bar dataKey={yAxisKey} fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div>
      <div className="prose">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text || ''}</ReactMarkdown>
      </div>
      {/* Only render charts for SQL/BOTH routes, and only after streaming is complete */}
      {!isStreaming && (route === 'SQL' || route === 'BOTH') && renderChart()}
    </div>
  );
}

export default ResultRenderer;
