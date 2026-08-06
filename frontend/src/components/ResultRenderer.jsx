import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

function ResultRenderer({ text, sqlResult, route }) {
  
  const renderChart = () => {
    if (!sqlResult || !sqlResult.results_json || sqlResult.results_json.length === 0) return null;
    
    const data = sqlResult.results_json;
    const keys = Object.keys(data[0]);
    if (keys.length < 2) return null;
    
    // Simple heuristic: first column is X, second column is Y if it's numeric
    const xAxisKey = keys[0];
    const yAxisKey = keys.find(k => k !== xAxisKey && typeof data[0][k] === 'number');

    if (!yAxisKey) return null;

    // Detect if we should use Line or Bar chart
    // If xAxisKey looks like a date, we might want a line chart, but bar is safer default
    const isDate = xAxisKey.toLowerCase().includes('date') || xAxisKey.toLowerCase().includes('time');

    // Calculate dynamic height for XAxis labels based on longest string
    const maxLabelLength = Math.max(...data.map(d => String(d[xAxisKey] || '').length));
    // Estimate ~5 pixels of height per character at a -45-degree angle
    const dynamicAxisHeight = Math.min(Math.max(Math.ceil(maxLabelLength * 5) + 10, 60), 200);
    // Base chart height + the dynamic axis height
    const chartHeight = 250 + dynamicAxisHeight;

    return (
      <div className="chart-container" style={{ height: chartHeight, width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          {isDate ? (
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
              <XAxis dataKey={xAxisKey} stroke="#000" angle={-45} textAnchor="end" height={dynamicAxisHeight} />
              <YAxis stroke="#000" />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e5e5e5', color: '#000' }} />
              <Line type="monotone" dataKey={yAxisKey} stroke="#000" strokeWidth={2} dot={{ r: 3, fill: '#000' }} />
            </LineChart>
          ) : (
            <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" vertical={false} />
              <XAxis dataKey={xAxisKey} stroke="#000" angle={-45} textAnchor="end" height={dynamicAxisHeight} />
              <YAxis stroke="#000" />
              <Tooltip cursor={{fill: '#f9f9f9'}} contentStyle={{ background: '#fff', border: '1px solid #e5e5e5', color: '#000' }} />
              <Bar dataKey={yAxisKey} fill="#000" />
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
      {route === 'SQL' || route === 'BOTH' ? renderChart() : null}
    </div>
  );
}

export default ResultRenderer;
