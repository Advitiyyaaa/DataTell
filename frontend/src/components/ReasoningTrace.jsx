import React from 'react';
import { X } from 'lucide-react';

function ReasoningTrace({ data, isOpen, onClose }) {
  if (!isOpen) return <div className="trace-sidebar" />;

  return (
    <div className={`trace-sidebar ${isOpen ? 'open' : ''}`}>
      <div className="trace-header">
        Reasoning Trace
        <button onClick={onClose} style={{ border: 'none', padding: '0.25rem' }} aria-label="Close trace">
          <X size={16} />
        </button>
      </div>
      <div className="trace-content">
        {!data ? (
          <div style={{ color: 'var(--text-secondary)' }}>Waiting for trace data…</div>
        ) : (
          <>
            {/* Route */}
            <div className="trace-section">
              <div className="trace-section-title">Route Taken</div>
              <div style={{ fontWeight: 600 }}>{data.route ?? '—'}</div>
              {data.route_reason && (
                <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {data.route_reason}
                </div>
              )}
            </div>

            {/* SQL Query */}
            {data.sql_result && (
              <div className="trace-section">
                <div className="trace-section-title">SQL Query</div>
                {data.sql_result.sql ? (
                  <pre>{data.sql_result.sql}</pre>
                ) : (
                  <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>No SQL generated</div>
                )}
                {data.sql_result.tool_ms != null && (
                  <div style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                    Execution time: {data.sql_result.tool_ms}ms
                    {data.sql_result.row_count != null && ` · ${data.sql_result.row_count} rows`}
                  </div>
                )}
              </div>
            )}

            {/* RAG Sources */}
            {data.rag_result?.sources?.length > 0 && (
              <div className="trace-section">
                <div className="trace-section-title">RAG Sources</div>
                <ul style={{ paddingLeft: '1.25rem', margin: 0 }}>
                  {data.rag_result.sources.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Performance */}
            {(data.total_ms != null || Object.keys(data.metadata || {}).length > 0) && (
              <div className="trace-section">
                <div className="trace-section-title">Performance</div>
                <table>
                  <tbody>
                    {data.total_ms != null && (
                      <tr>
                        <td>Total Time</td>
                        <td>{data.total_ms}ms</td>
                      </tr>
                    )}
                    {Object.entries(data.metadata || {}).map(([key, val]) => (
                      <tr key={key}>
                        <td>{key.replace(/_/g, ' ')}</td>
                        <td>{val != null ? `${val}ms` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Critic Evaluation */}
            {data.answer_quality && (
              <div className="trace-section">
                <div className="trace-section-title">Critic Evaluation</div>
                <div
                  style={{
                    fontWeight: 600,
                    color: data.answer_quality === 'PASS'
                      ? 'var(--color-pass)'
                      : 'var(--color-fail)',
                  }}
                >
                  {data.answer_quality}
                  {data.retry_count != null && ` (Retries: ${data.retry_count})`}
                </div>
                {data.critique && (
                  <div style={{ marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
                    {data.critique}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default ReasoningTrace;
