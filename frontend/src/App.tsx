import { useState } from 'react'
import { Search, Loader2, CheckCircle, Circle, XCircle, Bot } from 'lucide-react'
import { useResearch } from './hooks/useResearch'
import type { AgentState } from './hooks/useResearch'

function AgentCard({ agent }: { agent: AgentState }) {
  const icons = {
    idle: <Circle className="w-5 h-5 text-gray-400" />,
    running: <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />,
    done: <CheckCircle className="w-5 h-5 text-green-400" />,
    error: <XCircle className="w-5 h-5 text-red-400" />,
  }
  const borders = {
    idle: 'border-gray-700',
    running: 'border-blue-500 shadow-blue-500/20 shadow-lg',
    done: 'border-green-500',
    error: 'border-red-500',
  }
  return (
    <div className={`bg-gray-800 border rounded-lg p-4 transition-all duration-300 ${borders[agent.status]}`}>
      <div className="flex items-center gap-3">
        {icons[agent.status]}
        <div>
          <p className="font-semibold text-white">{agent.label}</p>
          <p className="text-sm text-gray-400">{agent.message}</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const { agents, result, isRunning, error, runResearch } = useResearch()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isRunning) runResearch(query.trim(), 3)
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <Bot className="w-6 h-6 text-blue-400" />
        <h1 className="text-xl font-bold">Multi-Agent Research Dashboard</h1>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* Search */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter a research topic..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            disabled={isRunning}
          />
          <button
            type="submit"
            disabled={isRunning || !query.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-semibold flex items-center gap-2 transition-colors"
          >
            {isRunning
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Running...</>
              : <><Search className="w-4 h-4" /> Research</>
            }
          </button>
        </form>

        {/* Agent Cards */}
        <div>
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Agent Activity</h2>
          <div className="grid grid-cols-2 gap-3">
            {agents.map(agent => <AgentCard key={agent.name} agent={agent} />)}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-500 rounded-lg p-4 text-red-300">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-4">
            {/* Stats */}
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: 'Sources', value: result.stats.sources_scraped },
                { label: 'Claims Checked', value: result.stats.claims_checked },
                { label: 'Supported', value: result.stats.supported },
                { label: 'Confidence', value: `${result.confidence}%` },
              ].map(s => (
                <div key={s.label} className="bg-gray-800 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-400">{s.value}</p>
                  <p className="text-xs text-gray-400 mt-1">{s.label}</p>
                </div>
              ))}
            </div>

            {/* Report */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Research Report</h2>
              <pre className="whitespace-pre-wrap text-gray-300 text-sm font-sans leading-relaxed">
                {result.report}
              </pre>
            </div>

            {/* Sources */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Sources</h3>
              {result.sources.map(url => (
                <a key={url} href={url} target="_blank" rel="noopener noreferrer"
                  className="block text-blue-400 hover:text-blue-300 text-sm truncate">
                  {url}
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}