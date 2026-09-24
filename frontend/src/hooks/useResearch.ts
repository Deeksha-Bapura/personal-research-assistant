import { useState, useCallback } from 'react'

export type AgentStatus = 'idle' | 'running' | 'done' | 'error'

export interface AgentState {
  name: string
  label: string
  status: AgentStatus
  message: string
}

export interface ResearchResult {
  report: string
  sources: string[]
  confidence: number
  stats: {
    sources_scraped: number
    claims_checked: number
    supported: number
    contradicted: number
    unverified: number
  }
}

const INITIAL_AGENTS: AgentState[] = [
  { name: 'scraper', label: 'Scraper', status: 'idle', message: 'Waiting...' },
  { name: 'summarizer', label: 'Summarizer', status: 'idle', message: 'Waiting...' },
  { name: 'fact_checker', label: 'Fact Checker', status: 'idle', message: 'Waiting...' },
  { name: 'synthesizer', label: 'Synthesizer', status: 'idle', message: 'Waiting...' },
]

export function useResearch() {
  const [agents, setAgents] = useState<AgentState[]>(INITIAL_AGENTS)
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const updateAgent = (name: string, patch: Partial<AgentState>) => {
    setAgents(prev => prev.map(a => a.name === name ? { ...a, ...patch } : a))
  }

  const reset = () => {
    setAgents(INITIAL_AGENTS)
    setResult(null)
    setError(null)
  }

  const runResearch = useCallback(async (query: string, numSources: number = 3) => {
    reset()
    setIsRunning(true)

    const response = await fetch('/research/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, num_sources: numSources }),
    })

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      let eventType = ''
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.replace('event:', '').trim()
        } else if (line.startsWith('data:')) {
          const data = JSON.parse(line.replace('data:', '').trim())
          if (eventType === 'agent_start') {
            updateAgent(data.agent, { status: 'running', message: data.message })
          } else if (eventType === 'agent_done') {
            updateAgent(data.agent, { status: 'done', message: data.message })
          } else if (eventType === 'report') {
            setResult(data)
          } else if (eventType === 'error') {
            setError(data.message)
          }
        }
      }
    }
    setIsRunning(false)
  }, [])

  return { agents, result, isRunning, error, runResearch }
}