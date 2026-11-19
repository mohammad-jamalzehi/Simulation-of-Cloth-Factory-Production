export type StationId =
  | 'RDM' | 'LDM' | 'BS' | 'BP' | 'PS' | 'JBF'
  | 'SC' | 'INS' | 'COR' | 'INS_2' | 'F' | 'PA'

export type NodeId = StationId | 'front' | 'back' | 'collar' | 'packed' | 'scrap'

export interface Snapshot {
  stations: Record<StationId, [busy: number, queued: number]>
  buffers: Record<'right' | 'left' | 'front' | 'back' | 'collar' | 'folded', number>
}

export interface TraceEvent {
  time: number
  event: string
  origin: NodeId
  transfers: [NodeId, NodeId][]
  state: Snapshot
}

export interface Trace {
  version: 1
  source: string
  sourceSha256: string
  seed: number
  horizon: number
  warmUp: number
  capacities: Record<StationId, number>
  initial: Snapshot
  events: TraceEvent[]
}

export async function loadTrace(signal: AbortSignal): Promise<Trace> {
  const response = await fetch(import.meta.env.BASE_URL + 'traces/seed-42.json', { signal })
  if (!response.ok) throw new Error(`Trace request returned HTTP ${response.status}.`)
  const trace: Trace = await response.json()
  if (trace.version !== 1 || !Array.isArray(trace.events) || !trace.events.length) {
    throw new Error('Unsupported or empty trace.')
  }
  return trace
}
