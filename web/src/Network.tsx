import {
  Background, BackgroundVariant, BaseEdge, Controls, Handle, MarkerType, Position,
  ReactFlow, ReactFlowProvider, getBezierPath, useReactFlow,
} from '@xyflow/react'
import type { Edge, EdgeProps, Node, NodeProps } from '@xyflow/react'
import { useEffect, useMemo, useRef } from 'react'
import { routes, stages, stations } from './topology'
import type { NodeId, Snapshot, StationId, Trace } from './trace'

type StationNode = Node<{
  station: NodeId
  status?: [number, number]
  capacity?: number
  buffer?: number
  activity?: 'dispatch' | 'receiving'
}, 'station'>
type Stage = (typeof stages)[number]
type StageNode = Node<{ stage: Stage }, 'stage'>
type TransferEdge = Edge<{ rework: boolean; current: boolean; pulses: number[]; laneY?: number }, 'transfer'>
const stationWidth = 220
const stationHeight = 104
const transferDuration = 720

const ports = {
  left: Position.Left, right: Position.Right, top: Position.Top, bottom: Position.Bottom,
} as const

function StageBand({ data: { stage } }: NodeProps<StageNode>) {
  return <div className={`stage-band ${stage.id}`} aria-hidden="true"><span>{stage.name}</span></div>
}

function Station({ data, selected }: NodeProps<StationNode>) {
  const { name, group, icon: Icon, detail } = stations[data.station]
  const busy = data.status?.[0] ?? 0
  const queued = data.status?.[1] ?? 0
  const capacity = data.capacity ?? 0
  const machineState = busy === 0 ? 'Idle' : busy === capacity ? 'All busy' : 'Busy'
  const unit = data.station === 'PA' ? 'batches' : 'jobs'
  return (
    <div className={`station ${group} ${data.activity ?? ''} ${selected ? 'selected' : ''}`} title={detail}>
      {Object.entries(ports).map(([id, position]) => (
        <span key={id}>
          <Handle type="target" position={position} id={id} />
          <Handle type="source" position={position} id={id} />
        </span>
      ))}
      <div className="station-heading">
        <span className="node-icon"><Icon size={18} aria-hidden="true" /></span>
        <strong>{name}</strong>
      </div>
      {data.status ? (
        <div className="station-readouts">
          <div className="machine-reading" aria-label={`${machineState}, ${busy} of ${capacity} servers working`}>
            <span className={busy ? 'working' : ''}>{machineState} <b>{busy}/{capacity}</b></span>
            <span className="machine-slots" aria-hidden="true">
              {Array.from({ length: capacity }, (_, index) => <i key={index} className={index < busy ? 'working' : ''} />)}
            </span>
          </div>
          <span className={`queue-reading ${queued > 0 ? 'waiting' : ''}`} title={`${queued} ${unit} waiting`}>
            <b>{queued}</b><span>{data.station === 'PA' ? 'Batches queued' : 'Queued'}</span>
          </span>
        </div>
      ) : (
        <span className="node-buffer">{data.buffer !== undefined
          ? <><b>{data.buffer}</b> pieces waiting</>
          : data.station === 'front' ? 'Feeds right and left pieces'
            : data.station === 'packed' ? '12 shirts per batch' : 'Failed reinspection'}</span>
      )}
    </div>
  )
}

function Token({ path }: { path: string }) {
  const motion = useRef<SVGAnimateMotionElement>(null)
  useEffect(() => motion.current?.beginElement?.(), [])
  return (
    <g className="transfer-token" aria-hidden="true">
      <circle className="token-halo" r="11" />
      <circle className="token-core" r="5.5" />
      <animateMotion ref={motion} path={path} begin="indefinite" dur={`${transferDuration}ms`} fill="freeze" />
    </g>
  )
}

function Transfer({ data, ...props }: EdgeProps<TransferEdge>) {
  let [path, x, y] = getBezierPath(props)
  if (data?.laneY !== undefined) {
    const laneY = data.laneY
    const { sourceX: bendX, targetX: endX } = props
    const direction = Math.sign(endX - bendX)
    const sourceY = Math.sign(laneY - props.sourceY)
    const targetY = Math.sign(props.targetY - laneY)
    const radius = 10
    path = [
      `M ${props.sourceX},${props.sourceY}`,
      `L ${bendX},${laneY - sourceY * radius}`,
      `Q ${bendX},${laneY} ${bendX + direction * radius},${laneY}`,
      `L ${endX - direction * radius},${laneY}`,
      `Q ${endX},${laneY} ${endX},${laneY + targetY * radius}`,
      `L ${props.targetX},${props.targetY}`,
    ].join(' ')
    x = (bendX + endX) / 2
    y = laneY
  }
  const current = data?.current
  const pulses = data?.pulses ?? []
  return (
    <g className={`route ${data?.rework ? 'rework' : ''} ${current ? 'current' : pulses.length ? 'recent' : ''}`}>
      {(current || pulses.length > 0) && <path className="route-glow" d={path} />}
      <BaseEdge id={props.id} path={path} markerEnd={props.markerEnd} />
      {pulses.map((sequence) => <Token key={sequence} path={path} />)}
      {current && <circle className={`transfer-static ${pulses.length ? 'while-moving' : ''}`} cx={x} cy={y} r="5" />}
    </g>
  )
}

const nodeTypes = { station: Station, stage: StageBand }
const edgeTypes = { transfer: Transfer }
const nodeIds = Object.keys(stations) as NodeId[]
const edgeKey = (from: NodeId, to: NodeId) => `${from}:${to}`

const stageNodes: StageNode[] = stages.map((stage) => ({
  id: `stage-${stage.id}`,
  type: 'stage',
  position: { x: 0, y: stage.y },
  style: { width: 1036, height: stage.height, pointerEvents: 'none' },
  data: { stage },
  zIndex: -1,
  draggable: false,
  selectable: false,
  focusable: false,
}))

function NetworkCanvas({ trace, snapshot, cursor, selected, onSelect, animationStart, eventInterval }: {
  trace: Trace
  snapshot: Snapshot
  cursor: number
  selected: NodeId | null
  onSelect: (node: NodeId | null) => void
  animationStart: number | null
  eventInterval: number
}) {
  const { fitView, getZoom, setCenter, viewportInitialized } = useReactFlow()
  const canvas = useRef<HTMLDivElement>(null)
  const fitted = useRef<boolean | null>(null)
  useEffect(() => {
    const panel = canvas.current
    if (!panel || !viewportInitialized) return
    const observer = new ResizeObserver(([entry]) => {
      const compact = entry.contentRect.width < 750
      if (compact !== fitted.current) {
        fitted.current = compact
        if (compact) void setCenter(stations.RDM.x + 75, stations.front.y + 50, { zoom: 0.82 })
        else void fitView({ padding: 0.045 })
      } else if (!compact) {
        void fitView({ padding: 0.045 })
      }
    })
    observer.observe(panel)
    return () => observer.disconnect()
  }, [fitView, setCenter, viewportInitialized])

  function focusStage(id: NodeId) {
    const station = stations[id]
    void setCenter(station.x + stationWidth / 2, station.y + stationHeight / 2, { zoom: Math.max(0.88, getZoom()) })
  }

  const event = trace.events[cursor - 1]
  const currentRoutes = new Set(event?.transfers.map(([from, to]) => edgeKey(from, to)))
  const receivers = new Set(event?.transfers.map(([, to]) => to))
  const pulses = useMemo(() => {
    const recent = new Map<string, number[]>()
    if (animationStart === null) return recent
    // Keep enough events for each marker to finish, bounded by the latest play action.
    const first = Math.max(animationStart, cursor - Math.ceil(transferDuration / eventInterval))
    for (let index = first; index < cursor; index += 1) {
      for (const [from, to] of trace.events[index].transfers) {
        const route = edgeKey(from, to)
        const sequences = recent.get(route) ?? []
        sequences.push(index)
        recent.set(route, sequences)
      }
    }
    return recent
  }, [animationStart, cursor, eventInterval, trace])
  const nodes: StationNode[] = nodeIds.map((id) => ({
    id,
    type: 'station',
    position: { x: stations[id].x, y: stations[id].y },
    selected: selected === id,
    ariaLabel: stations[id].name,
    data: {
      station: id,
      status: snapshot.stations[id as StationId],
      capacity: trace.capacities[id as StationId],
      buffer: id === 'back' || id === 'collar' ? snapshot.buffers[id] : undefined,
      activity: event?.origin === id ? 'dispatch' : receivers.has(id) ? 'receiving' : undefined,
    },
  }))
  const edges: TransferEdge[] = routes.map(([from, to, source = 'right', target = 'left', laneY]) => {
    const id = edgeKey(from, to)
    const rework = stations[from].group === 'rework' || stations[to].group === 'rework'
    const current = currentRoutes.has(id)
    return {
      id,
      source: from,
      target: to,
      sourceHandle: source,
      targetHandle: target,
      type: 'transfer',
      selectable: false,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: rework ? (current ? '#ffd1a0' : '#c68a57') : (current ? '#a1fbd5' : '#9a8ad3'),
        width: 13, height: 13,
      },
      zIndex: current ? 2 : 0,
      data: { rework, current, pulses: pulses.get(id) ?? [], laneY },
    }
  })
  return (
    <>
      <nav className="stage-nav" aria-label="Focus a manufacturing stage">
        {stages.map((stage) => <button type="button" className={stage.id} key={stage.id} onClick={() => focusStage(stage.focus)}>{stage.name}</button>)}
        <button className="focus-dispatch" type="button" disabled={!event} onClick={() => { if (event) focusStage(event.origin) }}>View dispatch</button>
      </nav>
      <div ref={canvas} className="network" aria-label="Interactive garment production network">
        <ReactFlow
          nodes={[...stageNodes, ...nodes]} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
          onNodeClick={(_, node) => onSelect(node.id as NodeId)}
          onSelectionChange={({ nodes: selection }) => { if (selection[0]) onSelect(selection[0].id as NodeId) }}
          onPaneClick={() => onSelect(null)}
          nodesDraggable={false} nodesConnectable={false} edgesFocusable={false}
          fitView fitViewOptions={{ padding: 0.045 }} minZoom={0.18} maxZoom={2.4}
          colorMode="dark" panOnScroll={false} zoomOnScroll={false}
          ariaLabelConfig={{ 'controls.ariaLabel': 'Network zoom controls' }}
        >
          <Background variant={BackgroundVariant.Dots} gap={19} size={1} color="#2b3245" />
          <Controls showInteractive={false} position="bottom-left" />
        </ReactFlow>
      </div>
      <div className="network-key">
        <span><i /> Production route</span>
        <span><i /> Rework route</span>
        <span><i /> Recorded transfer</span>
      </div>
    </>
  )
}

export function Network(props: Parameters<typeof NetworkCanvas>[0]) {
  return <ReactFlowProvider><NetworkCanvas {...props} /></ReactFlowProvider>
}
