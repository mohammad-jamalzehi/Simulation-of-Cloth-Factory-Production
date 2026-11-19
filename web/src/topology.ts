import {
  ArrowDownToLine, Box, CheckCircle2, CircleDot, Combine, Layers3,
  PackageCheck, RotateCcw, ScanLine, Scissors, Shirt, XCircle,
} from 'lucide-react'
import type { NodeId } from './trace'

type Port = 'left' | 'right' | 'top' | 'bottom'
// A lane Y gives long vertical-port routes a horizontal corridor between rows.
type Route = [from: NodeId, to: NodeId, source?: Port, target?: Port, laneY?: number]

export const stages = [
  { id: 'preparation', name: 'Front preparation', y: 0, height: 295, focus: 'RDM' },
  { id: 'assembly', name: 'Garment assembly', y: 304, height: 176, focus: 'JBF' },
  { id: 'finishing', name: 'Inspection & packing', y: 489, height: 146, focus: 'F' },
  { id: 'rework', name: 'Correction & reinspection', y: 644, height: 146, focus: 'INS_2' },
] as const satisfies ReadonlyArray<{ id: string; name: string; y: number; height: number; focus: NodeId }>

export const stations = {
  front: { name: 'Front arrivals', group: 'input', x: 18, y: 110, icon: Shirt, detail: 'Each arrival releases one right and one left front piece.' },
  back: { name: 'Back arrivals', group: 'input', x: 278, y: 338, icon: Layers3, detail: 'Back pieces wait for a completed front and a collar.' },
  collar: { name: 'Collar arrivals', group: 'input', x: 18, y: 338, icon: ArrowDownToLine, detail: 'Collars wait for a completed front and a back.' },
  RDM: { name: 'Right dose', group: 'process', x: 278, y: 42, icon: Scissors, detail: 'Prepares the right front piece for button sewing. Mean service time 1.5 min.' },
  LDM: { name: 'Left dose', group: 'process', x: 278, y: 178, icon: Scissors, detail: 'Prepares the left front piece for button piercing. Mean service time 1.5 min.' },
  BS: { name: 'Button sewing', group: 'process', x: 538, y: 42, icon: CircleDot, detail: 'Sews buttons onto right pieces. Completed pieces wait for a left piece. Mean service time 2.4 min.' },
  BP: { name: 'Button piercing', group: 'process', x: 538, y: 178, icon: CircleDot, detail: 'Pierces the left piece. Completed pieces wait for a right piece. Mean service time 2.4 min.' },
  PS: { name: 'Pocket sewing', group: 'process', x: 798, y: 110, icon: Shirt, detail: 'Accepts one completed right and left piece per front. Mean service time 3 min.' },
  JBF: { name: 'Garment assembly', group: 'process', x: 538, y: 338, icon: Combine, detail: 'Matches a front, back, and collar. Service duration is the maximum of two sampled task times.' },
  SC: { name: 'Sleeve connection', group: 'process', x: 798, y: 338, icon: Combine, detail: 'Attaches sleeves to assembled garments. Mean service time 30 min.' },
  INS: { name: 'First inspection', group: 'quality', x: 798, y: 522, icon: ScanLine, detail: 'Sends 3% of inspected shirts to correction. Other shirts go to folding. Mean service time 7.5 min.' },
  COR: { name: 'Correction', group: 'rework', x: 798, y: 677, icon: RotateCcw, detail: 'Reworks shirts routed from first inspection, then sends them to reinspection. Mean service time 6 min.' },
  INS_2: { name: 'Reinspection', group: 'rework', x: 538, y: 677, icon: ScanLine, detail: 'Sends 3% of reinspected shirts to scrap. Other shirts go to folding. Mean service time 7.5 min.' },
  F: { name: 'Folding', group: 'process', x: 538, y: 522, icon: Layers3, detail: 'Folded shirts collect until 12 are available to release a packing batch. Mean service time 12.6 min.' },
  PA: { name: 'Packing', group: 'process', x: 278, y: 522, icon: Box, detail: 'Packs one batch of 12 shirts at a time. The packing queue counts batches. Mean service time 1.2 min per batch.' },
  packed: { name: 'Packed output', group: 'output', x: 18, y: 522, icon: PackageCheck, detail: 'Each packing completion sends out 12 shirts.' },
  scrap: { name: 'Scrap', group: 'rework', x: 278, y: 677, icon: XCircle, detail: 'Receives shirts that fail reinspection.' },
} satisfies Record<NodeId, { name: string; group: string; x: number; y: number; icon: typeof CheckCircle2; detail: string }>

export const routes: Route[] = [
  ['front', 'RDM'], ['front', 'LDM'], ['RDM', 'BS'], ['LDM', 'BP'],
  ['BS', 'PS'], ['BP', 'PS'], ['PS', 'JBF', 'bottom', 'top', 316],
  ['back', 'JBF'], ['collar', 'JBF', 'bottom', 'bottom', 465],
  ['JBF', 'SC'], ['SC', 'INS', 'bottom', 'top'],
  ['INS', 'F', 'left', 'right'], ['INS', 'COR', 'bottom', 'top'],
  ['COR', 'INS_2', 'left', 'right'], ['INS_2', 'F', 'top', 'bottom'],
  ['INS_2', 'scrap', 'left', 'right'], ['F', 'PA', 'left', 'right'],
  ['PA', 'packed', 'left', 'right'],
]
