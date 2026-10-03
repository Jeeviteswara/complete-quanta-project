export const categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'] as const
export type Category = typeof categories[number]
export type Section = 'inspection' | 'dataset' | 'models' | 'evaluation'
export type Region = { id: number; label: string; x: number; y: number; width: number; height: number; score: number | null }
export type Inspection = {
  source: 'illustration' | 'model'
  status: 'NORMAL' | 'DEFECTIVE' | 'REVIEW'
  width: number
  height: number
  boxes: Region[]
  score: number | null
  threshold: number | null
  elapsed_ms: number | null
  heatmap: string | null
  quantum: { prediction: string; classical_prediction: string; margin: number } | null
  warning: string
  category: string
}
export type ServiceStatus = { connected: boolean; models: string[]; quantum_models: string[]; device: string; message: string }
export const sampleInspection: Inspection = {
  source: 'illustration', status: 'DEFECTIVE', width: 1024, height: 1024,
  boxes: [
    { id: 1, label: 'Surface corrosion', x: 601, y: 335, width: 99, height: 93, score: null },
    { id: 2, label: 'Damaged thread', x: 402, y: 507, width: 94, height: 101, score: null },
  ],
  score: null, threshold: null, elapsed_ms: null, heatmap: null, quantum: null, category: 'screw',
  warning: 'AI-generated illustration with manual annotations. Not an MVTec image or a model prediction.',
}
export function downloadJSON(value: unknown, filename: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
