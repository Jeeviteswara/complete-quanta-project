import type { Category } from '@/lib/inspection'

export type Classification = {
  precision: number | null
  recall: number | null
  f1: number | null
  balanced_accuracy: number | null
  false_positive_rate: number | null
  missed_defect_rate: number | null
  tp: number
  tn: number
  fp: number
  fn: number
}

export type Evaluation = {
  status: 'evaluated'
  category: Category
  test_images: number
  elapsed_seconds: number
  classification: Classification
  localization: {
    pixel_iou: number | null
    pixel_dice: number | null
    defective_image_macro_iou: number | null
    defective_image_macro_dice: number | null
    box_iou_with_unmatched_truth_as_zero: number | null
    box_precision: number | null
    box_recall: number | null
    box_matching_iou_threshold: number
    box_counts: { tp: number; fp: number; fn: number; ground_truth_count: number; matched_iou_sum: number }
  }
  quantum_comparison: {
    quantum: Classification
    classical: Classification
    protocol: string
    combined_inference_seconds: number
  } | null
  limitations: string[]
}

export type ResearchReport = {
  category: Category
  dataset: {
    counts: { train: number; val: number; test: number }
    total: number
    normal: number
    defective: number
    masks: number
    seed: number
    note: string
  } | null
  training: {
    status: 'trained_not_tested'
    category: Category
    seed: number
    training_images: number
    validation_images: number
    memory_patches: number
    device: string
    pixel_threshold: number
    image_threshold: number
    threshold_protocol: string
  } | null
  evaluation: Evaluation | null
  evaluation_stale: boolean
  warnings: string[]
  categories: {
    category: Category
    manifest_available: boolean
    spatial_available: boolean
    quantum_available: boolean
    evaluation_available: boolean
  }[]
}

export const categoryLabel = (category: string) => category.replaceAll('_', ' ').replace(/^./, letter => letter.toUpperCase())
export const formatRate = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`

export async function researchFetcher(url: string): Promise<ResearchReport> {
  const response = await fetch(url)
  const body = await response.json()
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Research reports could not be loaded.')
  return body
}
