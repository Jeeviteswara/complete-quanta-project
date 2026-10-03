'use client'

import { Database, Download, FileJson, FolderOpen, ShieldCheck } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { categories, downloadJSON, type Category } from '@/lib/inspection'
import { categoryLabel, formatRate, type ResearchReport } from '@/lib/research'

export function DatasetPanel({ report, category, onCategory, onDocs }: { report?: ResearchReport; category: Category; onCategory: (value: Category) => void; onDocs: () => void }) {
  const dataset = report?.dataset
  return <>
    <div className="research-intro"><div className="research-icon"><Database /></div><div><h2>MVTec Anomaly Detection</h2><p>{dataset ? `${dataset.total.toLocaleString()} recorded images · ${categoryLabel(category)} · seed ${dataset.seed}` : '15 categories. Pixel-level masks. Strictly separated experiments.'}</p></div><Badge variant="outline">{dataset ? 'Manifest loaded' : report ? 'No manifest' : 'Status unavailable'}</Badge></div>
    <div className="three-grid">{[
      { key: 'train' as const, title: 'Training', sub: 'Normal images only', note: 'Official train/good, split by source before augmentation.' },
      { key: 'val' as const, title: 'Validation', sub: 'Held-out normal images', note: 'A seeded normal subset used only for threshold calibration.' },
      { key: 'test' as const, title: 'Test', sub: 'Normal + defective images', note: 'Official test images and masks. Never used for fitting or tuning.' },
    ].map((split, index) => <article className="research-card" key={split.key}><span className="step-number">0{index + 1}</span><h3>{split.title}</h3><strong className="dataset-count">{dataset ? dataset.counts[split.key].toLocaleString() : '—'}</strong><strong>{split.sub}</strong><p>{split.note}</p></article>)}</div>
    {dataset && <section className="research-card"><div className="section-heading"><h3>Manifest summary</h3><Button variant="outline" size="sm" onClick={() => downloadJSON({ category, ...dataset }, `${category}-dataset-summary.json`)}><Download data-icon="inline-start" />Export summary</Button></div><div className="readiness-row"><span>Normal / defective images</span><strong>{dataset.normal} / {dataset.defective}</strong></div><div className="readiness-row"><span>Recorded masks</span><strong>{dataset.masks}</strong></div><p>{dataset.note}</p></section>}
    <section className="research-card"><div className="section-heading"><h3>Product categories</h3><span>Select a category to explore</span></div><div className="category-grid">{categories.map(item => {
      const entry = report?.categories.find(value => value.category === item)
      return <button className="category-option" aria-pressed={category === item} key={item} onClick={() => onCategory(item)}><FolderOpen size={17} /><span>{categoryLabel(item)}</span><span className="category-unavailable">{!report ? 'Unknown' : entry?.manifest_available ? 'Manifest' : 'Not loaded'}</span></button>
    })}</div></section>
    <div className="research-callout"><ShieldCheck size={22} /><div><h3>{dataset ? 'Good science starts with good splits.' : 'Connect your research data.'}</h3><p>{dataset ? 'The training and evaluation scripts recheck source hashes, masks, and split isolation. A loaded manifest is not proof of current file integrity.' : <>Generate <code>data/{category}-manifest.json</code> with the included dataset inspector. Reports appear here automatically when the local service is connected.</>}</p></div><Button variant="outline" onClick={onDocs}>Dataset setup</Button></div>
  </>
}

export function EvaluationPanel({ report, category, onDocs }: { report?: ResearchReport; category: Category; onDocs: () => void }) {
  const evaluation = report?.evaluation
  const classification = evaluation?.classification
  const localization = evaluation?.localization
  const metrics = [
    ['Precision', classification?.precision], ['Recall', classification?.recall], ['F1 score', classification?.f1],
    ['Balanced accuracy', classification?.balanced_accuracy], ['Pixel IoU', localization?.pixel_iou], ['Pixel Dice', localization?.pixel_dice],
    ['False-positive rate', classification?.false_positive_rate], ['Missed-defect rate', classification?.missed_defect_rate],
  ] as const
  return <>
    <div className="research-intro"><div className="research-icon"><ShieldCheck /></div><div><h2>Measurements, not assumptions</h2><p>{evaluation ? `${evaluation.test_images.toLocaleString()} held-out images · ${categoryLabel(category)} · ${evaluation.elapsed_seconds.toFixed(1)} seconds` : 'Metrics appear only after a real held-out evaluation.'}</p></div><Badge variant="outline">{evaluation ? report?.evaluation_stale ? 'Historical report' : 'Report available' : report ? 'Not evaluated' : 'Status unavailable'}</Badge></div>
    <div className="metrics-grid">{metrics.map(([label, value]) => <article className="research-card" key={label}><span className="metric-label">{label}</span><strong className="empty-metric">{formatRate(value)}</strong><span className="metric-caption">{!evaluation ? 'Awaiting held-out evaluation' : value == null ? 'Undefined for this test set' : report?.evaluation_stale ? 'Historical measured value' : 'Measured on held-out data'}</span></article>)}</div>
    {evaluation ? <>
      <div className="two-grid"><section className="research-card"><div className="section-heading"><h3>Image classification counts</h3></div><p>Region-backed REVIEW counts as positive; it does not establish physical damage.</p><dl className="confusion-grid">{[['True positive', classification!.tp], ['False positive', classification!.fp], ['False negative', classification!.fn], ['True negative', classification!.tn]].map(([label, count]) => <div key={label}><dt>{label}</dt><dd>{count}</dd></div>)}</dl></section>
      <section className="research-card"><div className="section-heading"><h3>Localization evidence</h3></div>{[
        ['Box precision', formatRate(localization!.box_precision)], ['Box recall', formatRate(localization!.box_recall)],
        ['Box IoU · unmatched truth = 0', formatRate(localization!.box_iou_with_unmatched_truth_as_zero)],
        ['Defective-image macro IoU', formatRate(localization!.defective_image_macro_iou)],
        ['Defective-image macro Dice', formatRate(localization!.defective_image_macro_dice)],
        ['Ground-truth regions', localization!.box_counts.ground_truth_count],
      ].map(([label, value]) => <div className="readiness-row" key={label}><span>{label}</span><strong>{value}</strong></div>)}<p>One-to-one matching at IoU ≥ {localization!.box_matching_iou_threshold.toFixed(2)}.</p></section></div>
      {evaluation.quantum_comparison && <section className="research-card"><div className="section-heading"><h3>Quantum vs. classical</h3><Badge variant="outline">No advantage claimed</Badge></div><p>{evaluation.quantum_comparison.protocol}</p><div className="comparison-table-wrap"><table className="comparison-table"><caption className="sr-only">Paired classifier metrics on the same test images</caption><thead><tr><th scope="col">Metric</th><th scope="col">Qiskit kernel</th><th scope="col">Classical SVM</th></tr></thead><tbody>{(['precision', 'recall', 'f1', 'balanced_accuracy'] as const).map(metric => <tr key={metric}><th scope="row">{categoryLabel(metric)}</th><td>{formatRate(evaluation.quantum_comparison!.quantum[metric])}</td><td>{formatRate(evaluation.quantum_comparison!.classical[metric])}</td></tr>)}</tbody></table></div></section>}
      <section className="research-card"><div className="section-heading"><h3>Interpretation and limitations</h3></div><ul className="report-limitations">{evaluation.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul><p>Report availability is not independent validation or manufacturing certification.</p></section>
    </> : <section className="research-card"><div className="section-heading"><h3>Experiment readiness</h3></div>{[
      ['Dataset manifest', !report ? 'Unknown' : report.dataset ? 'Available' : 'Not available'],
      ['Spatial artifact', !report ? 'Unknown' : report.categories.find(item => item.category === category)?.spatial_available ? 'Available' : 'Training required'],
      ['Quantum artifact', !report ? 'Unknown' : report.categories.find(item => item.category === category)?.quantum_available ? 'Available' : 'Optional · not trained'],
      ['Physical-defect validation', 'Requires independent evidence'],
    ].map(([label, value]) => <div className="readiness-row" key={label}><span>{label}</span><Badge variant="outline">{value}</Badge></div>)}</section>}
    <div className="research-callout"><FileJson size={23} /><div><h3>{evaluation ? 'Keep the evidence with the experiment.' : 'Start with a reproducible baseline.'}</h3><p>{evaluation ? 'Export aggregate metrics and limitations. Local image paths are excluded; the full per-image report stays on your workstation.' : <>Run the included evaluator and save <code>artifacts/{category}-evaluation.json</code>. Missing or undefined measurements are never filled with invented scores.</>}</p></div>{evaluation ? <Button variant="outline" onClick={() => downloadJSON({ ...evaluation, evaluation_stale: report?.evaluation_stale }, `${category}-evaluation-summary.json`)}><Download data-icon="inline-start" />Export report</Button> : <Button variant="outline" onClick={onDocs}>Evaluation guide</Button>}</div>
  </>
}
