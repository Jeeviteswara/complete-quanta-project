'use client'

import useSWR from 'swr'
import { Atom, Cpu, ArrowUpRight, CircleDashed, ArrowRight, Check, Info, RefreshCw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { DatasetPanel, EvaluationPanel } from '@/components/research-results'
import { categories, type Category, type Section, type ServiceStatus } from '@/lib/inspection'
import { categoryLabel, researchFetcher, type ResearchReport } from '@/lib/research'

export function QuantumBanner({ onOpen }: { onOpen: () => void }) {
  return <section className="quantum-banner"><div className="quantum-banner-icon"><Atom size={25} strokeWidth={1.4} /></div><div className="quantum-banner-copy"><div className="flex items-center gap-2"><h3>A classical foundation. A quantum perspective.</h3><Badge variant="outline">Experimental</Badge></div><p>Compare a Qiskit quantum kernel against a classical SVM. Same features. Same splits. Evidence first.</p></div><button onClick={onOpen}>Explore models <ArrowUpRight size={15} /></button></section>
}

export function ResearchPanel({ section, status, category, onCategory, onDocs }: { section: Exclude<Section, 'inspection'>; status?: ServiceStatus; category: Category; onCategory: (value: Category) => void; onDocs: () => void }) {
  const { data, error, isLoading, isValidating, mutate } = useSWR<ResearchReport>(`/api/research?category=${category}`, researchFetcher, { refreshInterval: 30000, shouldRetryOnError: false })
  // Do not present cached measurements as current evidence after a failed refresh.
  const report = error ? undefined : data
  const entry = report?.categories.find(item => item.category === category)
  const training = report?.training
  return <div className="research-page" aria-busy={isLoading}>
    <div className="research-toolbar"><div className="research-category"><label htmlFor="research-category">Product category</label><Select value={category} onValueChange={value => { if (value) onCategory(value as Category) }}><SelectTrigger id="research-category"><SelectValue>{categoryLabel(category)}</SelectValue></SelectTrigger><SelectContent><SelectGroup>{categories.map(item => <SelectItem key={item} value={item}>{categoryLabel(item)}</SelectItem>)}</SelectGroup></SelectContent></Select></div><Button variant="outline" disabled={isValidating} onClick={() => void mutate()}><RefreshCw data-icon="inline-start" className={isValidating ? 'animate-spin' : undefined} />{isValidating ? 'Checking…' : 'Refresh reports'}</Button></div>
    {isLoading && <p role="status" className="muted-copy">Checking local research artifacts…</p>}
    {error && <Alert><Info /><AlertTitle>Research service unavailable</AlertTitle><AlertDescription>{error instanceof Error ? error.message : 'Reports could not be loaded.'}</AlertDescription></Alert>}
    {report?.warnings.map(warning => <Alert key={warning}><Info /><AlertTitle>Report needs attention</AlertTitle><AlertDescription>{warning}</AlertDescription></Alert>)}
    {section === 'dataset' ? <DatasetPanel report={report} category={category} onCategory={onCategory} onDocs={onDocs} /> : section === 'evaluation' ? <EvaluationPanel report={report} category={category} onDocs={onDocs} /> : <>
      <div className="two-grid"><article className="research-card model-card"><div className="flex items-center justify-between"><div className="research-icon"><Cpu /></div><Badge variant="secondary">Spatial baseline</Badge></div><h2>ResNet-18 + patch memory</h2><p>A frozen PyTorch feature extractor and nearest-neighbor patch scoring. Small enough to target an RTX 4060 Laptop GPU.</p><ul className="model-features"><li><Check />Multi-scale spatial feature maps</li><li><Check />Normal-only memory bank</li><li><Check />Validation-calibrated threshold</li><li><Check />Connected-component localization</li></ul><div className="model-state"><CircleDashed size={15} />{!report ? 'Artifact status unavailable' : entry?.spatial_available ? `${categoryLabel(category)} artifact available` : 'Awaiting category training artifact'}</div></article>
      <article className="research-card model-card quantum-model"><div className="flex items-center justify-between"><div className="research-icon"><Atom /></div><Badge variant="outline">Quantum research</Badge></div><h2>Quantum-kernel SVM</h2><p>A genuine 4-qubit Qiskit ZZ feature map, exact statevector fidelity kernel, and a supervised support-vector classifier.</p><ul className="model-features"><li><Check />4 qubits · 2 circuit repetitions</li><li><Check />Train-only PCA and angle scaling</li><li><Check />Matched RBF-SVM comparison</li><li><Check />Image classification only—not a box generator</li></ul><div className="model-state"><CircleDashed size={15} />{!report ? 'Artifact status unavailable' : entry?.quantum_available ? `${categoryLabel(category)} comparison artifact available` : 'Awaiting independent labeled training data'}</div></article></div>
      {training && <section className="research-card"><div className="section-heading"><h3>Recorded training run · {categoryLabel(category)}</h3><Badge variant="outline">Not physical-defect validation</Badge></div>{[
        ['Training / validation images', `${training.training_images} / ${training.validation_images}`], ['Memory patches', training.memory_patches.toLocaleString()],
        ['Training device / seed', `${training.device} / ${training.seed}`], ['Pixel threshold', training.pixel_threshold.toFixed(5)], ['Image threshold', training.image_threshold.toFixed(5)],
      ].map(([label, value]) => <div className="readiness-row" key={label}><span>{label}</span><strong>{value}</strong></div>)}<p>{training.threshold_protocol}</p></section>}
      <section className="research-card"><div className="section-heading"><h3>One pipeline. Independent responsibilities.</h3><Badge variant="outline">No quantum advantage claimed</Badge></div><div className="pipeline-diagram">{['Product image', 'Spatial features', 'Anomaly regions', 'Human review'].map((label, index) => <div className="pipeline-stage" key={label}><span>0{index + 1}</span><strong>{label}</strong>{index < 3 && <ArrowRight size={16} />}</div>)}</div><p>Quantum and classical classifiers compare image-level predictions. They never generate or override spatial boxes.</p></section>
      <div className="research-callout"><Info size={23} /><div><h3>Local compute. Transparent limits.</h3><p>{status?.connected ? `The service reports ${status.device}. Artifact availability does not verify model quality.` : 'Connect the local Python service to inspect your trained artifacts.'} Qiskit uses CPU statevector simulation, not quantum hardware. Target GPU performance still needs measurement.</p></div><Button variant="outline" onClick={onDocs}>Training guide</Button></div>
    </>}
  </div>
}
