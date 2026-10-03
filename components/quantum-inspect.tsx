'use client'

import { useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import { ArrowRight, ArrowUpRight, Atom, BookOpen, ChevronDown, ChevronRight, CircleHelp, Cpu, FileImage, FlaskConical, ImagePlus, Info, LoaderCircle, Menu, Plus, ScanLine, Settings2, ShieldCheck, Upload, X, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { WorkspaceSidebar } from '@/components/workspace-sidebar'
import { InspectionViewer } from '@/components/inspection-viewer'
import { QuantumBanner, ResearchPanel } from '@/components/research-panels'
import { categories, sampleInspection, type Category, type Inspection, type Section, type ServiceStatus } from '@/lib/inspection'
import { cn } from '@/lib/utils'

const sectionCopy = {
  inspection: { title: 'See the defect. Know the difference.', description: 'Inspect product surfaces with spatial AI and experimental quantum intelligence.', breadcrumb: 'Image inspection' },
  dataset: { title: 'Better data. More reliable evidence.', description: 'Understand your dataset before you train, calibrate, or compare.', breadcrumb: 'Dataset explorer' },
  models: { title: 'Two approaches. One fair comparison.', description: 'A practical spatial baseline and a genuine quantum research model.', breadcrumb: 'Model laboratory' },
  evaluation: { title: 'Let the experiments speak.', description: 'Measure classification and localization independently, on held-out data.', breadcrumb: 'Evaluation' },
}
async function statusFetcher(url: string): Promise<ServiceStatus> { const response = await fetch(url); if (!response.ok) throw new Error('Status unavailable'); return response.json() }

export function QuantumInspect() {
  const [section, setSection] = useState<Section>('inspection')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [docsOpen, setDocsOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [quantumEnabled, setQuantumEnabled] = useState(true)
  const [category, setCategory] = useState<Category>('screw')
  const [file, setFile] = useState<File | null>(null)
  const [image, setImage] = useState<string | null>('/images/metal-inspection.png')
  const [result, setResult] = useState<Inspection | null>(sampleInspection)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const activeRequest = useRef<AbortController | null>(null)
  const selectionVersion = useRef(0)
  const { data: serviceStatus, error: statusError, isLoading: checkingStatus, mutate: refreshStatus } = useSWR<ServiceStatus>('/api/status', statusFetcher, { refreshInterval: 30000, shouldRetryOnError: false })
  const status = statusError ? undefined : serviceStatus
  const modelReady = Boolean(status?.connected && status.models.includes(category))
  const readiness = checkingStatus ? 'Checking local service…' : !status?.connected ? 'Connect the local Python service to run inference.' : !modelReady ? `Train a ${category.replaceAll('_', ' ')} spatial model before running an inspection.` : quantumEnabled && !status.quantum_models.includes(category) ? 'Spatial model ready. Quantum comparison is not trained for this category.' : 'Ready to inspect with your local model.'
  useEffect(() => () => { if (image?.startsWith('blob:')) URL.revokeObjectURL(image) }, [image])
  useEffect(() => () => { activeRequest.current?.abort() }, [])

  function cancelInspection() { selectionVersion.current++; activeRequest.current?.abort(); activeRequest.current = null; setBusy(false); setError('') }
  function changeCategory(value: Category) {
    if (value === category) return
    cancelInspection(); setCategory(value); setResult(null)
  }
  async function selectFile(selected?: File) {
    if (!selected) return
    cancelInspection()
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(selected.type)) { setError('Choose a PNG, JPEG, or WebP image.'); return }
    if (selected.size > 10 * 1024 * 1024) { setError('Your image exceeds the 10 MB upload limit.'); return }
    const version = selectionVersion.current
    try {
      const bitmap = await createImageBitmap(selected)
      const tooLarge = bitmap.width * bitmap.height > 25_000_000
      bitmap.close()
      if (version !== selectionVersion.current) return
      if (tooLarge) { setError('Choose an image smaller than 25 megapixels.'); return }
      setFile(selected); setImage(URL.createObjectURL(selected)); setResult(null)
    } catch { if (version === selectionVersion.current) setError('This file could not be decoded as an image.') }
  }
  function loadSample() { cancelInspection(); setFile(null); setCategory('screw'); setImage('/images/metal-inspection.png'); setResult(sampleInspection) }
  function reset() { cancelInspection(); setFile(null); setImage(null); setResult(null); if (input.current) input.current.value = ''; navigate('inspection') }
  async function inspect() {
    if (!image || !modelReady) return
    cancelInspection()
    setResult(null)
    const version = selectionVersion.current
    const controller = new AbortController()
    activeRequest.current = controller
    setBusy(true)
    try {
      const selected = file ?? new File([await (await fetch('/images/metal-inspection.png', { signal: controller.signal })).blob()], 'illustrative-bolt.png', { type: 'image/png' })
      const form = new FormData(); form.append('image', selected); form.append('category', category); form.append('quantum', String(quantumEnabled))
      const response = await fetch('/api/inspect', { method: 'POST', body: form, signal: controller.signal })
      const body = await response.json()
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'The inspection could not be completed. Check your image and model configuration.')
      if (selectionVersion.current === version) setResult(body)
    } catch (caught) {
      if (caught instanceof Error && caught.name !== 'AbortError' && selectionVersion.current === version) setError(caught.message)
    } finally { if (selectionVersion.current === version) { setBusy(false); activeRequest.current = null } }
  }
  function navigate(value: Section) { setSection(value); window.scrollTo({ top: 0, behavior: 'instant' }) }
  const copy = sectionCopy[section]
  return <div className="app-shell" data-app-ready="true">
    <WorkspaceSidebar section={section} onNavigate={navigate} onDocs={() => setDocsOpen(true)} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
    {mobileOpen && <button className="sidebar-scrim" aria-label="Dismiss navigation" onClick={() => setMobileOpen(false)} />}
    <div className="main-shell"><header className="topbar"><div className="breadcrumb"><button aria-label="Open navigation" className="mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button><span>Workspace</span><ChevronRight size={13} /><strong>{copy.breadcrumb}</strong></div><div className="topbar-actions"><span className="research-tag"><FlaskConical size={13} />RESEARCH PREVIEW</span><span className="topbar-divider" /><button aria-label="Open help" onClick={() => setDocsOpen(true)}><CircleHelp size={18} /></button><span className="profile-avatar" aria-label="Local researcher">QI</span></div></header>
    <main className="main-content"><div className="page-heading"><div><div className="eyebrow"><span />VISUAL QUALITY INTELLIGENCE</div><h1>{copy.title}</h1><p>{copy.description}</p></div><Button variant="outline" onClick={reset}><Plus data-icon="inline-start" />New inspection</Button></div>
    <div className="system-strip"><div><span className={cn('status-dot', !status?.connected && 'status-offline')} /><strong>{status?.connected ? 'Local service connected' : 'Local service not connected'}</strong><span className="strip-detail">{status?.models.length ? `${status.models.length} trained categories` : 'Model training required'}</span></div><div><Cpu size={14} /><span>RTX 4060 compatible target</span><span className="strip-divider" /><ShieldCheck size={14} /><span>Evidence-based localization</span></div></div>
    {section === 'inspection' ? <><div className="workspace-grid"><aside className="configuration-column"><section className="input-card"><div className="section-heading"><h2>Inspection setup</h2><Settings2 size={16} /></div><FieldGroup><Field><FieldLabel>Product image</FieldLabel><input ref={input} type="file" accept="image/png,image/jpeg,image/webp" aria-label="Upload product image" className="sr-only" onChange={event => { void selectFile(event.target.files?.[0]); event.target.value = '' }} /><button type="button" className={cn('upload-zone', dragging && 'dragging', file && 'has-file')} onClick={() => input.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); void selectFile(event.dataTransfer.files[0]) }}><span className="upload-icon">{file ? <FileImage size={21} /> : <Upload size={21} />}</span><strong>{file ? file.name : 'Drop your image here'}</strong><span>{file ? `${(file.size / 1024).toFixed(0)} KB · Click to replace` : <>or <b>browse files</b></>}</span><small>PNG, JPG, WebP · up to 10 MB</small></button></Field>
      <Field><FieldLabel htmlFor="category">Product category</FieldLabel><Select value={category} onValueChange={value => { if (value) changeCategory(value as Category) }}><SelectTrigger id="category" className="w-full"><SelectValue>{category.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase())}</SelectValue></SelectTrigger><SelectContent><SelectGroup>{categories.map(item => <SelectItem value={item} key={item}>{item.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase())}</SelectItem>)}</SelectGroup></SelectContent></Select><p className="field-hint">Use the category your model was trained on.</p></Field>
      <Field><FieldLabel>Detection model</FieldLabel><div className="model-selection"><span className="model-icon"><Cpu size={16} /></span><span><strong>Spatial anomaly model</strong><small>ResNet-18 · patch memory</small></span><CheckCircle2 size={15} className="text-primary ml-auto" /></div></Field>
      <Field orientation="horizontal"><Atom size={17} className="text-primary" /><div className="flex-1"><FieldLabel htmlFor="quantum">Quantum comparison</FieldLabel><p className="field-hint">Experimental · Qiskit</p></div><Switch id="quantum" aria-label="Include quantum comparison" size="sm" checked={quantumEnabled} onCheckedChange={value => { cancelInspection(); setQuantumEnabled(value); if (result?.source === 'model') setResult(null) }} /></Field></FieldGroup>
      <button className="settings-toggle" onClick={() => setSettingsOpen(!settingsOpen)} aria-expanded={settingsOpen}><Settings2 size={14} />Detection settings<ChevronDown size={14} className={cn('ml-auto transition-transform', settingsOpen && 'rotate-180')} /></button>{settingsOpen && <div className="settings-details"><div><span>Threshold</span><strong>Validation-calibrated</strong></div><div><span>Min. region area</span><strong>32 source pixels</strong></div><p>Test images never tune the operating threshold. Anomalies require human review.</p></div>}
      <Button className="run-button w-full" size="lg" disabled={!image || busy || !modelReady} aria-describedby="inspection-readiness" onClick={() => void inspect()}>{busy ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : <ScanLine data-icon="inline-start" />}{busy ? 'Inspecting…' : 'Run inspection'}{!busy && <ArrowRight data-icon="inline-end" />}</Button>{busy && <Button variant="outline" className="cancel-inspection w-full" onClick={cancelInspection}>Cancel request</Button>}<p id="inspection-readiness" className="readiness-note" role="status">{readiness}{busy && ' Cancel stops waiting; local inference may finish in the background.'}</p>{!modelReady && <Button variant="link" size="sm" disabled={checkingStatus} onClick={() => void refreshStatus()}>Recheck connection</Button>}<p className="private-note"><ShieldCheck size={12} />Images are processed, not stored.</p></section>
      <section className="sample-card"><div className="flex items-center gap-2"><ImagePlus size={15} /><h3>Take a closer look</h3></div><p>Explore the workspace with an annotated illustration.</p><button className="sample-file" onClick={loadSample}><img src="/images/metal-inspection.png" alt="" /><span><strong>Metal fastener</strong><small>2 manual annotations</small></span><ArrowRight size={15} /></button><span className="sample-disclaimer">Sample only. Not a model result.</span></section>
    </aside><div className="inspection-column">{error && <Alert variant="destructive"><Info /><AlertTitle>Inspection unavailable</AlertTitle><AlertDescription>{error}</AlertDescription><button className="alert-dismiss" aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></Alert>}<InspectionViewer key={`${image}-${category}-${result?.elapsed_ms ?? 'pending'}`} image={image} filename={file?.name ?? (image ? 'metal_fastener_illustration.png' : '')} result={result} busy={busy} /><QuantumBanner onOpen={() => navigate('models')} /></div></div><div className="workspace-footnote"><ShieldCheck size={14} /><span>Every box needs spatial evidence. Every result needs context.</span><button onClick={() => setDocsOpen(true)}>How it works <ArrowUpRight size={13} /></button></div></> : <ResearchPanel section={section} status={status} category={category} onCategory={changeCategory} onDocs={() => setDocsOpen(true)} />}
    </main><footer className="main-footer"><span>QuantumInspect <span className="footer-dot">/</span> Research with rigor.</span><span>PyTorch <span>·</span> Qiskit <span>·</span> MVTec AD</span></footer></div>
    <Dialog open={docsOpen} onOpenChange={setDocsOpen}><DialogContent className="max-w-2xl"><DialogHeader><div className="dialog-eyebrow"><BookOpen size={17} />QUANTUMINSPECT GUIDE</div><DialogTitle>From image to evidence.</DialogTitle><DialogDescription>A local-first research pipeline. The interface never substitutes demonstration data for real inference.</DialogDescription></DialogHeader><div className="documentation-content"><section><h3>01 · Prepare your dataset</h3><p>Download MVTec AD from the official provider and review its non-commercial license. Keep the official test images untouched. The dataset inspector verifies masks and checks duplicate content across splits.</p></section><section><h3>02 · Train on your Windows workstation</h3><p>The included <code>README.md</code> covers Windows 11, CUDA-enabled PyTorch, local FastAPI setup, category training, and evaluation. The baseline uses frozen ResNet-18 features instead of DINOv2 to reduce memory requirements.</p></section><section><h3>03 · Keep quantum experimental</h3><p>The Qiskit model uses a 4-qubit fidelity kernel. Supply independently collected labeled train/validation images; MVTec test defects must not become training data. Both SVMs use the same reduced features and tuning budget.</p></section><section><h3>04 · Inspect, then verify</h3><p>Upload a product image, select the matching category, and run an inspection. Real heatmaps and coordinates come only from a trained backend. Anomaly-only findings are marked REVIEW, not asserted to be physical damage.</p></section><div className="docs-caveat"><Info size={17} /><p>The research panels display your local manifests and aggregate evaluation reports when available. Artifact presence is not proof of model quality. Shadows and reflections can still produce anomalies; independently labeled hard negatives and physical-defect masks are needed for reliable production decisions.</p></div></div><Button onClick={() => setDocsOpen(false)}>Back to workspace <ArrowRight data-icon="inline-end" /></Button></DialogContent></Dialog>
  </div>
}
