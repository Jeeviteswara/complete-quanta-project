'use client'

import { useRef, useState } from 'react'
import { ScanLine, ImageIcon, Flame, ZoomIn, ZoomOut, Maximize, Crosshair, CircleAlert, Download, Info, RotateCcw, Layers, CheckCircle2 } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import { downloadJSON, type Inspection } from '@/lib/inspection'

export function InspectionViewer({ image, filename, result, busy }: { image: string | null; filename: string; result: Inspection | null; busy: boolean }) {
  const [tab, setTab] = useState('boxes')
  const [showLabels, setShowLabels] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [selected, setSelected] = useState<number | null>(null)
  const surface = useRef<HTMLDivElement>(null)
  const sample = result?.source === 'illustration'
  const regions = result?.boxes ?? []
  return <section className="viewer-card" aria-label="Image inspection viewer">
    <div className="viewer-top"><div className="flex items-center gap-2"><ScanLine size={17} /><h2>Inspection workspace</h2></div><Badge variant="outline">{sample ? 'Illustrative sample' : image ? 'Your image' : 'No image'}</Badge></div>
    <div className="viewer-body"><div className="image-column">
      <div className="viewer-toolbar"><Tabs value={tab} onValueChange={value => setTab(String(value))}><TabsList><TabsTrigger value="original"><ImageIcon />Original</TabsTrigger><TabsTrigger value="boxes"><ScanLine />Bounding boxes</TabsTrigger><TabsTrigger value="heatmap"><Flame />Heatmap</TabsTrigger></TabsList></Tabs><span className="image-channel">RGB</span></div>
      <div className={cn('image-surface', busy && 'is-scanning', zoom > 1 && 'is-zoomed')} ref={surface}>
        {image ? <div className="image-frame" style={{ transform: `scale(${zoom})` }}>
          {/* Native image dimensions anchor every overlay to the same coordinate space. */}
          <img src={image} alt={sample ? 'Illustrative steel bolt with surface corrosion and a damaged thread' : 'Uploaded product for defect inspection'} className="inspection-image" />
          {tab === 'heatmap' && result?.heatmap && <img src={result.heatmap} alt="Model anomaly score heatmap" className="heatmap-overlay" />}
          {tab === 'boxes' && result && regions.map((region, index) => <button key={region.id} aria-label={`Region ${region.id}: ${region.label}`} className={cn('defect-box', index % 2 === 1 && 'secondary-region', selected === region.id && 'selected-region')} style={{ left: `${region.x / result.width * 100}%`, top: `${region.y / result.height * 100}%`, width: `${region.width / result.width * 100}%`, height: `${region.height / result.height * 100}%` }} onClick={() => setSelected(region.id)}>{showLabels && <span className="box-label">{String(region.id).padStart(2, '0')} <span>{sample ? region.label : 'Anomaly region'}</span></span>}</button>)}
        </div> : <div className="viewer-empty"><ImageIcon size={34} strokeWidth={1.2} /><strong>A closer look starts here</strong><p>Upload a product image to inspect its surface.</p></div>}
        {tab === 'heatmap' && !result?.heatmap && <div className="heatmap-unavailable"><Flame size={25} /><strong>No model heatmap yet</strong><p>{sample ? 'Manual annotations are not an anomaly heatmap.' : 'Run a trained model to generate a spatial heatmap.'}</p></div>}
        <div className="canvas-top-label"><span className="status-dot" />{busy ? 'ANALYZING IMAGE' : 'IMAGE VIEWPORT'}</div>
        {sample && <span className="sample-watermark">ILLUSTRATION · NOT MODEL OUTPUT</span>}
        <div className="zoom-controls"><Button variant="ghost" size="icon-sm" aria-label="Zoom out" onClick={() => setZoom(Math.max(.75, zoom - .25))}><ZoomOut /></Button><span>{Math.round(zoom * 100)}%</span><Button variant="ghost" size="icon-sm" aria-label="Zoom in" onClick={() => setZoom(Math.min(2.5, zoom + .25))}><ZoomIn /></Button><span className="control-divider" /><Button variant="ghost" size="icon-sm" aria-label="Reset zoom" onClick={() => setZoom(1)}><RotateCcw /></Button><Button variant="ghost" size="icon-sm" aria-label="Expand image" onClick={() => { if (document.fullscreenElement) void document.exitFullscreen(); else void surface.current?.requestFullscreen().catch(() => setZoom(1.5)) }}><Maximize /></Button></div>
      </div>
      <div className="image-statusbar"><span className="flex items-center gap-2"><ImageIcon size={13} /><span className="truncate">{filename || 'No image selected'}</span></span><span>{result ? `${result.width} × ${result.height} px` : 'Original resolution'}</span></div>
      <div className="overlay-options"><div className="flex items-center gap-2"><Switch id="show-labels" aria-label="Show region labels" size="sm" checked={showLabels} onCheckedChange={setShowLabels} /><label htmlFor="show-labels">Show region labels</label></div><span><span className="legend-square" />Localized region</span></div>
    </div>
    <aside className="results-column" aria-label="Inspection results"><div className="results-heading"><h3>Inspection results</h3><span className="tiny-label">{sample ? 'SAMPLE' : 'LIVE'}</span></div>
      {result ? <><div className={cn('result-verdict', result.status === 'NORMAL' && 'normal-verdict', result.status === 'REVIEW' && 'review-verdict')}>
        <div className="flex items-center gap-2">{result.status === 'NORMAL' ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}<strong>{result.status}</strong></div><p>{sample ? 'Manually annotated example' : result.status === 'NORMAL' ? 'No valid anomaly regions found' : result.status === 'REVIEW' ? 'Anomaly evidence needs human review' : 'Localized defect evidence found'}</p>
      </div><dl className="result-metrics"><div><dt>{sample ? 'Annotated regions' : 'Localized regions'}</dt><dd>{regions.length.toString().padStart(2, '0')}</dd></div><div><dt>Anomaly score <Info size={12} /></dt><dd>{result.score === null ? '—' : result.score.toFixed(3)}</dd></div><div><dt>Pixel threshold</dt><dd>{result.threshold === null ? '—' : result.threshold.toFixed(3)}</dd></div><div><dt>Processing time</dt><dd>{result.elapsed_ms === null ? '—' : `${Math.round(result.elapsed_ms)} ms`}</dd></div></dl>
      <div className="regions-title"><span>REGION DETAILS</span><span>{regions.length}</span></div><div className="region-list">{regions.length ? regions.map((region, index) => <button key={region.id} className={cn('region-detail', selected === region.id && 'region-detail-selected')} onClick={() => { setSelected(region.id); setTab('boxes') }}><div className="region-title"><span className={cn('region-number', index % 2 === 1 && 'secondary-number')}>{String(region.id).padStart(2, '0')}</span><strong>{region.label}</strong><Crosshair size={14} /></div><div className="region-coordinates"><span>x <b>{region.x}</b></span><span>y <b>{region.y}</b></span><span>w <b>{region.width}</b></span><span>h <b>{region.height}</b></span></div><div className="region-score">{region.score === null ? 'Manual annotation · no confidence score' : `Region score ${region.score.toFixed(3)} · not a probability`}</div></button>) : <p className="muted-copy">No boxes. Image-level predictions never create regions.</p>}</div>
      {result.quantum && <div className="quantum-result"><strong>Experimental image classification</strong><span>Qiskit kernel <b>{result.quantum.prediction}</b></span><span>Classical SVM <b>{result.quantum.classical_prediction}</b></span><small>Quantum margin: {result.quantum.margin.toFixed(3)} · not a probability. Neither classifier generates boxes.</small></div>}<div className="result-note"><Info size={14} /><p>{result.warning}</p></div><Button variant="outline" className="w-full mt-auto" onClick={() => downloadJSON(result, sample ? 'quantuminspect-illustrative-sample.json' : 'quantuminspect-result.json')}><Download data-icon="inline-start" />Export result</Button></> : <div className="waiting-results"><Layers size={30} strokeWidth={1.3} /><h4>{busy ? 'Looking closer…' : 'Ready when you are'}</h4><p>{busy ? 'Computing spatial anomaly evidence.' : 'Results and region coordinates will appear after an inspection.'}</p></div>}
    </aside></div>
  </section>
}
