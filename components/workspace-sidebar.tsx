'use client'

import { Atom, ScanLine, Database, Boxes, ChartNoAxesCombined, BookOpen, ArrowUpRight, FlaskConical, ChevronDown, PanelLeftClose } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Section } from '@/lib/inspection'

const links = [
  { id: 'inspection' as const, label: 'Inspection', icon: ScanLine },
  { id: 'dataset' as const, label: 'Dataset explorer', icon: Database },
  { id: 'models' as const, label: 'Model laboratory', icon: Boxes },
  { id: 'evaluation' as const, label: 'Evaluation', icon: ChartNoAxesCombined },
]
export function WorkspaceSidebar({ section, onNavigate, onDocs, mobileOpen, onClose }: { section: Section; onNavigate: (section: Section) => void; onDocs: () => void; mobileOpen: boolean; onClose: () => void }) {
  return <aside className={cn('workspace-sidebar', mobileOpen && 'mobile-open')}>
    <a href="/" className="brand"><span className="brand-mark"><Atom size={23} strokeWidth={1.7} /></span><span>Quantum<span className="font-normal">Inspect</span><span className="brand-dot">.</span></span></a>
    <button className="workspace-picker" onClick={onDocs}><span className="workspace-avatar">Q</span><span className="flex flex-1 flex-col text-left"><strong>Research lab</strong><small>Local development</small></span><ChevronDown size={14} /></button>
    <div className="nav-label">WORKSPACE</div>
    <nav aria-label="Main navigation" className="main-nav">{links.map(({ id, label, icon: Icon }) => <button key={id} className={cn('nav-item', section === id && 'active')} aria-current={section === id ? 'page' : undefined} onClick={() => { onNavigate(id); onClose() }}><Icon size={18} strokeWidth={1.7} /><span>{label}</span>{id === 'models' && <span className="nav-count">2</span>}</button>)}</nav>
    <div className="sidebar-bottom"><div className="research-note"><FlaskConical size={19} /><strong>Built for evidence.</strong><p>Classical precision.<br />Quantum exploration.</p><button onClick={() => { onNavigate('models'); onClose() }}>Explore the pipeline <ArrowUpRight size={13} /></button></div>
    <button className="nav-item" onClick={onDocs}><BookOpen size={18} strokeWidth={1.7} />Documentation<ArrowUpRight className="ml-auto" size={14} /></button>
    <div className="sidebar-footer"><span className="status-dot" /><span>Research edition</span><span className="ml-auto font-mono">v0.1</span></div></div>
    <button className="mobile-close" aria-label="Close navigation" onClick={onClose}><PanelLeftClose size={20} /></button>
  </aside>
}
