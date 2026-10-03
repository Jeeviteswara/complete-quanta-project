'use client'

import { useState } from 'react'
import useSWR from 'swr'
import { Check, Copy, Info, RefreshCw, Terminal } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { categoryLabel, researchFetcher, type ResearchReport } from '@/lib/research'
import type { Category, ServiceStatus } from '@/lib/inspection'

function CommandBlock({ label, command }: { label: string; command: string }) {
  const [feedback, setFeedback] = useState('')
  async function copy() {
    try {
      await navigator.clipboard.writeText(command)
      setFeedback('Copied to clipboard.')
    } catch {
      setFeedback('Clipboard unavailable. Select and copy the command below.')
    }
  }
  return <div className="flex min-w-0 flex-col gap-2 rounded-lg border p-3">
    <div className="flex items-center justify-between gap-2"><strong className="text-xs">{label}</strong><Button variant="ghost" size="sm" aria-label={`Copy ${label.toLowerCase()} commands`} onClick={() => void copy()}>{feedback === 'Copied to clipboard.' ? <Check data-icon="inline-start" /> : <Copy data-icon="inline-start" />}Copy</Button></div>
    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs leading-6" tabIndex={0} aria-label={label}><code>{command}</code></pre>
    {feedback && <p className="text-xs text-muted-foreground" role="status">{feedback}</p>}
  </div>
}

export function SetupGuide({ category, status, onRefresh }: { category: Category; status?: ServiceStatus; onRefresh: () => void }) {
  const { data, error, isLoading, isValidating, mutate } = useSWR<ResearchReport>(`/api/research?category=${category}`, researchFetcher, { refreshInterval: 30000, shouldRetryOnError: false })
  const report = error ? undefined : data
  const entry = report?.categories.find(item => item.category === category)
  const checkpoints = [
    { label: 'Python service', value: status?.connected ? `Connected · ${status.device}` : 'Not connected', ready: Boolean(status?.connected) },
    { label: `${categoryLabel(category)} dataset`, value: !report ? 'Not checked' : report.dataset ? `${report.dataset.total} images recorded` : 'Manifest needed', ready: Boolean(report?.dataset) },
    { label: 'Spatial model', value: !status?.connected ? 'Not checked' : status.models.includes(category) ? 'Artifact available' : 'Training needed', ready: Boolean(status?.connected && status.models.includes(category)) },
    { label: 'Held-out evaluation', value: !report ? 'Not checked' : report.evaluation ? report.evaluation_stale ? 'Historical · rerun needed' : 'Report available' : 'Evaluation needed', ready: Boolean(report?.evaluation && !report.evaluation_stale) },
  ]
  return <section className="flex min-w-0 flex-col gap-4" aria-label="Project setup checklist">
    <div className="flex flex-wrap items-center justify-between gap-3"><h3 className="font-semibold">Your next steps · {categoryLabel(category)}</h3><Button variant="outline" size="sm" disabled={isValidating} onClick={() => { onRefresh(); void mutate() }}><RefreshCw data-icon="inline-start" className={isValidating ? 'animate-spin' : undefined} />Refresh setup</Button></div>
    <ul className="flex flex-col divide-y rounded-lg border px-3" aria-busy={isLoading}>{checkpoints.map(item => <li key={item.label} className="flex flex-wrap items-center justify-between gap-2 py-3"><span className="text-xs">{item.label}</span><Badge variant={item.ready ? 'secondary' : 'outline'}>{item.ready && <Check data-icon="inline-start" />}{item.value}</Badge></li>)}</ul>
    <p className="text-xs leading-5 text-muted-foreground">Quantum comparison is optional: {entry?.quantum_available ? 'artifact present; compatibility is checked during inference.' : report ? 'independent labeled training data is still needed.' : 'availability has not been checked.'} Artifact presence does not establish accuracy.</p>
    {error && <p role="status" className="text-xs text-muted-foreground">Research reports could not be reached. Dataset and evaluation status remain unknown.</p>}
    {report?.warnings.map(warning => <p key={warning} role="status" className="text-xs text-destructive">{warning}</p>)}
    <Alert><Info /><AlertTitle>A local workstation is required</AlertTitle><AlertDescription>The hosted preview cannot access a Python service on your computer. Use the connected GitHub repository or the v0 installation command to run both services locally. Publishing the interface does not train or host the models.</AlertDescription></Alert>
    <details className="rounded-lg border p-3">
      <summary className="cursor-pointer font-medium"><span className="inline-flex items-center gap-2"><Terminal className="size-4" />Workstation setup commands</span></summary>
      <div className="mt-4 flex min-w-0 flex-col gap-4">
        <p className="text-xs leading-5 text-muted-foreground">For your own workstation only—not the v0 preview. Requires Python 3.11 or 3.12, Node.js 22+, and pnpm 12.3.4. These commands install CPU PyTorch. For your RTX 4060, use the <a className="underline underline-offset-2" href="https://pytorch.org/get-started/locally/" target="_blank" rel="noreferrer">official CUDA installation selector</a> instead of the CPU install line.</p>
        <Tabs defaultValue="windows" className="min-w-0"><TabsList aria-label="Workstation operating system"><TabsTrigger value="windows">Windows</TabsTrigger><TabsTrigger value="unix">Linux</TabsTrigger></TabsList>{(['windows', 'unix'] as const).map(platform => {
          const python = platform === 'windows' ? '.\\.venv\\Scripts\\python.exe' : '.venv/bin/python'
          return <TabsContent value={platform} key={platform} className="flex min-w-0 flex-col gap-3">
            <CommandBlock label="1. Install dependencies" command={`${platform === 'windows' ? 'py -3.12' : 'python3.12'} -m venv .venv\n${python} -m pip install -r backend/requirements.txt\n${python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\npnpm install --frozen-lockfile`} />
            <div className="text-xs leading-5 text-muted-foreground">Download <a className="underline underline-offset-2" href="https://www.mvtec.com/company/research/datasets/mvtec-ad" target="_blank" rel="noreferrer">MVTec AD</a> under its license and place the category folder in <code>data/mvtec_ad/{category}</code>. Keep official test images untouched.</div>
            <CommandBlock label="2. Train and evaluate" command={`${python} -m backend.pipeline --root data/mvtec_ad --category ${category}`} />
            <p className="text-xs leading-5 text-muted-foreground">This runs the existing inspection, training, and evaluation stages. Existing outputs are protected. Add <code>--labeled-manifest data/labeled-{category}.json</code> only when independent labeled train/validation data is ready; never use official test defects.</p>
            <CommandBlock label="3. Start Python service" command={`${python} -m uvicorn backend.api:app --host 127.0.0.1 --port 8000`} />
            <CommandBlock label="4. Start frontend in another terminal" command="pnpm dev" />
            <p className="text-xs leading-5 text-muted-foreground">Open <code>http://localhost:3000</code> on that same workstation, then refresh setup. See README.md for individual stages, recovery, and evaluation limitations.</p>
          </TabsContent>
        })}</Tabs>
      </div>
    </details>
  </section>
}
