import { BACKEND_URL } from '@/lib/backend'

export const dynamic = 'force-dynamic'
export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, { cache: 'no-store', signal: AbortSignal.timeout(3000) })
    if (!response.ok) throw new Error('Service unavailable')
    return Response.json(await response.json(), { headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return Response.json({ connected: false, models: [], quantum_models: [], device: 'unavailable', message: 'Start the local Python service described in README.md. No model results are available.' }, { headers: { 'Cache-Control': 'no-store' } })
  }
}
