import { BACKEND_URL } from '@/lib/backend'
import { categories, type Category } from '@/lib/inspection'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const category = new URL(request.url).searchParams.get('category') ?? 'screw'
  if (!categories.includes(category as Category)) {
    return Response.json({ detail: 'Select a valid product category.' }, { status: 422 })
  }
  try {
    const response = await fetch(`${BACKEND_URL}/research?category=${encodeURIComponent(category)}`, {
      cache: 'no-store', signal: AbortSignal.timeout(10000),
    })
    if (!response.ok) throw new Error('Research service unavailable')
    return Response.json(await response.json(), { headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return Response.json({ detail: 'Research reports are unavailable. Connect the local Python service to view your dataset manifests, training artifacts, and measured evaluations.' }, { status: 503, headers: { 'Cache-Control': 'no-store' } })
  }
}
