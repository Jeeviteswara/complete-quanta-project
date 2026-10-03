import { BACKEND_URL, MAX_IMAGE_BYTES } from '@/lib/backend'
import { categories } from '@/lib/inspection'

export const runtime = 'nodejs'
export const maxDuration = 120

export async function POST(request: Request) {
  const origin = request.headers.get('origin')
  if (origin && origin !== new URL(request.url).origin) return Response.json({ detail: 'Cross-origin inspection requests are not allowed.' }, { status: 403 })
  const contentType = request.headers.get('content-type') ?? ''
  if (!contentType.startsWith('multipart/form-data')) return Response.json({ detail: 'Expected an image upload.' }, { status: 415 })
  if (Number(request.headers.get('content-length') ?? 0) > MAX_IMAGE_BYTES + 65536) return Response.json({ detail: 'Image exceeds the 10 MB limit.' }, { status: 413 })
  try {
    const reader = request.body?.getReader()
    if (!reader) return Response.json({ detail: 'No image received.' }, { status: 400 })
    const chunks: Uint8Array[] = []
    let size = 0
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > MAX_IMAGE_BYTES + 65536) { await reader.cancel(); return Response.json({ detail: 'Image exceeds the 10 MB limit.' }, { status: 413 }) }
      chunks.push(value)
    }
    const body = Buffer.concat(chunks)
    let parsed: FormData
    try {
      parsed = await new Response(body, { headers: { 'Content-Type': contentType } }).formData()
    } catch {
      return Response.json({ detail: 'The upload form is malformed. Choose your image again.' }, { status: 400 })
    }
    if (['image', 'category', 'quantum'].some(key => parsed.getAll(key).length > 1)) return Response.json({ detail: 'Upload fields must not be repeated.' }, { status: 422 })
    const quantum = parsed.get('quantum')
    if (quantum !== null && quantum !== 'true' && quantum !== 'false') return Response.json({ detail: 'Quantum comparison must be true or false.' }, { status: 422 })
    const image = parsed.get('image')
    if (!(image instanceof File) || image.size > MAX_IMAGE_BYTES || !['image/png', 'image/jpeg', 'image/webp'].includes(image.type)) return Response.json({ detail: 'Choose a PNG, JPEG, or WebP image up to 10 MB.' }, { status: 422 })
    if (!categories.includes(parsed.get('category') as typeof categories[number])) return Response.json({ detail: 'Select a valid product category.' }, { status: 422 })
    const response = await fetch(`${BACKEND_URL}/inspect`, { method: 'POST', body: parsed, signal: AbortSignal.any([request.signal, AbortSignal.timeout(110000)]) })
    return Response.json(await response.json(), { status: response.status, headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return Response.json({ detail: 'The local inference service is unavailable or timed out. Start the FastAPI service and train a category model using the included guide. No prediction or boxes were generated.' }, { status: 503 })
  }
}
