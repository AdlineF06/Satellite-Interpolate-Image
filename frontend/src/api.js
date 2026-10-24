const j = async (r) => {
  const body = await r.text()
  let data
  try { data = body ? JSON.parse(body) : {} }
  catch { throw new Error(`API ${r.status}: ${body.slice(0, 240) || r.statusText}. Check that the backend is running the latest code.`) }
  if (!r.ok) throw new Error(data.detail || `API ${r.status}`)
  return data
}
export const getRegions = () => fetch('/api/regions').then(j)
export const getFrames = (id) => fetch('/api/frames/' + id).then(j)
export const interpolate = (region, start) =>
  fetch('/api/interpolate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ region, start }) }).then(j)
export const interpolateAtTimes = (region, before, target, after, lat, lng) =>
  fetch('/api/interpolate-times', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ region, before, target, after, lat, lng }) }).then(j)
