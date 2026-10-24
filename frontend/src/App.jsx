import { useEffect, useRef, useState } from 'react'
import { getRegions, getFrames, interpolate, interpolateAtTimes } from './api'
import './styles.css'

const STAGES = ['Preprocessing', 'CNN features', 'Optical flow', 'Warp and refine', 'Evaluation']
const METHODS = [['averaging', 'Frame averaging'], ['flow_warp', 'Flow warp'], ['warp_refine', 'Warp-and-Refine']]
const TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN
const asLocalInput = (date) => new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
function defaultTimes() {
  const after = new Date()
  after.setMinutes(Math.floor(after.getMinutes() / 10) * 10, 0, 0)
  after.setHours(after.getHours() - 3)
  return { before: asLocalInput(new Date(after.getTime() - 20 * 60000)), target: asLocalInput(new Date(after.getTime() - 10 * 60000)), after: asLocalInput(after) }
}
const displayTime = (value) => Number.isNaN(Date.parse(value)) ? value : new Date(value).toLocaleString()

function FlowView({ src, arrows, size }) {
  const c = useRef()
  useEffect(() => {
    const img = new Image(); img.src = src
    img.onload = () => {
      const ctx = c.current?.getContext('2d'); if (!ctx) return
      const [w, h] = size
      ctx.drawImage(img, 0, 0, w, h); ctx.strokeStyle = '#5ef2ff'; ctx.fillStyle = '#5ef2ff'; ctx.lineWidth = 1.5
      arrows.forEach(({ x, y, dx, dy }) => {
        const k = 4, ex = x + dx * k, ey = y + dy * k, a = Math.atan2(dy, dx)
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(ex, ey)
        ctx.lineTo(ex - 5 * Math.cos(a - 0.5), ey - 5 * Math.sin(a - 0.5))
        ctx.lineTo(ex - 5 * Math.cos(a + 0.5), ey - 5 * Math.sin(a + 0.5)); ctx.fill()
      })
    }
  }, [src, arrows, size])
  return <canvas ref={c} width={size[0]} height={size[1]} className="sat" />
}

function Compare({ left, right }) {
  const [p, setP] = useState(50)
  return <div className="cmp"><img src={right} className="sat" alt="Ground truth" />
    <img src={left} className="sat top" style={{ clipPath: `inset(0 ${100 - p}% 0 0)` }} alt="Generated" />
    <div className="bar" style={{ left: p + '%' }} />
    <input type="range" min="0" max="100" value={p} onChange={(e) => setP(+e.target.value)} aria-label="Comparison slider" />
    <span className="tag l">Generated</span><span className="tag r">Real frame</span></div>
}

function MetricDashboard({ result }) {
  const methods = [
    ['Frame average', result.metrics.averaging],
    ['Flow warp', result.metrics.flow_warp],
    ['Warp + CNN refine', result.metrics.warp_refine],
  ]
  const maxPsnr = Math.max(1, ...methods.map(([, metric]) => metric.psnr))
  const maxMae = Math.max(0.001, ...methods.map(([, metric]) => metric.mae))
  return <section className="metric-dashboard" aria-label="Interpolation metrics">
    <div className="metric-heading"><div><span className="eyebrow">MODEL EVALUATION</span><h3>How close is the generated frame?</h3></div></div>
    <div className="metric-legend"><span>PSNR ↑ higher is better</span><span>SSIM ↑ closer to 1 is better</span><span>MAE ↓ lower is better</span></div>
    <div className="metric-chart" role="img" aria-label="Chart comparing PSNR, SSIM and MAE across interpolation methods">
      {methods.map(([name, metric]) => <div className="metric-row" key={name}>
        <b>{name}</b>
        <div className="metric-bars">
          <div className="metric-bar-line"><span>PSNR</span><div className="metric-track"><i className="psnr-bar" style={{ width: `${Math.max(2, metric.psnr / maxPsnr * 100)}%` }} /></div><strong>{metric.psnr.toFixed(2)} dB</strong></div>
          <div className="metric-bar-line"><span>SSIM</span><div className="metric-track"><i className="ssim-bar" style={{ width: `${Math.max(2, metric.ssim * 100)}%` }} /></div><strong>{metric.ssim.toFixed(3)}</strong></div>
          <div className="metric-bar-line"><span>MAE</span><div className="metric-track"><i className="mae-bar" style={{ width: `${Math.max(2, metric.mae / maxMae * 100)}%` }} /></div><strong>{metric.mae.toFixed(4)}</strong></div>
        </div>
      </div>)}
    </div>
    <p className="metric-explainer">These metrics compare each estimate with the NASA target-time frame. PSNR and SSIM summarize image similarity; MAE is average pixel error. They are image-quality metrics, not classification accuracy.</p>
    <div className="metric-facts"><div><span>Mean optical flow</span><b>{result.mean_motion_px.toFixed(2)} px</b></div><div><span>Interpolation time</span><b>{result.timing_ms.total} ms</b></div><div><span>Frame size</span><b>{result.size[0]} × {result.size[1]}</b></div><div><span>Target time</span><b>{displayTime(result.times.mid)}</b></div></div>
  </section>
}

function DiagnosticsPanel({ result }) {
  const diagnostic = result.diagnostics
  if (!diagnostic) return <section className="diagnostic-panel"><h2>Validation plots unavailable</h2><p>Restart the backend to receive pixel residual data for this result.</p></section>
  const totalPixels = diagnostic.pixel_error_counts.reduce((sum, count) => sum + count, diagnostic.histogram_outliers || 0) || 1
  const relativeFrequencies = diagnostic.pixel_error_counts.map((count) => count / totalPixels)
  const maxFrequency = Math.max(1e-8, ...relativeFrequencies)
  const zoomHistogram = maxFrequency < 0.02
  // Square-root scaling keeps rare-error bins visible beside the large near-zero bin.
  const visibleFrequencies = relativeFrequencies.map((frequency) => Math.sqrt(frequency))
  const maxVisibleFrequency = Math.max(1e-8, ...visibleFrequencies)
  const errorValues = diagnostic.loss_curve.map((loss) => Math.max(0, loss))
  const maxLoss = Math.max(1e-6, ...errorValues)
  const minLoss = Math.min(...errorValues)
  const points = (values, min, max) => values.map((value, index) => {
    const x = values.length < 2 ? 50 : index / (values.length - 1) * 100
    const y = 92 - ((value - min) / Math.max(max - min, 1e-9)) * 82
    return `${x},${y}`
  }).join(' ')
  const lossPoints = points(errorValues, Math.max(0, minLoss - (maxLoss - minLoss) * 0.12), maxLoss + Math.max((maxLoss - minLoss) * 0.12, maxLoss * 0.04))
  const similarityPoints = points(diagnostic.similarity_curve, 0, 1)
  return <div className="diagnostics-stack">
    <section className="diagnostic-panel">
      <div className="diagnostic-title"><div><span className="eyebrow">PIXEL VALIDATION</span><h2>Loss and similarity curves</h2></div><span className="chart-badge">validation</span></div>
      <p className="diagnostic-copy">Pixel MAE loss and fraction of pixels within each error tolerance, compared with the NASA target frame. Tolerances adapt to the observed pixel-error range.</p>
      <div className="line-chart-wrap"><svg className="line-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Validation loss and pixel similarity across error tolerance thresholds">
        {[10, 30, 50, 70, 90].map((y) => <line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}
        <polyline points={lossPoints} className="loss-line" /><polyline points={similarityPoints} className="similarity-line" />
      </svg></div>
      <div className="curve-legend"><span><i className="legend-loss" />Residual loss (zoomed)</span><span><i className="legend-similarity" />Pixels within tolerance</span></div>
      <div className="axis-labels"><span>0 tolerance</span><span>{diagnostic.histogram_max ?? '0.5'} tolerance</span></div>
      <p className="diagnostic-note">{diagnostic.curve_note}</p>
    </section>
    <section className="diagnostic-panel">
      <div className="diagnostic-title"><div><span className="eyebrow">GENERATED − REAL</span><h2>Residual distribution</h2></div><span className="chart-badge">pixel count</span></div>
      <p className="diagnostic-copy">Frequency of absolute grayscale pixel differences. Range is zoomed to the 99.5th percentile and bar heights use square-root scaling so smaller bins remain visible.</p>
      <div className="histogram" role="img" aria-label="Histogram of absolute pixel differences between generated and real image">
        {diagnostic.pixel_error_counts.map((count, index) => <div className="hist-bin" key={index} title={`${diagnostic.pixel_error_bins[index].toFixed(4)} difference: ${count} pixels (${(relativeFrequencies[index] * 100).toFixed(2)}%)`}><i style={{ height: `${count ? Math.max(4, visibleFrequencies[index] / maxVisibleFrequency * 100) : 0}%` }} /><span>{index % 4 === 0 ? diagnostic.pixel_error_bins[index].toFixed(3) : ''}</span></div>)}
      </div>
      <div className="axis-labels"><span>0 · identical</span><span>Absolute pixel difference {zoomHistogram ? '(zoomed frequency)' : '(relative frequency)'}</span><span>1 · largest</span></div>
      <div className="residual-summary"><span>Mean absolute error<b>{result.metrics.warp_refine.mae.toFixed(4)}</b></span><span>Within 0.10<b>{(diagnostic.similarity_curve[3] * 100).toFixed(1)}%</b></span><span>Pixels evaluated<b>{result.size[0] * result.size[1]}</b></span></div>
    </section>
  </div>
}

function ImageModal({ result, onClose }) {
  useEffect(() => {
    const closeOnEscape = (event) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', closeOnEscape)
    document.body.style.overflow = 'hidden'
    return () => { window.removeEventListener('keydown', closeOnEscape); document.body.style.overflow = '' }
  }, [onClose])
  return <div className="image-modal" role="dialog" aria-modal="true" aria-label="Large generated and real frame comparison" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="modal-panel"><div className="modal-header"><div><span className="eyebrow">FRAME COMPARISON</span><h2>{result.location ? `${result.location.lat}°, ${result.location.lng}°` : 'Generated and real frame'}</h2></div><button className="modal-close" onClick={onClose} aria-label="Close image popup">×</button></div>
      <p className="modal-help">Drag the slider to reveal the generated frame and NASA real frame.</p><Compare left={result.generated} right={result.ground_truth} />
    </div>
  </div>
}

function Player({ result, smooth }) {
  const seq = smooth ? [result.frame_a, result.generated, result.frame_b] : [result.frame_a, result.frame_b]
  const [i, setI] = useState(0)
  useEffect(() => { setI(0); const t = setInterval(() => setI((v) => (v + 1) % seq.length), smooth ? 450 : 900); return () => clearInterval(t) }, [result, smooth])
  return <img src={seq[i % seq.length]} className="sat" alt="Interpolated frame playback" />
}

function CesiumGlobe({ regions, selected, onSelect, rotating, pointPicking, onExactPointSelected, onAvailabilityChange }) {
  const host = useRef(null)
  const viewer = useRef(null)
  const token = useRef(TOKEN)
  const selectCallback = useRef(onSelect)
  const selection = useRef(selected)
  const rotationState = useRef(rotating)
  const availabilityCallback = useRef(onAvailabilityChange)
  const pointPickingState = useRef(pointPicking)
  const exactPointCallback = useRef(onExactPointSelected)
  const [viewerReady, setViewerReady] = useState(false)
  const [viewerError, setViewerError] = useState(false)
  const [earthStyle, setEarthStyle] = useState('aerial')

  useEffect(() => { selectCallback.current = onSelect }, [onSelect])
  useEffect(() => { selection.current = selected }, [selected])
  useEffect(() => { rotationState.current = rotating }, [rotating])
  useEffect(() => { availabilityCallback.current = onAvailabilityChange }, [onAvailabilityChange])
  useEffect(() => { pointPickingState.current = pointPicking }, [pointPicking])
  useEffect(() => { exactPointCallback.current = onExactPointSelected }, [onExactPointSelected])

  useEffect(() => {
    let alive = true
    let cleanup = () => {}
    if (!TOKEN) return () => {}

    import('cesium').then(async (Cesium) => {
      if (!alive || !host.current) return
      window.CESIUM_BASE_URL = '/cesium/'
      window.Cesium = Cesium
      Cesium.Ion.defaultAccessToken = token.current
      const imagery = await Cesium.IonImageryProvider.fromAssetId(2)
      const isMobile = window.matchMedia('(max-width: 600px)').matches
      const v = new Cesium.Viewer(host.current, {
        animation: false, baseLayerPicker: false, fullscreenButton: false,
        geocoder: false, homeButton: false, infoBox: false, sceneModePicker: false,
        selectionIndicator: false, timeline: false, navigationHelpButton: false,
        resolutionScale: isMobile ? 0.65 : 1,
        baseLayer: new Cesium.ImageryLayer(imagery),
      })
      if (!alive) { v.destroy(); return }
      viewer.current = v
      setViewerReady(true)
      setViewerError(false)
      availabilityCallback.current?.(true)
      v.scene.globe.enableLighting = true
      v.scene.globe.showGroundAtmosphere = true
      if (isMobile) v.scene.globe.maximumScreenSpaceError = 4
      v.scene.backgroundColor = Cesium.Color.fromCssColorString('#050a1c')
      v.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(80, 20, 19000000) })
      const pins = new Map()
      const drawPins = () => {
        for (const [id, entity] of pins) { v.entities.remove(entity); pins.delete(id) }
        for (const area of regions) {
          const active = selection.current?.id === area.id
          const entity = v.entities.add({
            id: area.id,
            position: Cesium.Cartesian3.fromDegrees(area.lng, area.lat),
            point: { pixelSize: active ? 12 : 8, color: Cesium.Color.fromCssColorString(active ? '#ffffff' : '#5ef2ff'), outlineColor: Cesium.Color.fromCssColorString('#062140'), outlineWidth: 2, disableDepthTestDistance: 10000000 },
            label: { text: area.name, font: '12px sans-serif', fillColor: Cesium.Color.fromCssColorString('#dcecff'), outlineColor: Cesium.Color.fromCssColorString('#050a1c'), outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, -16), scaleByDistance: new Cesium.NearFarScalar(500000, 1, 18000000, 0.55), disableDepthTestDistance: 10000000 },
          })
          pins.set(area.id, entity)
        }
      }
      drawPins()
      const handler = new Cesium.ScreenSpaceEventHandler(v.scene.canvas)
      handler.setInputAction((movement) => {
        if (pointPickingState.current && selection.current) {
          const cartesian = v.camera.pickEllipsoid(movement.position, v.scene.globe.ellipsoid)
          if (!cartesian) return
          const point = Cesium.Cartographic.fromCartesian(cartesian)
          const lat = Cesium.Math.toDegrees(point.latitude)
          const lng = Cesium.Math.toDegrees(point.longitude)
          v.entities.removeById('exact-selected-point')
          v.entities.add({
            id: 'exact-selected-point', position: cartesian,
            point: { pixelSize: 13, color: Cesium.Color.WHITE, outlineColor: Cesium.Color.fromCssColorString('#05d9f5'), outlineWidth: 3, disableDepthTestDistance: 10000000 },
            label: { text: `${lat.toFixed(3)}°, ${lng.toFixed(3)}°`, font: '13px sans-serif', fillColor: Cesium.Color.WHITE, outlineColor: Cesium.Color.fromCssColorString('#050a1c'), outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, -20), disableDepthTestDistance: 10000000 },
          })
          exactPointCallback.current?.({ lat: Number(lat.toFixed(5)), lng: Number(lng.toFixed(5)) })
          return
        }
        const hit = v.scene.pick(movement.position)
        const area = hit?.id && regions.find((item) => item.id === hit.id.id)
        if (area) selectCallback.current(area)
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK)
      let lastTick
      const orbit = (_scene, time) => {
        if (!rotationState.current || selection.current) { lastTick = time; return }
        if (lastTick) {
          const seconds = Cesium.JulianDate.secondsDifference(time, lastTick)
          if (seconds > 0 && seconds < 1) v.camera.rotate(Cesium.Cartesian3.UNIT_Z, -seconds * 0.035)
        }
        lastTick = time
      }
      v.scene.preRender.addEventListener(orbit)
      cleanup = () => { v.scene.preRender.removeEventListener(orbit); handler.destroy(); v.destroy(); viewer.current = null; setViewerReady(false); availabilityCallback.current?.(false) }
    }).catch((e) => {
      console.error('Cesium viewer failed to load', e)
      if (alive) {
        host.current?.replaceChildren()
        setViewerError(true)
        availabilityCallback.current?.(false)
      }
    })

    return () => { alive = false; cleanup() }
  }, [regions])

  useEffect(() => {
    const v = viewer.current
    if (!v) return
    v.clock.shouldAnimate = rotating && !selected
    if (selected) v.camera.flyTo({ destination: window.Cesium?.Cartesian3?.fromDegrees(selected.lng, selected.lat, 2200000), duration: 2 })
    else v.camera.flyTo({ destination: window.Cesium?.Cartesian3?.fromDegrees(80, 20, 19000000), duration: 1.8 })
  }, [selected, rotating, viewerReady])

  useEffect(() => {
    const v = viewer.current
    if (!v || !viewerReady) return
    const styles = { aerial: 2, sentinel: 3954, street: 3 }
    let cancelled = false
    ;(async () => {
      try {
        const provider = await window.Cesium.IonImageryProvider.fromAssetId(styles[earthStyle])
        if (cancelled || v.isDestroyed()) return
        provider.errorEvent.addEventListener((error) => console.error(`Could not load ${earthStyle} imagery tile`, error))
        v.imageryLayers.removeAll()
        v.imageryLayers.addImageryProvider(provider)
      } catch (e) {
        console.error(`Could not load ${earthStyle} ion imagery`, e)
      }
    })()
    return () => { cancelled = true }
  }, [earthStyle, viewerReady])

  return <>
    {!TOKEN ? <div className="cesium-missing"><div className="fallback-earth" aria-hidden="true"/><h2>Cesium ion token needed</h2><p>Add <code>VITE_CESIUM_ION_TOKEN</code> to <code>frontend/.env.local</code>, then restart Vite.</p></div>
      : viewerError ? <div className="cesium-failure"><div className="fallback-earth" aria-hidden="true"/><h2>3D globe could not start</h2><p>Your browser reported a WebGL startup error. The area search and image interpolation controls still work.</p><p>In Chrome, enable <b>Settings → System → Use graphics acceleration when available</b>, relaunch the browser, then refresh this app.</p></div>
        : <div className="cesium-host" ref={host} />}
    <label className="earth-style">Earth appearance
      <select value={earthStyle} onChange={(e) => setEarthStyle(e.target.value)} disabled={!TOKEN || viewerError}>
        <option value="aerial">Aerial imagery</option><option value="sentinel">Sentinel satellite</option><option value="street">Street map</option>
      </select>
    </label>
  </>
}

export default function App() {
  const [regions, setRegions] = useState([])
  const [sel, setSel] = useState(null)
  const [frames, setFrames] = useState([])
  const [start, setStart] = useState(0)
  const [res, setRes] = useState(null)
  const [stage, setStage] = useState(-1)
  const [tab, setTab] = useState('compare')
  const [smooth, setSmooth] = useState(true)
  const [source, setSource] = useState('nasa')
  const [requestedTimes, setRequestedTimes] = useState(defaultTimes)
  const [search, setSearch] = useState('')
  const [rotating, setRotating] = useState(true)
  const [pointPicking, setPointPicking] = useState(false)
  const [customPoint, setCustomPoint] = useState(false)
  const [globeAvailable, setGlobeAvailable] = useState(false)
  const [err, setErr] = useState('')
  const [imageModal, setImageModal] = useState(false)
  const [dashboardOpen, setDashboardOpen] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const edgeSwipeStartX = useRef(0)
  const sidebarSwipeStartX = useRef(0)

  useEffect(() => { getRegions().then(setRegions).catch(() => setErr('Backend not reachable. Start it on port 8001.')) }, [])
  useEffect(() => {
    if (!sel || source !== 'demo') { setFrames([]); return }
    getFrames(sel.id).then(setFrames).catch((e) => setErr(e.message))
  }, [sel, source])

  const pick = (r) => {
    setSel(r); setRes(null); setStart(0); setStage(-1); setFrames([]); setErr(''); setPointPicking(false); setCustomPoint(false)
  }
  const back = () => { setSel(null); setRes(null); setFrames([]); setErr(''); setPointPicking(false); setCustomPoint(false) }
  const chooseExactPoint = ({ lat, lng }) => {
    setSel((current) => current ? { ...current, lat, lng } : current)
    setCustomPoint(true); setPointPicking(false); setRes(null); setErr('')
  }
  const closeImageModal = () => setImageModal(false)
  const editCoordinate = (key, value) => {
    setSel((current) => current ? { ...current, [key]: value === '' ? '' : Number(value) } : current)
    setCustomPoint(true); setRes(null)
  }

  const run = async () => {
    setRes(null); setErr(''); let s = 0; setStage(0)
    const tm = setInterval(() => { s = Math.min(s + 1, 4); setStage(s) }, 450)
    try {
      const result = source === 'nasa'
        ? await interpolateAtTimes(sel.id, ...['before', 'target', 'after'].map((key) => new Date(requestedTimes[key]).toISOString()), sel.lat, sel.lng)
        : await interpolate(sel.id, start)
      setRes(result); setStage(5)
    }
    catch (e) { setErr(e.message); setStage(-1) } finally { clearInterval(tm) }
  }
  const shownRegions = regions.filter((r) => `${r.name} ${r.event} ${r.satellite}`.toLowerCase().includes(search.toLowerCase()))
  const selectedMillis = ['before', 'target', 'after'].map((key) => new Date(requestedTimes[key]).getTime())
  const validRequestedTimes = selectedMillis.every(Number.isFinite) && selectedMillis[0] < selectedMillis[1] && selectedMillis[1] < selectedMillis[2] && selectedMillis[2] - selectedMillis[0] <= 120 * 60000
  const validPoint = Number.isFinite(sel?.lat) && Number.isFinite(sel?.lng) && sel.lat >= -90 && sel.lat <= 90 && sel.lng >= -180 && sel.lng <= 180

  return <div className={'app' + (dashboardOpen ? ' dashboard-open' : '') + (mobileSidebarOpen ? ' mobile-sidebar-open' : '')}>
    <div className="globe"><CesiumGlobe regions={regions} selected={sel} onSelect={pick} rotating={rotating} pointPicking={pointPicking} onExactPointSelected={chooseExactPoint} onAvailabilityChange={setGlobeAvailable} /></div>
    <div className="frame">
      <header><b className="logo">◉ Orbit Interpolator</b><div className="header-actions">
        <button className="toggle mobile-view-toggle" onClick={() => setMobileSidebarOpen((open) => !open)}>{mobileSidebarOpen ? 'Close areas' : 'Areas'}</button>
        <button className="toggle" onClick={() => setRotating((v) => !v)} disabled={!globeAvailable}>{rotating ? 'Pause orbit' : 'Resume orbit'}</button>
        <span className="ion-badge">CESIUM ION</span></div></header>

      <button className="dashboard-scrim" onClick={() => setDashboardOpen(false)} aria-label="Close full-screen dashboard" />
      <button className="mobile-sidebar-scrim" onClick={() => setMobileSidebarOpen(false)} aria-label="Close area menu" />
      <div className="mobile-edge-swipe" aria-hidden="true"
        onTouchStart={(event) => { edgeSwipeStartX.current = event.touches[0].clientX }}
        onTouchEnd={(event) => {
          if (event.changedTouches[0].clientX - edgeSwipeStartX.current > 55) setMobileSidebarOpen(true)
        }} />
      <aside className="area-sidebar"
        onTouchStart={(event) => { sidebarSwipeStartX.current = event.touches[0].clientX }}
        onTouchEnd={(event) => {
          if (event.changedTouches[0].clientX - sidebarSwipeStartX.current < -55) setMobileSidebarOpen(false)
        }}>
        <div className="sidebar-heading"><div><span className="eyebrow">SATELLITE WORKSPACE</span><h1>{sel ? 'Selected area' : 'Explore Earth'}</h1></div>
          <div className="heading-actions">{sel && <button className="link" onClick={back}>All areas</button>}<button className="dashboard-toggle" onClick={() => setDashboardOpen((open) => !open)} aria-expanded={dashboardOpen}>{dashboardOpen ? 'Close' : 'Full screen'}<span aria-hidden="true">{dashboardOpen ? ' ↙' : ' ↗'}</span></button></div></div>
        <p className="sidebar-copy">Search storm regions, select an area, then generate an interpolated satellite frame.</p>
        <label className="region-search"><span>Search {regions.length} areas</span>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search area, event, satellite…" /></label>

        {!sel && err && <p className="err backend-error">{err}</p>}
        {sel ? <section className="selected-area">
          <h2>{sel.name}</h2><dl className="meta">
            <div><dt>Satellite</dt><dd>{sel.satellite}</dd></div><div><dt>Event</dt><dd>{sel.event}</dd></div>
            <div><dt>Position</dt><dd>{sel.lat}°, {sel.lng}°</dd></div><div><dt>Source</dt><dd>{source === 'nasa' ? 'NASA GIBS' : `${frames.length} demo frames`}</dd></div>
          </dl>
          {source === 'nasa' && <div className="exact-point-control">
            <button type="button" className={'pick-point' + (pointPicking ? ' active' : '')} onClick={() => setPointPicking((value) => !value)} disabled={!globeAvailable}>
              {pointPicking ? 'Click a point on the globe…' : customPoint ? 'Choose another exact point' : 'Pick exact point on globe'}
            </button>
            <small>{pointPicking ? 'Click the location you want in the globe view.' : customPoint ? 'NASA images will be centered on your selected map point.' : 'Using the preset center. Select a point to change the image location.'}</small>
            <span className="coordinate-caption">Or enter coordinates</span>
            <div className="coordinate-inputs">
              <label>Latitude<input type="number" min="-90" max="90" step="0.00001" value={sel.lat} onChange={(event) => editCoordinate('lat', event.target.value)} /></label>
              <label>Longitude<input type="number" min="-180" max="180" step="0.00001" value={sel.lng} onChange={(event) => editCoordinate('lng', event.target.value)} /></label>
            </div>
            {!validPoint && <small className="time-error">Enter latitude from -90 to 90 and longitude from -180 to 180.</small>}
          </div>}
          <div className="source-switch" aria-label="Image source">
            <button className={source === 'nasa' ? 'on' : ''} onClick={() => { setSource('nasa'); setRes(null); setErr('') }}>NASA satellite imagery</button>
            <button className={source === 'demo' ? 'on' : ''} onClick={() => { setSource('demo'); setRes(null); setErr('') }}>Demo sequence</button>
          </div>
          {source === 'nasa' ? <div className="time-form">
            <p>Choose the image times around the frame you want to generate. Times are entered in your local timezone.</p>
            <label>Before image<input type="datetime-local" value={requestedTimes.before} onChange={(e) => setRequestedTimes((v) => ({ ...v, before: e.target.value }))} /></label>
            <label>Generate at<input type="datetime-local" value={requestedTimes.target} onChange={(e) => setRequestedTimes((v) => ({ ...v, target: e.target.value }))} /></label>
            <label>After image<input type="datetime-local" value={requestedTimes.after} onChange={(e) => setRequestedTimes((v) => ({ ...v, after: e.target.value }))} /></label>
            <small>NASA uses the nearest available observation for each time. Its geostationary archive is limited; recent imagery is generally available for about 90 days. Source images must be within 120 minutes.</small>
            {!validRequestedTimes && <small className="time-error">Enter increasing times with no more than 120 minutes between the source images.</small>}
          </div> : <>
          <h3>Choose the missing frame</h3>
          <div className="strip">{frames.map((f, i) => {
            const inPair = i === start || i === start + 2, gap = i === start + 1
            return <button key={i} disabled={i > 2} onClick={() => { setStart(i); setRes(null) }} className={'th' + (inPair ? ' on' : '') + (gap ? ' gap' : '')}>
              <img src={f.image} alt={f.time} /><span>{gap ? 'generate' : f.time}</span></button>
          })}</div>
          </>}
          <button className="cta" onClick={run} disabled={(source === 'demo' && !frames.length) || (source === 'nasa' && (!validRequestedTimes || !validPoint)) || (stage >= 0 && stage < 5)}>{stage >= 0 && stage < 5 ? 'Generating…' : source === 'nasa' ? 'Fetch images and generate frame' : `Generate ${frames[start + 1]?.time || ''} frame`}</button>
          <ol className="stages">{STAGES.map((name, i) => <li key={name} className={i < stage ? 'done' : i === stage ? 'now' : ''}>{name}</li>)}</ol>
          {err && <p className="err">{err}</p>}
          {res && <div className="result-workspace">
          <div className="result-main">
            <div className="result-heading"><div><span className="eyebrow">INTERPOLATION RESULT</span><h3>{sel.name}</h3></div><span className="gen-pill">{displayTime(res.times.mid)}</span></div>
            <div className="tabs">{[['compare', 'Compare'], ['flow', 'Flow'], ['play', 'Playback']].map(([key, label]) => <button key={key} className={tab === key ? 'on' : ''} onClick={() => setTab(key)}>{label}</button>)}</div>
            {tab === 'compare' && <button className="image-preview" onClick={() => setImageModal(true)} aria-label="Open large generated and real image comparison"><Compare left={res.generated} right={res.ground_truth} /><span>Click image to enlarge and compare</span></button>}
            {tab === 'flow' && <section className="result-visualization"><h4>Optical Flow Visualization</h4><p>Arrows show the estimated cloud movement between the input frames.</p><FlowView src={res.frame_a} arrows={res.arrows} size={res.size} /></section>}
            {tab === 'play' && <section className="result-visualization"><h4>Frame Playback</h4><p>View the before, generated, and after frames in sequence.</p><Player result={res} smooth={smooth} /><label className="sw"><input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} /> Include generated frame</label></section>}
            <p className="note">{res.observation_note || `Cloud motion ${res.mean_motion_px} px / 10 min`}{res.data_source ? ` · ${res.data_source}` : ''} · inference {res.timing_ms.total} ms</p>
            <table><thead><tr><th>Method</th><th>PSNR</th><th>SSIM</th><th>MAE</th></tr></thead>
              <tbody>{METHODS.map(([key, label]) => <tr key={key} className={key === 'warp_refine' ? 'best' : ''}><td>{label}</td><td>{res.metrics[key].psnr}</td><td>{res.metrics[key].ssim}</td><td>{res.metrics[key].mae}</td></tr>)}</tbody></table>
          </div>
          <aside className="result-analytics"><MetricDashboard result={res} /><DiagnosticsPanel result={res} /></aside>
          </div>}
        </section> : <div className="area-list">{shownRegions.map((r) => <button key={r.id} className="area-item" onClick={() => pick(r)}>
          <span className="area-dot"/><span className="area-info"><b>{r.name}</b><small>{r.event} · {r.satellite}</small></span><span className="area-arrow">↗</span></button>)}
          {!shownRegions.length && <p className="empty">No areas match “{search}”.</p>}</div>}
        <footer className="sidebar-footer"><span className="status-dot"/> NASA imagery or synthetic demo sequences</footer>
      </aside>
    </div>
    {imageModal && res && <ImageModal result={res} onClose={closeImageModal} />}
  </div>
}
