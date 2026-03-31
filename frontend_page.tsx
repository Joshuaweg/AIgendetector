'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

type Prediction = {
  class: 'AI-Generated' | 'Real'
  confidence: number
  probabilities: { ai_generated: number; real: number }
}

type ClassifyResult = {
  success: boolean
  video_id?: string
  prediction?: Prediction
  attribution_video_url?: string
  error?: string
}

type FeedbackState = 'idle' | 'pending' | 'submitted'
type AttributionStatus = 'idle' | 'pending' | 'complete' | 'error'

const MAX_DURATION = 30
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']
const FLASK_BASE = process.env.NEXT_PUBLIC_CLASSIFIER_API_URL || 'http://localhost:5000'
const LAMBDA_WAKE_URL = process.env.NEXT_PUBLIC_LAMBDA_WAKE_URL || ''

export default function GenAIClassifierDemo() {
  const [file, setFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [serverWaking, setServerWaking] = useState(false)
  const [withExplanations, setWithExplanations] = useState(false)
  const [result, setResult] = useState<ClassifyResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('idle')
  const [modelUsed, setModelUsed] = useState('')
  const [attributionStatus, setAttributionStatus] = useState<AttributionStatus>('idle')
  const [attributionUrl, setAttributionUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const attributionAbortRef = useRef<AbortController | null>(null)

  // Cancel any in-flight attribution request on unmount
  useEffect(() => {
    return () => {
      attributionAbortRef.current?.abort()
    }
  }, [])

  const validateAndSet = useCallback((f: File) => {
    if (!ALLOWED_TYPES.includes(f.type)) {
      setError('Invalid format. Use MP4, MOV, AVI, or WebM.')
      return
    }
    const url = URL.createObjectURL(f)
    const vid = document.createElement('video')
    vid.src = url
    vid.onloadedmetadata = () => {
      if (vid.duration > MAX_DURATION) {
        setError(`Video must be ≤${MAX_DURATION}s (yours: ${Math.round(vid.duration)}s)`)
        URL.revokeObjectURL(url)
        return
      }
      setFile(f)
      setVideoUrl(url)
      setResult(null)
      setError(null)
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const f = e.dataTransfer.files[0]
      if (f) validateAndSet(f)
    },
    [validateAndSet]
  )

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) validateAndSet(f)
  }

  const fetchAttribution = useCallback(async (currentFile: File) => {
    attributionAbortRef.current?.abort()
    const controller = new AbortController()
    attributionAbortRef.current = controller

    setAttributionStatus('pending')
    setAttributionUrl(null)

    try {
      const formData = new FormData()
      formData.append('file', currentFile)
      formData.append('generate_explanations', 'true')

      const res = await fetch(`${FLASK_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      const data: ClassifyResult = await res.json()

      if (!data.success || !data.video_id) {
        setAttributionStatus('error')
        return
      }

      const videoId = data.video_id
      while (true) {
        if (controller.signal.aborted) return
        await new Promise((r) => setTimeout(r, 4000))
        if (controller.signal.aborted) return

        try {
          const statusRes = await fetch(`${FLASK_BASE}/api/attributions/status/${videoId}`, {
            signal: controller.signal,
          })
          if (!statusRes.ok) continue

          const statusData = await statusRes.json()

          if (statusData.status === 'ready') {
            const downloadUrl = `${FLASK_BASE}/api/download/${videoId}`
            const headRes = await fetch(downloadUrl, { method: 'HEAD', signal: controller.signal })
            if (!headRes.ok) continue
            setAttributionUrl(downloadUrl)
            setAttributionStatus('complete')
            return
          } else if (statusData.status === 'error') {
            setAttributionStatus('error')
            return
          }
        } catch (pollError) {
          if ((pollError as Error).name === 'AbortError') return
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setAttributionStatus('error')
      }
    }
  }, [])

  const analyze = async (currentFile?: File) => {
    const f = currentFile ?? file
    if (!f) return
    setLoading(true)
    setError(null)
    setAttributionStatus('idle')
    setAttributionUrl(null)
    attributionAbortRef.current?.abort()

    try {
      const formData = new FormData()
      formData.append('file', f)

      let res: Response
      try {
        res = await fetch(`${FLASK_BASE}/api/predict`, { method: 'POST', body: formData })
      } catch {
        setLoading(false)
        wakeAndRetry(f)
        return
      }

      const data: ClassifyResult = await res.json()
      if (!data.success) throw new Error(data.error || 'Classification failed')
      setResult(data)
      setFeedbackState('idle')
      setModelUsed('')

      if (withExplanations) {
        fetchAttribution(f)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const wakeAndRetry = async (f: File) => {
    setServerWaking(true)
    setError(null)

    await fetch(LAMBDA_WAKE_URL, { method: 'POST' }).catch(() => {})

    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 5000))
      try {
        const res = await fetch(`${FLASK_BASE}/api/health`)
        if (res.ok) {
          setServerWaking(false)
          analyze(f)
          return
        }
      } catch {}
    }

    setServerWaking(false)
    setError('Server took too long to start. Please try again.')
  }

  const submitFeedback = async (wasCorrect: boolean) => {
    if (!result?.video_id || !result?.prediction) return
    setFeedbackState('pending')
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: result.video_id,
          prediction: result.prediction.class,
          was_correct: wasCorrect,
          model_used: modelUsed.trim() || undefined,
        }),
      })
    } finally {
      setFeedbackState('submitted')
    }
  }

  const isAI = result?.prediction?.class === 'AI-Generated'
  const pred = result?.prediction

  return (
    <main className="min-h-screen bg-black text-white py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <a
          href="/"
          className="text-orange-400 hover:text-blue-400 text-sm mb-8 block transition-colors duration-200"
        >
          ← Back
        </a>
        <h1 className="text-3xl font-bold text-orange-400 mb-2">GenAI Video Classifier</h1>
        <p className="text-gray-400 text-sm mb-8">
          Upload a short video (max 30s) to detect AI-generated content. The result is stamped
          directly onto the video as an authenticity badge. Tested at 87% accuracy — the model
          rarely flags real videos as AI-generated, so a positive detection is a strong signal.
        </p>

        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-all duration-200 ${
            dragging
              ? 'border-orange-400 bg-gray-900'
              : 'border-gray-600 hover:border-gray-500 hover:bg-gray-900/50'
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
            className="hidden"
            onChange={handleFileChange}
          />
          {file ? (
            <p className="text-orange-400 font-medium">{file.name}</p>
          ) : (
            <>
              <p className="text-gray-300">Drop video here or click to browse</p>
              <p className="text-gray-600 text-xs mt-1">MP4 · MOV · AVI · WebM · max 30 seconds</p>
            </>
          )}
        </div>

        {/* Options */}
        <label className="mt-4 flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none w-fit">
          <input
            type="checkbox"
            checked={withExplanations}
            onChange={(e) => setWithExplanations(e.target.checked)}
            className="accent-orange-400"
          />
          Generate IG attribution video (shows regions driving the AI-Generated or Real prediction)
        </label>

        {/* Error */}
        {error && (
          <div className="mt-4 bg-red-950 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {/* Analyze button */}
        {file && (
          <button
            onClick={() => analyze()}
            disabled={loading || serverWaking}
            className="mt-6 w-full bg-gray-800 text-orange-400 py-3 rounded-lg font-semibold border-2 border-gray-600 hover:text-blue-400 hover:bg-gray-700 hover:border-gray-500 transition-all duration-300 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing...
              </span>
            ) : (
              'Analyze Video'
            )}
          </button>
        )}

        {/* Server warming UI */}
        {serverWaking && (
          <div className="mt-4 flex items-center gap-3 text-gray-400 text-sm bg-gray-900 border border-gray-700 rounded-lg px-4 py-3">
            <svg className="h-4 w-4 animate-spin flex-shrink-0" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Server warming up — this takes 60–90 seconds on first use…
          </div>
        )}

        {/* Results */}
        {result && pred && videoUrl && (
          <div className="mt-10 space-y-6">
            {/* Video with stamp */}
            <div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Result</p>
              <div className="relative rounded-lg overflow-hidden bg-gray-950">
                <video src={videoUrl} controls className="w-full" />
                <div
                  className={`absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1.5 rounded font-bold text-xs tracking-widest shadow-xl backdrop-blur-sm border ${
                    isAI
                      ? 'bg-red-950/90 border-red-500 text-red-300'
                      : 'bg-green-950/90 border-green-500 text-green-300'
                  }`}
                >
                  {isAI ? '⚠ AI GENERATED' : '✓ AUTHENTIC'}
                </div>
              </div>
            </div>

            {/* Confidence breakdown */}
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 space-y-5">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Classification</span>
                <span className={`font-bold text-base ${isAI ? 'text-red-400' : 'text-green-400'}`}>
                  {pred.class}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Confidence</span>
                <span className="text-white font-medium">{(pred.confidence * 100).toFixed(1)}%</span>
              </div>

              <div className="space-y-3">
                {[
                  { label: 'AI Generated', value: pred.probabilities.ai_generated, color: 'bg-red-500' },
                  { label: 'Real', value: pred.probabilities.real, color: 'bg-green-500' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{label}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`${color} h-2 rounded-full transition-all duration-700`}
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Feedback */}
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 space-y-4">
              <p className="text-sm text-gray-400 font-medium">Was this classification correct?</p>

              {feedbackState === 'submitted' ? (
                <p className="text-green-400 text-sm">Thanks for the feedback.</p>
              ) : (
                <>
                  <div className="space-y-1">
                    <label className="text-xs text-gray-500">
                      {isAI
                        ? 'Which model generated this video? (optional)'
                        : 'If AI-generated, which model? (optional)'}
                    </label>
                    <input
                      type="text"
                      value={modelUsed}
                      onChange={(e) => setModelUsed(e.target.value)}
                      placeholder="e.g. Sora, Kling, Runway Gen-3…"
                      className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded px-3 py-2 placeholder-gray-600 focus:outline-none focus:border-orange-400"
                    />
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => submitFeedback(true)}
                      disabled={feedbackState === 'pending'}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg border border-green-700 text-green-400 text-sm hover:bg-green-950 transition-colors disabled:opacity-50"
                    >
                      ✓ Correct
                    </button>
                    <button
                      onClick={() => submitFeedback(false)}
                      disabled={feedbackState === 'pending'}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg border border-red-800 text-red-400 text-sm hover:bg-red-950 transition-colors disabled:opacity-50"
                    >
                      ✗ Incorrect
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* IG attribution video */}
            {attributionStatus !== 'idle' && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 uppercase tracking-wider">
                  IG Attribution Visualization
                </p>
                <p className="text-xs text-gray-600">
                  Highlighted regions show areas most influential to the{' '}
                  <span className={isAI ? 'text-red-400' : 'text-green-400'}>
                    {pred?.class ?? 'classified'}
                  </span>{' '}
                  label.
                </p>

                {attributionStatus === 'pending' && (
                  <div className="rounded-lg bg-gray-950 border border-gray-800 p-6 flex items-center gap-3 text-gray-400 text-sm">
                    <svg className="h-4 w-4 animate-spin flex-shrink-0" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Generating attribution video…
                  </div>
                )}

                {attributionStatus === 'complete' && attributionUrl && (
                  <div className="rounded-lg overflow-hidden bg-gray-950">
                    <video src={attributionUrl} controls className="w-full" />
                  </div>
                )}

                {attributionStatus === 'error' && (
                  <div className="rounded-lg bg-red-950 border border-red-800 p-4 text-red-400 text-sm">
                    Attribution generation failed.
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
