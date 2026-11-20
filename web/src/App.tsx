import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
} from 'react'
import {
  BrowserRouter,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useLocation,
  useParams,
  useSearchParams,
} from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import {
  addCaseNote,
  approveCase,
  assetUrl,
  getCase,
  getCases,
  getCosts,
  getHealth,
  getLogs,
  getReport,
  rejectCase,
  saveCase,
  startAnalysis,
  watchJob,
  type CaseDetail,
  type CaseSummary,
  type DebateMessage,
  type HealthResponse,
  type JobResponse,
  type LogCall,
} from './api'
import {
  DefangedLink,
  ErrorState,
  JobStepper,
  LoadingState,
  Panel,
  RiskBar,
  VerdictBadge,
} from './components'
import './index.css'

const buttonClass =
  'inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50'
const primaryButton = `${buttonClass} bg-teal-700 text-white hover:bg-teal-800 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400`
const secondaryButton = `${buttonClass} border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800`

function App() {
  const [darkMode, setDarkMode] = useState(
    () => localStorage.getItem('phishing-desk-theme') === 'dark',
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('phishing-desk-theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
        <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
          <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
            <NavLink to="/" className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-teal-700 text-sm font-black text-white dark:bg-teal-500 dark:text-slate-950">
                PD
              </span>
              <span>
                <span className="block text-sm font-bold tracking-tight">
                  phishing-desk
                </span>
                <span className="block text-[11px] text-slate-500 dark:text-slate-400">
                  Defensive email review
                </span>
              </span>
            </NavLink>
            <nav aria-label="Main navigation" className="flex items-center gap-1">
              <NavigationLink to="/">Analyze</NavigationLink>
              <NavigationLink to="/cases">Cases</NavigationLink>
              <NavigationLink to="/logs">Logs &amp; Costs</NavigationLink>
              <button
                type="button"
                aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                onClick={() => setDarkMode((value) => !value)}
                className={`${secondaryButton} ml-1 h-10 w-10 px-0`}
              >
                {darkMode ? '☼' : '◐'}
              </button>
            </nav>
          </div>
        </header>

        <main className="mx-auto min-h-[calc(100vh-150px)] max-w-7xl px-4 py-7 sm:px-6 lg:py-10">
          <Routes>
            <Route path="/" element={<AnalyzePage />} />
            <Route path="/cases" element={<CasesPage />} />
            <Route path="/cases/:caseId" element={<CaseRoute />} />
            <Route path="/logs" element={<LogsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>
        <StatusFooter />
      </div>
    </BrowserRouter>
  )
}

function NavigationLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        `rounded-lg px-3 py-2 text-sm font-medium transition ${
          isActive
            ? 'bg-teal-50 text-teal-800 dark:bg-teal-950/60 dark:text-teal-200'
            : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900'
        }`
      }
    >
      {children}
    </NavLink>
  )
}

function CaseRoute() {
  const location = useLocation()
  return <CasePage key={location.pathname} />
}

function AnalyzePage() {
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const previewRef = useRef<string | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [offline, setOffline] = useState(false)
  const [noDocker, setNoDocker] = useState(false)
  const [maskPii, setMaskPii] = useState(true)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
  }, [])

  const updateFile = (selected: File | null) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = selected?.type.startsWith('image/')
      ? URL.createObjectURL(selected)
      : null
    setImagePreview(previewRef.current)
    setFile(selected)
  }

  const chooseFile = (selected: File | null) => {
    setError('')
    if (!selected) {
      updateFile(null)
      return
    }
    if (!/\.(txt|eml|png|jpe?g|webp)$/i.test(selected.name)) {
      setError('Choose a .txt, .eml, .png, .jpg, .jpeg, or .webp file.')
      return
    }
    updateFile(selected)
    setText('')
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files[0] ?? null)
  }

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFile(event.target.files?.[0] ?? null)
    event.target.value = ''
  }

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!file && !text.trim()) {
      setError('Paste an email or choose a file to analyze.')
      return
    }
    setBusy(true)
    setError('')
    try {
      const result = await startAnalysis({
        text: file ? undefined : text,
        file: file ?? undefined,
        offline,
        noDocker,
        maskPii,
      })
      navigate(`/cases/${encodeURIComponent(result.case_id)}?job=${encodeURIComponent(result.job_id)}`)
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-8 max-w-2xl">
        <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-teal-800 dark:border-teal-900 dark:bg-teal-950/40 dark:text-teal-200">
          <span className="h-2 w-2 rounded-full bg-teal-500" />
          Local analysis workspace
        </p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Review a suspicious email
        </h1>
        <p className="mt-3 text-base leading-7 text-slate-600 dark:text-slate-400">
          Extract evidence, inspect link and sender indicators, and record a
          human-reviewed outcome.
        </p>
      </div>

      <form onSubmit={submit} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-5">
          <Panel title="Email content" eyebrow="New case">
            <label htmlFor="email-text" className="mb-2 block text-sm font-medium">
              Paste the message text
            </label>
            <textarea
              id="email-text"
              value={text}
              onChange={(event) => {
                setText(event.target.value)
                if (event.target.value.trim()) updateFile(null)
              }}
              rows={13}
              placeholder={'From: sender@example.com\nSubject: Message subject\n\nPaste the email body here…'}
              className="w-full resize-y rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 font-mono text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-teal-500 focus:ring-4 focus:ring-teal-500/10 dark:border-slate-700 dark:bg-slate-950 dark:placeholder:text-slate-600"
            />
            <div className="mt-2 flex justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>Email content is treated as untrusted text.</span>
              <span>{text.length.toLocaleString()} characters</span>
            </div>
          </Panel>

          <Panel title="Upload a file or screenshot" eyebrow="Alternative input">
            <div
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`rounded-2xl border-2 border-dashed p-6 text-center transition ${
                dragging
                  ? 'border-teal-500 bg-teal-50 dark:bg-teal-950/30'
                  : 'border-slate-300 bg-slate-50/70 dark:border-slate-700 dark:bg-slate-950/60'
              }`}
            >
              <input
                id="email-file"
                type="file"
                accept=".txt,.eml,.png,.jpg,.jpeg,.webp"
                onChange={onFileChange}
                className="sr-only"
              />
              <div aria-hidden="true" className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-white text-xl shadow-sm dark:bg-slate-900">
                ⇧
              </div>
              <p className="font-semibold">Drop a file here</p>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Text email or image screenshot
              </p>
              <label
                htmlFor="email-file"
                className={`${secondaryButton} mt-4 cursor-pointer`}
              >
                Browse files
              </label>
              {file && (
                <div className="mt-5 rounded-xl border border-slate-200 bg-white p-3 text-left dark:border-slate-800 dark:bg-slate-900">
                  <div className="flex items-center justify-between gap-3">
                    <span className="truncate text-sm font-medium">{file.name}</span>
                    <button
                      type="button"
                      onClick={() => setFile(null)}
                      className="text-xs font-semibold text-slate-500 hover:text-red-600"
                    >
                      Remove
                    </button>
                  </div>
                  {imagePreview && (
                    <img
                      src={imagePreview}
                      alt="Selected email screenshot preview"
                      className="mt-3 max-h-64 w-full rounded-lg border border-slate-200 object-contain dark:border-slate-800"
                    />
                  )}
                </div>
              )}
            </div>
            <p className="mt-3 text-xs leading-5 text-amber-800 dark:text-amber-200">
              Links come from visible text only and may not match real URLs.
            </p>
          </Panel>
        </div>

        <aside className="space-y-5">
          <Panel title="Analysis options">
            <div className="space-y-4">
              <Toggle
                label="Offline mode"
                description="Use the local rule-based model."
                checked={offline}
                onChange={setOffline}
              />
              <Toggle
                label="No Docker"
                description="Allow local chart generation."
                checked={noDocker}
                onChange={setNoDocker}
              />
              <Toggle
                label="Mask personal data"
                description="Mask email addresses and phone numbers in saved cases."
                checked={maskPii}
                onChange={setMaskPii}
              />
            </div>
            <div className="mt-5 rounded-xl bg-slate-50 p-3 text-xs leading-5 text-slate-600 dark:bg-slate-950 dark:text-slate-400">
              Analysis runs locally through the Phase 7A backend. Uploads are
              limited to 5 MB.
            </div>
          </Panel>
          {error && <ErrorState message={error} />}
          <button type="submit" disabled={busy} className={`${primaryButton} w-full py-3`}>
            {busy ? 'Starting analysis…' : 'Analyze email'}
          </button>
        </aside>
      </form>
    </div>
  )
}

function Toggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4">
      <span>
        <span className="block text-sm font-semibold">{label}</span>
        <span className="mt-1 block text-xs leading-5 text-slate-500 dark:text-slate-400">
          {description}
        </span>
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-5 w-9 shrink-0 cursor-pointer appearance-none rounded-full bg-slate-300 transition checked:bg-teal-600 after:block after:ml-0.5 after:mt-0.5 after:h-4 after:w-4 after:rounded-full after:bg-white after:transition checked:after:ml-4 dark:bg-slate-700"
      />
    </label>
  )
}

function CasePage() {
  const { caseId = '' } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const jobId = searchParams.get('job')
  const [data, setData] = useState<CaseDetail | null>(null)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [actionBusy, setActionBusy] = useState(false)
  const [rejectOpen, setRejectOpen] = useState(false)
  const [noteOpen, setNoteOpen] = useState(false)
  const [rejectReason, setRejectReason] = useState('')
  const [note, setNote] = useState('')
  const [report, setReport] = useState<string | null>(null)
  const [reportError, setReportError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    if (!jobId) {
      void getCase(caseId, controller.signal)
        .then(setData)
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setError(errorMessage(reason))
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false)
        })
      return () => controller.abort()
    }

    void watchJob(
      jobId,
      (nextJob) => setJob(nextJob),
      controller.signal,
    )
      .then(async (finished) => {
        if (finished.status === 'failed') {
          throw new Error(finished.error_text || 'The background job failed.')
        }
        const detail = await getCase(caseId, controller.signal)
        if (controller.signal.aborted) return
        setData(detail)
        navigate(`/cases/${encodeURIComponent(caseId)}`, { replace: true })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(reason))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [caseId, jobId, navigate])

  const refresh = async () => {
    const updated = await getCase(caseId)
    setData(updated)
  }

  const act = async (operation: () => Promise<unknown>) => {
    setActionBusy(true)
    setActionError('')
    try {
      await operation()
      await refresh()
    } catch (reason) {
      setActionError(errorMessage(reason))
    } finally {
      setActionBusy(false)
    }
  }

  const approve = () => act(() => approveCase(caseId))

  const reject = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!rejectReason.trim()) return
    setActionBusy(true)
    setActionError('')
    try {
      const started = await rejectCase(caseId, rejectReason.trim())
      setRejectOpen(false)
      setRejectReason('')
      setJob(null)
      setError('')
      setActionBusy(false)
      navigate(`/cases/${encodeURIComponent(caseId)}?job=${encodeURIComponent(started.job_id)}`)
    } catch (reason) {
      setActionError(errorMessage(reason))
      setActionBusy(false)
    }
  }

  const addNote = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!note.trim()) return
    await act(async () => {
      await addCaseNote(caseId, note.trim())
      setNote('')
      setNoteOpen(false)
    })
  }

  const saveAndExit = async () => {
    setActionBusy(true)
    setActionError('')
    try {
      await saveCase(caseId)
      navigate('/cases')
    } catch (reason) {
      setActionError(errorMessage(reason))
    } finally {
      setActionBusy(false)
    }
  }

  const showReport = async () => {
    setReportError('')
    try {
      setReport(await getReport(caseId))
    } catch (reason) {
      setReportError(errorMessage(reason))
    }
  }

  const downloadReport = async () => {
    try {
      const markdown = report ?? await getReport(caseId)
      setReport(markdown)
      const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${caseId}.md`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (reason) {
      setReportError(errorMessage(reason))
    }
  }

  if (loading && !data) {
    return (
      <div className="mx-auto max-w-5xl space-y-5">
        {job && (
          <Panel title="Analysis in progress" eyebrow="Live job">
            <JobStepper currentStep={job.current_step} status={job.status} />
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">
              Current step: {job.current_step ?? 'Queued'}
            </p>
          </Panel>
        )}
        {error ? <ErrorState message={error} /> : <LoadingState label="Loading case…" />}
      </div>
    )
  }
  if (!data) {
    return (
      <div className="mx-auto max-w-3xl space-y-4">
        <PageHeading eyebrow="Case" title="Case unavailable" />
        <ErrorState message={error || 'The case could not be loaded.'} />
        <NavLink to="/cases" className={secondaryButton}>Back to cases</NavLink>
      </div>
    )
  }

  const verdict = parseVerdict(data.current_verdict)
  const lastSkeptic = [...data.debate_history]
    .reverse()
    .find((message) => message.speaker.toLowerCase() === 'skeptic')
  const skepticAgreed = Boolean(lastSkeptic && /\bAGREE\s*$/.test(lastSkeptic.message.trim()))
  const attempt = Math.max(1, ...data.debate_history.map((message) => message.attempt))
  const rejectionCount = Math.max(0, attempt - 1)
  const canDecide = data.status === 'OPEN'
  const jobPending = job?.status === 'queued' || job?.status === 'running'

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {job && (
        <Panel title={job.status === 'failed' ? 'Job failed' : 'Analysis progress'} eyebrow="Live job">
          <JobStepper currentStep={job.current_step} status={job.status} />
          <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">
            Current step: {job.current_step ?? 'Queued'}
          </p>
          {job.status === 'failed' && job.error_text && <ErrorState message={job.error_text} />}
        </Panel>
      )}
      {error && <ErrorState message={error} />}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <PageHeading eyebrow={`Case ${data.id}`} title="Analysis findings" />
        <StatusBadge status={data.status} />
      </div>

      <Panel title="Verdict" eyebrow="Decision summary" className="overflow-hidden">
        <div className="grid gap-6 md:grid-cols-[1fr_260px]">
          <div>
            <div className="flex flex-wrap items-center gap-4">
              <VerdictBadge verdict={verdict.name} />
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {verdict.confidence || 'Confidence not provided'}
              </span>
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
              <div
                className={`h-full rounded-full ${
                  verdict.name === 'PHISHING'
                    ? 'bg-red-500'
                    : verdict.name === 'SUSPICIOUS'
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                }`}
                style={{ width: `${verdict.confidenceValue}%` }}
              />
            </div>
            <p className="mt-5 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Sender domain findings
            </p>
            <p className="mt-1 text-sm leading-6">{verdict.senderFinding || data.sender_domain_findings}</p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
            <p className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Reasons
            </p>
            <ol className="space-y-2 text-sm leading-5">
              {verdict.reasons.length ? verdict.reasons.slice(0, 3).map((reason, index) => (
                <li key={`${index}-${reason}`} className="flex gap-2">
                  <span className="font-bold text-teal-700 dark:text-teal-300">{index + 1}.</span>
                  <span>{reason}</span>
                </li>
              )) : <li className="text-slate-500">No reasons were returned.</li>}
            </ol>
          </div>
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Link findings" eyebrow={`${data.link_findings.length} checked`}>
          {data.link_findings.length ? (
            <div className="space-y-4">
              {data.link_findings.map((finding, index) => (
                <div key={`${finding.display_url}-${index}`} className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
                  <DefangedLink finding={finding} />
                  <div className="mt-3">
                    <RiskBar score={finding.risk_score} />
                  </div>
                  {finding.reasons.length > 0 && (
                    <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-400">
                      {finding.reasons.map((reason) => <li key={reason}>{reason}</li>)}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          ) : <EmptyState text="No links were extracted from this email." />}
        </Panel>

        <Panel title="Authentication results" eyebrow="Header evidence">
          <div className="flex flex-wrap gap-2">
            {['SPF', 'DKIM', 'DMARC'].map((protocol) => {
              const status = data.auth_results.find((item) => item.startsWith(`${protocol}:`))?.split(':').slice(1).join(':').trim() ?? 'not available'
              const explicitFail = status.toLowerCase() === 'fail'
              const passed = status.toLowerCase() === 'pass'
              return (
                <span
                  key={protocol}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold ${
                    explicitFail
                      ? 'border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200'
                      : passed
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200'
                        : 'border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300'
                  }`}
                >
                  {protocol}: {status}
                </span>
              )
            })}
          </div>
          <p className="mt-4 text-xs leading-5 text-slate-500 dark:text-slate-400">
            Missing or unavailable authentication results are informational, not evidence of phishing.
          </p>
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Debate timeline" eyebrow="Classifier & skeptic">
          <div className="mb-4">
            {data.current_verdict.includes('UNRESOLVED') ? (
              <span className="rounded-full bg-amber-100 px-3 py-1.5 text-xs font-bold text-amber-900 dark:bg-amber-950 dark:text-amber-200">UNRESOLVED</span>
            ) : (
              <span className={`rounded-full px-3 py-1.5 text-xs font-bold ${
                skepticAgreed
                  ? 'bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200'
                  : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
              }`}>
                Skeptic {skepticAgreed ? 'AGREED' : 'not yet agreed'}
              </span>
            )}
          </div>
          {data.debate_history.length ? (
            <div className="space-y-3">
              {data.debate_history.map((message, index) => (
                <DebateBubble key={`${message.attempt}-${message.round}-${index}`} message={message} />
              ))}
            </div>
          ) : <EmptyState text="No debate messages were saved." />}
        </Panel>

        <Panel title="Policy matches" eyebrow="Local guidance">
          {data.policy_matches.length ? (
            <div className="space-y-3">
              {data.policy_matches.map((item, index) => (
                <EvidenceCard key={`${item.rule_id}-${index}`} id={item.rule_id} title={item.rule_text} detail={item.why_matches} />
              ))}
            </div>
          ) : <EmptyState text="No policy matches were returned." />}
          <h3 className="mb-3 mt-6 text-sm font-semibold">Similar past cases</h3>
          {data.similar_past_cases.length ? (
            <div className="space-y-3">
              {data.similar_past_cases.map((item, index) => (
                <EvidenceCard key={`${item.case_id}-${index}`} id={item.case_id} title={item.summary} detail={item.why_similar} />
              ))}
            </div>
          ) : <EmptyState text="No similar sample cases were found." />}
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Original email text" eyebrow="Plain text only">
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-slate-50 p-4 font-mono text-xs leading-5 text-slate-700 dark:bg-slate-950 dark:text-slate-300">
            {data.email_text}
          </pre>
        </Panel>
        <Panel title="Risk chart" eyebrow="Rule-based scores">
          {data.chart_url ? (
            <img
              src={assetUrl(data.chart_url)}
              alt="Email sender and link risk score chart"
              className="w-full rounded-xl border border-slate-200 dark:border-slate-800"
            />
          ) : <EmptyState text="No chart is available for this case." />}
          <div className="mt-4 rounded-xl bg-slate-50 p-3 text-sm dark:bg-slate-950">
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 dark:text-slate-400">Recorded analysis cost</span>
              <strong>{formatCurrency(readTotalCost(data.cost))}</strong>
            </div>
          </div>
        </Panel>
      </div>

      {data.analyst_notes.length > 0 && (
        <Panel title="Analyst notes">
          <ul className="space-y-2 text-sm">
            {data.analyst_notes.map((item, index) => <li key={`${index}-${item}`} className="rounded-lg bg-slate-50 p-3 dark:bg-slate-950">{item}</li>)}
          </ul>
        </Panel>
      )}

      <Panel title="Case actions" eyebrow="Human approval">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => void approve()} disabled={!canDecide || actionBusy || jobPending} className={primaryButton}>
            {actionBusy ? 'Working…' : 'Approve'}
          </button>
          <button
            type="button"
            onClick={() => setRejectOpen((value) => !value)}
            disabled={!canDecide || actionBusy || jobPending || rejectionCount >= 2}
            className={`${buttonClass} border border-amber-300 bg-amber-50 text-amber-900 hover:bg-amber-100 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200`}
          >
            Reject
          </button>
          <button type="button" onClick={() => setNoteOpen((value) => !value)} disabled={actionBusy} className={secondaryButton}>
            Add note
          </button>
          <button type="button" onClick={() => void saveAndExit()} disabled={actionBusy} className={secondaryButton}>
            Save and exit
          </button>
          <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
            Rejections: {rejectionCount}/2
          </span>
        </div>
        {rejectOpen && (
          <form onSubmit={reject} className="mt-4 rounded-xl border border-amber-200 bg-amber-50/70 p-4 dark:border-amber-900 dark:bg-amber-950/20">
            <label htmlFor="reject-reason" className="mb-2 block text-sm font-semibold">What concrete evidence should the debate reconsider?</label>
            <textarea id="reject-reason" value={rejectReason} onChange={(event) => setRejectReason(event.target.value)} maxLength={1000} rows={3} required className="w-full rounded-lg border border-amber-300 bg-white p-3 text-sm outline-none focus:ring-2 focus:ring-amber-400 dark:border-amber-900 dark:bg-slate-950" />
            <div className="mt-3 flex gap-2">
              <button type="submit" disabled={actionBusy} className={primaryButton}>Send challenge</button>
              <button type="button" onClick={() => setRejectOpen(false)} className={secondaryButton}>Cancel</button>
            </div>
          </form>
        )}
        {noteOpen && (
          <form onSubmit={addNote} className="mt-4 rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <label htmlFor="analyst-note" className="mb-2 block text-sm font-semibold">Analyst note</label>
            <textarea id="analyst-note" value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} rows={3} required className="w-full rounded-lg border border-slate-300 bg-white p-3 text-sm outline-none focus:ring-2 focus:ring-teal-500 dark:border-slate-700 dark:bg-slate-950" />
            <button type="submit" disabled={actionBusy} className={`${primaryButton} mt-3`}>Save note</button>
          </form>
        )}
        {actionError && <div className="mt-4"><ErrorState message={actionError} /></div>}
        <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-slate-200 pt-4 dark:border-slate-800">
          <button type="button" onClick={() => void showReport()} disabled={!data.report_url} className={secondaryButton}>
            View report
          </button>
          <button type="button" onClick={() => void downloadReport()} disabled={!data.report_url} className={secondaryButton}>
            Download report
          </button>
          {reportError && <span className="text-sm text-red-700 dark:text-red-300">{reportError}</span>}
        </div>
      </Panel>

      {report && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" role="dialog" aria-modal="true" aria-label="Case report">
          <div className="max-h-[85vh] w-full max-w-3xl overflow-auto rounded-2xl bg-white p-6 shadow-2xl dark:bg-slate-900">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-lg font-bold">Case report</h2>
              <button type="button" onClick={() => setReport(null)} className={secondaryButton}>Close</button>
            </div>
            <div className="markdown-content">
              <ReactMarkdown components={{ a: ({ children }) => <span>{children}</span> }}>{report}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function DebateBubble({ message }: { message: DebateMessage }) {
  const classifier = message.speaker.toLowerCase() === 'classifier'
  return (
    <div className={`max-w-[95%] rounded-2xl p-3 text-sm ${
      classifier
        ? 'mr-auto rounded-tl-md bg-slate-100 dark:bg-slate-800'
        : 'ml-auto rounded-tr-md bg-teal-50 dark:bg-teal-950/50'
    }`}>
      <div className="mb-1 flex items-center gap-2 text-[11px] font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        <span>{message.speaker}</span>
        <span>Attempt {message.attempt} · Round {message.round}</span>
      </div>
      <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-5">{message.message}</pre>
    </div>
  )
}

function EvidenceCard({ id, title, detail }: { id?: string; title?: string; detail?: string }) {
  return (
    <article className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
      <div className="flex items-start gap-2">
        {id && <span className="rounded-md bg-teal-50 px-2 py-1 text-[10px] font-bold text-teal-800 dark:bg-teal-950 dark:text-teal-200">{id}</span>}
        <p className="text-sm font-semibold leading-5">{title || 'Evidence item'}</p>
      </div>
      {detail && <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">{detail}</p>}
    </article>
  )
}

function CasesPage() {
  const [rows, setRows] = useState<CaseSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [verdictFilter, setVerdictFilter] = useState('ALL')
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const controller = new AbortController()
    void getCases(controller.signal)
      .then(setRows)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(reason))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  const filtered = useMemo(() => rows.filter((item) => {
    const matchesStatus = statusFilter === 'ALL' || item.status === statusFilter
    const matchesVerdict = verdictFilter === 'ALL' || item.verdict === verdictFilter
    const matchesQuery = `${item.id} ${item.verdict} ${item.status}`.toLowerCase().includes(query.toLowerCase())
    return matchesStatus && matchesVerdict && matchesQuery
  }), [rows, statusFilter, verdictFilter, query])

  return (
    <div className="space-y-6">
      <PageHeading eyebrow="Case library" title="Cases" subtitle="Browse saved analyses and resume open decisions." />
      <Panel title="Saved cases">
        <div className="mb-4 grid gap-3 sm:grid-cols-[1fr_180px_180px]">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search case ID or verdict" aria-label="Search cases" className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-teal-500 dark:border-slate-700 dark:bg-slate-950" />
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} aria-label="Filter by status" className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950">
            <option value="ALL">All statuses</option>
            {['OPEN', 'APPROVED', 'REJECTED', 'NEEDS_REVIEW'].map((status) => <option key={status}>{status}</option>)}
          </select>
          <select value={verdictFilter} onChange={(event) => setVerdictFilter(event.target.value)} aria-label="Filter by verdict" className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950">
            <option value="ALL">All verdicts</option>
            {['PHISHING', 'SUSPICIOUS', 'LEGIT', 'UNKNOWN'].map((verdict) => <option key={verdict}>{verdict}</option>)}
          </select>
        </div>
        {error && <ErrorState message={error} />}
        {loading ? <LoadingState label="Loading saved cases…" /> : filtered.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
                <tr><th className="py-3 pr-4">Case ID</th><th className="py-3 pr-4">Status</th><th className="py-3 pr-4">Verdict</th><th className="py-3 pr-4">Confidence</th><th className="py-3 pr-4">Date</th><th className="py-3">Open</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {filtered.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50 dark:hover:bg-slate-900/70">
                    <td className="py-3 pr-4 font-mono text-xs">{item.id}</td>
                    <td className="py-3 pr-4"><StatusBadge status={item.status} /></td>
                    <td className="py-3 pr-4"><VerdictBadge verdict={item.verdict} /></td>
                    <td className="py-3 pr-4">{item.confidence || '—'}</td>
                    <td className="py-3 pr-4 text-xs text-slate-500">{formatDate(item.date)}</td>
                    <td className="py-3">
                      <button type="button" onClick={() => navigate(`/cases/${encodeURIComponent(item.id)}`)} className={`${secondaryButton} px-3 py-1.5`}>
                        {item.status === 'OPEN' ? 'Resume' : 'View'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : !error ? <EmptyState text="No cases match these filters." /> : null}
      </Panel>
    </div>
  )
}

function LogsPage() {
  const [calls, setCalls] = useState<LogCall[]>([])
  const [cost, setCost] = useState<{ total_cost: number; cost_today: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    void Promise.all([getLogs(20, controller.signal), getCosts(controller.signal)])
      .then(([logs, costs]) => {
        setCalls(logs.calls)
        setCost(costs)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(reason))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  return (
    <div className="space-y-6">
      <PageHeading eyebrow="Observability" title="Logs & Costs" subtitle="Recent model-call metadata and saved case spend." />
      {error && <ErrorState message={error} />}
      <div className="grid gap-4 sm:grid-cols-2">
        <CostCard title="Total saved cost" amount={cost?.total_cost ?? 0} loading={loading} />
        <CostCard title="Cost today" amount={cost?.cost_today ?? 0} loading={loading} />
      </div>
      <Panel title="Recent LLM calls" eyebrow={`${calls.length} latest entries`}>
        {loading ? <LoadingState label="Loading call logs…" /> : calls.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
                <tr><th className="py-3 pr-4">Agent</th><th className="py-3 pr-4">Model</th><th className="py-3 pr-4">Tokens</th><th className="py-3 pr-4">Cost</th><th className="py-3">Time</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {calls.map((call, index) => (
                  <tr key={`${call.agent}-${call.timestamp ?? call.time}-${index}`}>
                    <td className="py-3 pr-4 font-medium">{call.agent || '—'}</td>
                    <td className="py-3 pr-4 text-xs">{call.model || '—'}</td>
                    <td className="py-3 pr-4 tabular-nums">{(call.total_tokens ?? call.tokens ?? ((call.prompt_tokens ?? 0) + (call.completion_tokens ?? 0))).toLocaleString()}</td>
                    <td className="py-3 pr-4 tabular-nums">{formatCurrency(call.cost ?? 0)}</td>
                    <td className="py-3 text-xs text-slate-500">{formatDate(call.timestamp ?? call.time ?? '')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : !error ? <EmptyState text="No model calls have been logged yet." /> : null}
      </Panel>
    </div>
  )
}

function CostCard({ title, amount, loading }: { title: string; amount: number; loading: boolean }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">{title}</p>
      {loading ? <div className="mt-3 h-8 w-28 animate-pulse rounded bg-slate-200 dark:bg-slate-800" /> : <p className="mt-2 text-3xl font-bold tracking-tight">{formatCurrency(amount)}</p>}
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">USD, based on saved AutoGen usage</p>
    </div>
  )
}

function StatusFooter() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    let timer: number | undefined
    const update = async () => {
      try {
        const result = await getHealth(controller.signal)
        setHealth(result)
        setError('')
      } catch (reason) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setError(errorMessage(reason))
        }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(() => void update(), 10000)
      }
    }
    void update()
    return () => {
      controller.abort()
      if (timer) window.clearTimeout(timer)
    }
  }, [])

  return (
    <footer className="border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 text-xs sm:px-6">
        <div className="flex flex-wrap items-center gap-4">
          <span className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${health ? 'bg-emerald-500' : 'bg-red-500'}`} />
            Backend {health ? 'online' : 'offline'}
          </span>
          <span>Groq key: {health ? (health.groq_key_configured ? 'configured' : 'not configured') : 'unknown'}</span>
          <span>Docker: {health ? (health.docker_available ? 'available' : 'unavailable') : 'unknown'}</span>
        </div>
        {error && <span className="text-red-700 dark:text-red-300">{error}</span>}
        <span className="text-slate-500">Local defensive analysis · Phase 7B</span>
      </div>
    </footer>
  )
}

function PageHeading({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle?: string }) {
  return (
    <div>
      <p className="mb-2 text-xs font-bold uppercase tracking-[0.16em] text-teal-700 dark:text-teal-300">{eyebrow}</p>
      <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
      {subtitle && <p className="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">{subtitle}</p>}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const color =
    status === 'APPROVED'
      ? 'bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200'
      : status === 'NEEDS_REVIEW'
        ? 'bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200'
        : status === 'REJECTED'
          ? 'bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200'
          : 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200'
  return <span className={`inline-flex rounded-full px-3 py-1.5 text-[11px] font-bold tracking-wide ${color}`}>{status}</span>
}

function EmptyState({ text }: { text: string }) {
  return <p className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">{text}</p>
}

function NotFoundPage() {
  return (
    <div className="mx-auto max-w-xl py-16 text-center">
      <p className="text-sm font-bold uppercase tracking-widest text-teal-700 dark:text-teal-300">404</p>
      <h1 className="mt-3 text-3xl font-bold">Page not found</h1>
      <NavLink to="/" className={`${primaryButton} mt-6`}>Return to Analyze</NavLink>
    </div>
  )
}

function parseVerdict(text: string): {
  name: string
  confidence: string
  confidenceValue: number
  senderFinding: string
  reasons: string[]
} {
  const verdict = text.match(/\b(PHISHING|SUSPICIOUS|LEGIT)\b/i)?.[1]?.toUpperCase() ?? 'UNKNOWN'
  const confidence = text.match(/Confidence:\s*(\d{1,3}%?)/i)?.[1] ?? ''
  const confidenceValue = Math.min(100, Number.parseInt(confidence, 10) || 0)
  const senderFinding = text.match(/Sender domain findings:\s*(.+)/i)?.[1]?.trim() ?? ''
  const reasons = [...text.matchAll(/^\s*(?:\d+[.)]|[-*])\s+(.+)\s*$/gm)]
    .map((match) => match[1].trim())
    .filter((reason) => !/^TERMINATE$/i.test(reason))
    .slice(0, 3)
  return { name: verdict, confidence, confidenceValue, senderFinding, reasons }
}

function readTotalCost(value: Record<string, unknown>): number {
  const excluding = value.usage_excluding_cached_inference
  const including = value.usage_including_cached_inference
  const selected = (excluding ?? including) as Record<string, unknown> | undefined
  return typeof selected?.total_cost === 'number' ? selected.total_cost : 0
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 6 }).format(value)
}

function formatDate(value: string): string {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : 'An unexpected error occurred.'
}

export default App
