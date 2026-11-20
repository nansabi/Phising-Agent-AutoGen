import type { JobStep, JobStatus, LinkFinding } from './api'
import type { ReactNode } from 'react'

const JOB_STEPS: JobStep[] = [
  'Extract',
  'Check',
  'Classify',
  'Debate',
  'Policy',
  'Chart',
  'Awaiting approval',
]

export function VerdictBadge({ verdict }: { verdict: string }) {
  const normalized = verdict.toUpperCase()
  const tone =
    normalized === 'PHISHING'
      ? 'red'
      : normalized === 'SUSPICIOUS'
        ? 'amber'
        : normalized === 'LEGIT'
          ? 'green'
          : 'slate'
  return (
    <span
      data-testid="verdict-badge"
      data-tone={tone}
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-bold tracking-wide ${
        tone === 'red'
          ? 'border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/60 dark:text-red-200'
          : tone === 'amber'
            ? 'border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/60 dark:text-amber-200'
            : tone === 'green'
              ? 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/60 dark:text-emerald-200'
              : 'border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200'
      }`}
    >
      <span aria-hidden="true" className="h-2 w-2 rounded-full bg-current" />
      {normalized}
    </span>
  )
}

export function DefangedLink({ finding }: { finding: LinkFinding }) {
  return (
    <code
      className="block max-w-full overflow-x-auto whitespace-nowrap rounded-lg bg-slate-100 px-3 py-2 text-xs text-slate-800 dark:bg-slate-950 dark:text-slate-200"
      title="Defanged URL; not clickable"
    >
      {finding.display_url}
    </code>
  )
}

export function JobStepper({
  currentStep,
  status,
}: {
  currentStep: JobStep | null
  status: JobStatus
}) {
  const activeIndex = currentStep ? JOB_STEPS.indexOf(currentStep) : -1
  return (
    <ol
      aria-label="Analysis progress"
      data-testid="job-stepper"
      className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4"
    >
      {JOB_STEPS.map((step, index) => {
        const complete = activeIndex >= 0 && index < activeIndex
        const active = index === activeIndex
        const failed = status === 'failed' && active
        return (
          <li
            key={step}
            aria-current={active ? 'step' : undefined}
            className={`flex min-h-11 items-center gap-3 rounded-xl border px-3 py-2 text-sm ${
              failed
                ? 'border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200'
                : active
                  ? 'border-teal-500 bg-teal-50 font-semibold text-teal-900 ring-1 ring-teal-500 dark:bg-teal-950/50 dark:text-teal-100'
                  : complete
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-200'
                    : 'border-slate-200 bg-white text-slate-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400'
            }`}
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-current text-xs">
              {complete ? '✓' : index + 1}
            </span>
            <span>{step}</span>
            {active && status !== 'done' && status !== 'failed' && (
              <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-current" />
            )}
          </li>
        )
      })}
    </ol>
  )
}

export function RiskBar({ score }: { score: number | null }) {
  const value = Math.max(0, Math.min(100, score ?? 0))
  const color =
    value >= 60
      ? 'bg-red-500'
      : value >= 25
        ? 'bg-amber-500'
        : 'bg-emerald-500'
  return (
    <div className="flex min-w-32 items-center gap-2">
      <div
        aria-label={`Risk score ${value} out of 100`}
        className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"
      >
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
      <span className="w-12 text-right text-xs font-semibold tabular-nums">
        {score === null ? '—' : `${value}/100`}
      </span>
    </div>
  )
}

export function Panel({
  title,
  eyebrow,
  children,
  className = '',
}: {
  title: string
  eyebrow?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}
    >
      <div className="mb-4">
        {eyebrow && (
          <p className="mb-1 text-[11px] font-bold uppercase tracking-[0.16em] text-teal-700 dark:text-teal-300">
            {eyebrow}
          </p>
        )}
        <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      </div>
      {children}
    </section>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200"
    >
      {message}
    </div>
  )
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div
      role="status"
      className="flex items-center gap-3 py-10 text-sm text-slate-500 dark:text-slate-400"
    >
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
      {label}
    </div>
  )
}
