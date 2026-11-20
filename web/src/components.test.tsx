import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DefangedLink, JobStepper, VerdictBadge } from './components'

describe('VerdictBadge', () => {
  it('uses the phishing danger color', () => {
    render(<VerdictBadge verdict="PHISHING" />)
    expect(screen.getByTestId('verdict-badge')).toHaveAttribute('data-tone', 'red')
    expect(screen.getByTestId('verdict-badge').className).toContain('bg-red-50')
  })

  it('uses the suspicious warning color', () => {
    render(<VerdictBadge verdict="SUSPICIOUS" />)
    expect(screen.getByTestId('verdict-badge')).toHaveAttribute('data-tone', 'amber')
  })

  it('uses the legitimate success color', () => {
    render(<VerdictBadge verdict="LEGIT" />)
    expect(screen.getByTestId('verdict-badge')).toHaveAttribute('data-tone', 'green')
  })
})

describe('DefangedLink', () => {
  it('shows only the defanged URL as plain non-clickable text', () => {
    render(
      <DefangedLink
        finding={{
          url: 'http://192.0.2.44/path',
          display_url: 'hxxp://192[.]0[.]2[.]44/path',
          risk_score: 60,
          reasons: [],
        }}
      />,
    )
    expect(screen.getByText('hxxp://192[.]0[.]2[.]44/path')).toBeInTheDocument()
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByText('http://192.0.2.44/path')).not.toBeInTheDocument()
  })
})

describe('JobStepper', () => {
  it('marks completed and active steps while a job runs', () => {
    render(<JobStepper currentStep="Debate" status="running" />)
    expect(screen.getByText('Extract').closest('li')).toHaveTextContent('✓')
    expect(screen.getByText('Check').closest('li')).toHaveTextContent('✓')
    expect(screen.getByText('Debate').closest('li')).toHaveAttribute('aria-current', 'step')
    expect(screen.getByText('Policy').closest('li')).not.toHaveAttribute('aria-current')
  })

  it('marks the approval wait step when a job is done', () => {
    render(<JobStepper currentStep="Awaiting approval" status="done" />)
    expect(screen.getByText('Awaiting approval').closest('li')).toHaveAttribute('aria-current', 'step')
    expect(screen.getByText('Chart').closest('li')).toHaveTextContent('✓')
  })
})
