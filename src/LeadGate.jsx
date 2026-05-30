import { useState } from 'react'
import { supabase } from './supabase.js'
import sparkqbLogo from './assets/sparkqb-logo.svg'

const ROLES = [
  { id: 'athlete_parent', label: 'Athlete / Parent' },
  { id: 'coach',          label: 'Coach' },
]

const STORAGE_KEY = 'sparkqb_lead_submitted'

export function isLeadSubmitted() {
  return !!localStorage.getItem(STORAGE_KEY)
}

export default function LeadGate({ onUnlock }) {
  const [email,     setEmail]     = useState('')
  const [role,      setRole]      = useState('athlete_parent')
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    if (!email.trim() || !/\S+@\S+\.\S+/.test(email)) {
      setError('Please enter a valid email address.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const { error: sbError } = await supabase
        .from('leads')
        .insert([{ email: email.trim().toLowerCase(), role }])

      if (sbError && sbError.code !== '23505') {
        // 23505 = duplicate — still let them in
        throw sbError
      }

      localStorage.setItem(STORAGE_KEY, email.trim().toLowerCase())
      onUnlock()
    } catch (err) {
      console.error('Lead gate error:', err)
      setError('Something went wrong. Please try again.')
      setLoading(false)
    }
  }

  return (
    <div className="gate-overlay">
      <div className="gate-modal">
        <img src={sparkqbLogo} alt="SparkQB" className="gate-logo" />
        <h1 className="gate-title">Film Tool</h1>
        <p className="gate-sub">Enter your email to get started. Free to use.</p>

        <form onSubmit={handleSubmit} className="gate-form">
          <input
            className="gate-input"
            type="email"
            placeholder="your@email.com"
            value={email}
            onChange={e => setEmail(e.target.value)}
            autoComplete="email"
            autoFocus
            required
          />

          <div className="gate-roles">
            {ROLES.map(r => (
              <button
                key={r.id}
                type="button"
                className={`gate-role-btn ${role === r.id ? 'active' : ''}`}
                onClick={() => setRole(r.id)}
              >
                {r.label}
              </button>
            ))}
          </div>

          {error && <p className="gate-error">{error}</p>}

          <button type="submit" className="gate-submit" disabled={loading}>
            {loading ? 'Loading…' : 'Get Started →'}
          </button>
        </form>

        <p className="gate-fine">No password needed. No spam.</p>
      </div>
    </div>
  )
}
