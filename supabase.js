import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL  = 'https://fillsvkpjtvkbgnrvjkq.supabase.co'
const SUPABASE_ANON = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpbGxzdmtwanR2a2JnbnJ2amtxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODAxNjgzNDYsImV4cCI6MjA5NTc0NDM0Nn0.9QtCHni1mZ2woxYIGV4GrBYYWQn--Tlbb4S3BdwQpPU'

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON)
