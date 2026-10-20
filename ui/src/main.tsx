import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Self-hosted, Cyrillic included: the page never reaches for a font CDN.
import '@fontsource/golos-text/400.css'
import '@fontsource/golos-text/500.css'
import '@fontsource/golos-text/600.css'
import '@fontsource/fira-sans-condensed/500.css'
import '@fontsource/fira-sans-condensed/600.css'
import './styles.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
