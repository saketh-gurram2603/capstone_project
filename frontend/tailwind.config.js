/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        mid: {
          primary:   '#0D1117',
          secondary: '#161B22',
          surface:   '#1E2530',
          card:      '#242D3B',
          border:    'rgba(255,255,255,0.07)',
        },
        ops: {
          blue:     '#4F8CFF',
          teal:     '#23C6A8',
          warn:     '#F4B740',
          critical: '#F05A5A',
          purple:   '#9B7FEA',
        },
        txt: {
          primary:   '#F5F7FA',
          secondary: '#9CA7B3',
          muted:     '#4B5563',
        },
      },
      borderRadius: {
        card:  '16px',
        panel: '20px',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      boxShadow: {
        glow:      '0 0 24px rgba(79,140,255,0.18)',
        'glow-teal': '0 0 20px rgba(35,198,168,0.15)',
        card:      '0 4px 24px rgba(0,0,0,0.4)',
        panel:     '0 8px 40px rgba(0,0,0,0.5)',
      },
      backgroundImage: {
        'grid-pattern':
          'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '32px 32px',
      },
    },
  },
  plugins: [],
}
