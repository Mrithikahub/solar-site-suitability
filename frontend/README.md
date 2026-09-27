# Solarsite frontend

React 19 + Vite + TypeScript + Tailwind CSS v4, with react-leaflet maps, Recharts charts and Motion animations.

```bash
npm install
cp .env.example .env.development   # VITE_API_URL=http://localhost:8000
npm run dev                         # http://localhost:5173
npm run build                       # production build in dist/
```

Pages: `/` landing, `/map` map dashboard, `/compare`, `/methodology` (includes the high-resolution case study; `/tamil-nadu` redirects there).
Static figures, imagery and metrics in `public/` are refreshed with `python scripts/sync_frontend_assets.py`
from the repository root. Deployment: see `../docs/DEPLOYMENT.md` (Vercel, root directory `frontend`).
