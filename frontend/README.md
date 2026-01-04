# JARVIS-2v Frontend

Modern Next.js web interface for the JARVIS-2v modular AI system.

## Features

- **Dashboard**: System status, metrics, and quick overview
- **Adapter Graph**: View and manage modular AI adapters
- **Quantum Lab**: Run synthetic quantum experiments
- **Console**: Chat-like interface for interacting with JARVIS
- **Settings**: Configure deployment modes and system settings

## Tech Stack

- **Next.js 14** with App Router
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons

## Quick Start

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at http://localhost:3000

### Production Build

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Integration

The frontend communicates with the FastAPI backend through the `lib/api-client.ts` module. Make sure the backend is running before starting the frontend.

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Dashboard
│   ├── adapters/          # Adapters page
│   ├── quantum/           # Quantum Lab
│   ├── console/           # Chat console
│   ├── settings/          # Settings
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   └── Navigation.tsx     # Main navigation
├── lib/                   # Utilities
│   └── api-client.ts      # API client
├── public/                # Static assets
├── next.config.js         # Next.js configuration
├── tailwind.config.ts     # Tailwind configuration
└── package.json           # Dependencies
```

## Deployment

See the main project README and `docs/DEPLOYMENT.md` for deployment instructions for:

- Vercel
- Netlify
- Docker
- shiper.app

## Development Tips

### Hot Reload

The dev server supports hot module replacement. Changes to code will reflect immediately.

### Type Safety

All API calls are type-safe with TypeScript interfaces defined in `lib/api-client.ts`.

### Styling

Uses Tailwind CSS with custom JARVIS theme colors:
- Primary: `#00d4ff` (jarvis-primary)
- Accent: `#00ffaa` (jarvis-accent)
- Dark: `#0a0f1c` (jarvis-dark)

### Adding New Pages

1. Create a new directory in `app/`
2. Add a `page.tsx` file
3. The route will be automatically available

## Customization

### Theme Colors

Edit `tailwind.config.ts` to customize colors:

```ts
colors: {
  jarvis: {
    primary: "#00d4ff",
    secondary: "#0088cc",
    dark: "#0a0f1c",
    darker: "#060911",
    accent: "#00ffaa",
  },
}
```

### API Endpoint

Change the backend URL in `next.config.js`:

```js
env: {
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
}
```

## License

Part of the JARVIS-2v project
