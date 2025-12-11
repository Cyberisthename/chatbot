# J.A.R.V.I.S. Chatbot - Vercel Deployment Branch

> 🌐 **This is the clean deployment branch** for Vercel/Netlify without Git LFS model files.

## ⚡ Quick Deploy

**Branch**: `deploy/vercel-clean-webapp-no-lfs`

```bash
# 1. Connect your GitHub repo to Vercel at vercel.com
# 2. Select this branch: deploy/vercel-clean-webapp-no-lfs  
# 3. Click Deploy
# 4. ✅ Done!
```

📖 **[See Quick Start Guide →](QUICKSTART_VERCEL.md)**

## 🎯 What's Different in This Branch

This branch is specifically configured for **web deployment** without large model files:

### ✅ What's Included
- Web interface (`index.html`) with Tailwind CSS
- Node.js/Express server (`server.js`)
- Socket.IO for real-time communication
- Demo/mock AI responses (no models needed)
- All necessary configs (`vercel.json`, `package.json`)

### ❌ What's Excluded
- `*.gguf` model files (not needed for demo UI)
- `quantacap/artifacts/` (large experiment files)
- Git LFS tracking (causes deployment issues)
- Python components (backend only)
- Training data and scripts

## 📁 Key Files

| File | Purpose |
|------|---------|
| `index.html` | Main web interface |
| `server.js` | Node.js backend server |
| `jarvis-core.js` | LLM engine (mock mode) |
| `vercel.json` | Vercel configuration |
| `package.json` | Node.js dependencies |
| `.vercelignore` | Files to exclude from deployment |
| `.gitattributes` | LFS disabled |
| `.gitignore` | Updated to exclude models |

## 🚀 Deployment Guides

Choose your speed:

1. **⚡ 5-Minute Quick Start**: [QUICKSTART_VERCEL.md](QUICKSTART_VERCEL.md)
2. **📖 Detailed Guide**: [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)
3. **✅ Step-by-Step Checklist**: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

## 🎨 What You Get

A live web application with:

- **Beautiful UI**: Modern chat interface with Tailwind CSS
- **Real-time Chat**: Socket.IO powered messaging
- **API Endpoints**: RESTful API for integrations
- **Demo Responses**: Mock AI responses (no model files needed)
- **Fast Deployment**: Builds in ~30-60 seconds
- **Free Hosting**: Works on Vercel's free tier

## 🔧 Configuration

### Vercel Settings

```json
{
  "branch": "deploy/vercel-clean-webapp-no-lfs",
  "buildCommand": "npm run build",
  "outputDirectory": "",
  "installCommand": "npm install",
  "framework": "Other"
}
```

### Environment Variables

Vercel automatically sets:
- `NODE_ENV=production`
- `PORT` (assigned by Vercel)

No manual environment variables needed for basic deployment.

## 📊 What Works vs. What Doesn't

### ✅ Fully Functional
- Web UI loads and renders
- Chat interface works
- Message sending/receiving
- API endpoints respond
- Socket.IO connections
- Health checks
- Demo/mock responses

### ⚠️ Mock/Demo Mode
- AI responses are simulated
- No actual LLM inference
- No model loading
- Responses are pre-generated

### 🔌 To Enable Real AI

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) section on:
- Connecting to external inference API
- Running local instance with models
- Hybrid deployment (UI on Vercel + AI backend elsewhere)

## 🔄 Updating Your Deployment

```bash
# 1. Make sure you're on this branch
git checkout deploy/vercel-clean-webapp-no-lfs

# 2. Make your changes
# ... edit files ...

# 3. Test locally (optional)
npm install
npm start
# Visit http://localhost:3001

# 4. Commit and push
git add .
git commit -m "Update web interface"
git push origin deploy/vercel-clean-webapp-no-lfs

# 5. Vercel auto-deploys (if auto-deploy enabled)
```

## 🌲 Branch Strategy

```
main
├── Full system with models
├── Python backend
├── Training scripts
└── All components

deploy/vercel-clean-webapp-no-lfs  ← You are here
├── Web UI only
├── Node.js server
├── No model files
└── Clean deployment
```

## 🐛 Common Issues

### Build Fails - "Pointer file error"
**Cause**: Wrong branch or LFS files present  
**Fix**: Ensure you're on `deploy/vercel-clean-webapp-no-lfs`

### 404 Not Found
**Cause**: Routing misconfiguration  
**Fix**: Check `vercel.json` routes

### Module Not Found
**Cause**: Missing dependencies  
**Fix**: Run `npm install`, commit `package-lock.json`

### Slow Build Times
**Cause**: Too many files being uploaded  
**Fix**: Check `.vercelignore` is properly configured

## 📖 Documentation

- [Main README](README.md) - Full project overview
- [Quick Start Guide](QUICKSTART_VERCEL.md) - Deploy in 5 minutes
- [Detailed Deployment Guide](VERCEL_DEPLOYMENT.md) - Complete instructions
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Step-by-step validation

## 💡 Pro Tips

1. **Enable Auto-Deploy**: In Vercel, set production branch to this branch
2. **Preview Deployments**: Every push gets a preview URL
3. **Custom Domain**: Add your domain in Vercel settings
4. **Environment Variables**: Add any custom variables in Vercel dashboard
5. **Logs**: Check Vercel function logs for debugging

## 📈 Performance

Expected metrics:
- **Build Time**: 30-90 seconds
- **Cold Start**: < 1 second
- **Response Time**: 50-200ms
- **Bundle Size**: ~5MB
- **Memory Usage**: ~100MB

## 🤝 Contributing

To contribute to this deployment branch:

1. Fork the repo
2. Create a feature branch from `deploy/vercel-clean-webapp-no-lfs`
3. Make your changes
4. Test deployment
5. Submit a PR

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙋 Need Help?

1. **Deployment Issues**: See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)
2. **Quick Questions**: Check [QUICKSTART_VERCEL.md](QUICKSTART_VERCEL.md)
3. **Step-by-Step**: Use [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
4. **GitHub Issues**: Open an issue on the repo
5. **Vercel Support**: [vercel.com/support](https://vercel.com/support)

---

**Ready to deploy?** → [Start with the Quick Start Guide](QUICKSTART_VERCEL.md) 🚀
