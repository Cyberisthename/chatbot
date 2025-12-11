# 🚀 Vercel Deployment Guide for J.A.R.V.I.S. Chatbot

This guide will help you deploy the J.A.R.V.I.S. web interface to Vercel without any LFS or model file issues.

## 📋 Overview

This deployment branch (`deploy/vercel-clean-webapp-no-lfs`) has been specially configured to deploy cleanly on Vercel:

- ✅ **No Git LFS dependencies** - Model files are excluded from deployment
- ✅ **Lightweight build** - Only web UI and API server are deployed
- ✅ **Mock inference mode** - Works without actual model files
- ✅ **Zero-config deployment** - Vercel auto-detects Node.js project

## 🎯 What Gets Deployed

- **Web Interface**: Modern Tailwind CSS chat UI (`index.html`)
- **API Server**: Node.js/Express backend with Socket.IO (`server.js`)
- **Mock AI Responses**: Demo mode that works without model files
- **Health Check Endpoints**: `/api/health`, `/api/status`

## 📦 What's Excluded

The following large files are **NOT** included in deployment (added to `.gitignore`):

- ❌ `*.gguf` - Model files (jarvis-13b-q4_0.gguf, jarvis-34b-q4_0.gguf, etc.)
- ❌ `quantacap/artifacts/` - Large experiment bundles
- ❌ `models/` - Model directory
- ❌ `releases/` - Binary releases
- ❌ Python-specific components (for local development only)

## 🔧 Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Go to Vercel Dashboard**
   - Visit [vercel.com](https://vercel.com)
   - Click "New Project"

2. **Import Your GitHub Repository**
   - Select "Import Git Repository"
   - Choose your repository: `Cyberisthename/chatbot`
   - Click "Import"

3. **Configure Project Settings**
   ```
   Framework Preset: Other (or leave as auto-detected)
   Root Directory: ./
   Build Command: npm run build (or leave default)
   Output Directory: (leave empty - we serve from root)
   Install Command: npm install
   ```

4. **Select Branch**
   - Branch: `deploy/vercel-clean-webapp-no-lfs`
   - ⚠️ **IMPORTANT**: Make sure you select this branch, not `main` or `master`

5. **Environment Variables** (Optional)
   - `NODE_ENV` = `production` (automatically set by Vercel)
   - `PORT` = (automatically set by Vercel)

6. **Deploy**
   - Click "Deploy"
   - Wait 1-2 minutes for deployment
   - Your site will be live at `https://your-project.vercel.app`

### Option 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from the correct branch
git checkout deploy/vercel-clean-webapp-no-lfs

# Deploy to production
vercel --prod
```

## 🔍 Verifying Your Deployment

Once deployed, test these endpoints:

1. **Home Page**: `https://your-project.vercel.app/`
   - Should show the J.A.R.V.I.S. chat interface

2. **Health Check**: `https://your-project.vercel.app/api/health`
   - Should return JSON with status "healthy"

3. **System Status**: `https://your-project.vercel.app/api/status`
   - Shows system operational status

4. **Model Info**: `https://your-project.vercel.app/api/model`
   - Shows model configuration (will indicate demo mode)

## 🎨 Expected Behavior

### ✅ What Works

- Beautiful web interface with Tailwind CSS
- Real-time chat UI
- API endpoints for health checks and status
- Socket.IO connections
- Mock/demo AI responses (simulated)

### ⚠️ What's Limited

- **No Real AI Inference**: Responses are simulated/mocked
- **No Model Loading**: GGUF models are not present
- **Demo Mode Only**: For actual AI inference, you need to:
  - Run locally with models, OR
  - Connect to a separate inference backend

## 🔌 Connecting to a Real AI Backend

To enable real AI responses, you have two options:

### Option A: Connect to External Inference API

Update `index.html` to point to your inference backend:

```javascript
async callJarvisAPI(message) {
  const response = await fetch('YOUR_INFERENCE_API_URL/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  return await response.json();
}
```

### Option B: Use Local Development Setup

For full AI capabilities with models:

```bash
# Clone the main branch
git checkout main

# Install dependencies including Python
npm install
pip install -r requirements.txt

# Place model files in models/ directory
# Run locally
npm start
```

## 🐛 Troubleshooting

### Build Fails with LFS Errors

**Problem**: `Pointer file error: Unable to parse pointer`

**Solution**: Make sure you're deploying the `deploy/vercel-clean-webapp-no-lfs` branch:
```bash
git checkout deploy/vercel-clean-webapp-no-lfs
git push origin deploy/vercel-clean-webapp-no-lfs
```

### "Module not found" Errors

**Problem**: Missing dependencies during build

**Solution**: Ensure `package.json` has all required dependencies:
```bash
npm install
git add package-lock.json
git commit -m "Update dependencies"
git push
```

### Site Shows 404 or Blank Page

**Problem**: Static files not being served correctly

**Solution**: Check that:
- `index.html` is in the root directory
- `vercel.json` routes are configured correctly
- No build errors in Vercel logs

### API Endpoints Return 503

**Problem**: Server initialization issues

**Solution**: This is normal if model files are missing. The web UI should still load. Check Vercel function logs for details.

## 📊 Deployment Configuration Files

### `vercel.json`
```json
{
  "version": 2,
  "builds": [{ "src": "server.js", "use": "@vercel/node" }],
  "routes": [
    { "src": "/api/(.*)", "dest": "/server.js" },
    { "src": "/(.*)", "dest": "/server.js" }
  ]
}
```

### `.gitattributes`
```
# LFS disabled for this deployment branch
```

### `.gitignore` (Additions)
```
# Model files excluded from deployment
*.gguf
*.bin
quantacap/artifacts/
models/
```

## 🔄 Updating Your Deployment

When you make changes:

```bash
# Make sure you're on the deployment branch
git checkout deploy/vercel-clean-webapp-no-lfs

# Make your changes
# ... edit files ...

# Commit and push
git add .
git commit -m "Update web interface"
git push origin deploy/vercel-clean-webapp-no-lfs
```

Vercel will automatically redeploy when you push to the configured branch.

## 🌐 Custom Domain

To add a custom domain:

1. Go to your Vercel project dashboard
2. Click "Settings" → "Domains"
3. Add your domain (e.g., `jarvis.yourdomain.com`)
4. Update your DNS records as instructed

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Node.js Support](https://vercel.com/docs/runtimes#official-runtimes/node-js)
- [Troubleshooting Vercel Builds](https://vercel.com/support)

## 💡 Tips

1. **Fast Builds**: This branch builds in ~30-60 seconds (no heavy model files)
2. **Free Tier**: Works perfectly on Vercel's free tier
3. **Auto-Deploy**: Enable auto-deploy in Vercel for this branch
4. **Preview Deployments**: Every push gets a preview URL
5. **Logs**: Check Vercel function logs if API endpoints have issues

## ✨ Next Steps

1. Deploy the web UI using this branch
2. Test the interface at your Vercel URL
3. (Optional) Set up a separate AI inference backend
4. (Optional) Connect the UI to your inference backend
5. (Optional) Add custom domain

---

**Need Help?** Check the Vercel logs or open an issue on GitHub.

**Success Criteria**: You should be able to:
- ✅ Visit your Vercel URL and see the J.A.R.V.I.S. interface
- ✅ Send messages (get demo/mock responses)
- ✅ No build errors or LFS issues
- ✅ Fast deployment times (< 2 minutes)
