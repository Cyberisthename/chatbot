# JARVIS-2v Deployment Checklist

Use this checklist to ensure a smooth deployment.

---

## 📋 Pre-Deployment Checklist

### Code Quality

- [ ] All core functionality tested locally
- [ ] Backend health check passes
- [ ] Frontend connects to backend successfully
- [ ] All pages load without errors
- [ ] No console errors in browser

### Configuration

- [ ] `config.yaml` configured for target environment
- [ ] Environment variables documented
- [ ] CORS settings updated for production domains
- [ ] API URL configured in frontend

### Security

- [ ] Sensitive data removed from code
- [ ] No API keys or secrets in version control
- [ ] CORS origins restricted (not `*` in production)
- [ ] HTTPS enabled for production

### Documentation

- [ ] README.md updated
- [ ] API documentation reviewed
- [ ] Deployment instructions tested
- [ ] Environment variables documented

---

## 🚀 Local Development Checklist

### First Time Setup

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] Dependencies installed (backend)
- [ ] Dependencies installed (frontend)
- [ ] `config.yaml` exists in project root

### Testing Locally

```bash
# Start both services
./scripts/start_all_local.sh
```

- [ ] Backend starts on port 8000
- [ ] Frontend starts on port 3000
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Dashboard loads at http://localhost:3000
- [ ] Can create an adapter
- [ ] Can run a quantum experiment
- [ ] Console chat works

---

## 🐳 Docker Deployment Checklist

### Build

```bash
docker build -t jarvis-2v:latest .
```

- [ ] Build completes without errors
- [ ] Image size is reasonable (<2GB)
- [ ] No security warnings

### Test

```bash
docker run -d -p 8000:8000 -p 3000:3000 jarvis-2v:latest
```

- [ ] Container starts successfully
- [ ] Health check passes
- [ ] Both services accessible
- [ ] Logs show no errors

### Cleanup

```bash
docker stop $(docker ps -q --filter ancestor=jarvis-2v:latest)
```

---

## ☁️ Vercel Deployment Checklist

### Prerequisites

- [ ] Vercel account created
- [ ] GitHub repo connected to Vercel
- [ ] Backend deployed separately (e.g., Fly.io, Railway)

### Configuration

- [ ] `vercel.json` configured
- [ ] Environment variable set: `NEXT_PUBLIC_API_URL`
- [ ] Build command verified
- [ ] Output directory set to `frontend`

### Deployment

```bash
cd frontend
vercel --prod
```

- [ ] Build succeeds
- [ ] Frontend accessible at Vercel URL
- [ ] Frontend can connect to backend
- [ ] All pages load correctly
- [ ] No CORS errors

### Backend Deployment (Separate)

Options:
- [ ] Fly.io: `flyctl deploy`
- [ ] Railway: `railway up`
- [ ] Render: Connect GitHub repo
- [ ] DigitalOcean: Connect GitHub repo

---

## 🌐 Netlify Deployment Checklist

### Configuration

- [ ] `netlify.toml` configured
- [ ] Base directory set to `frontend`
- [ ] Build command: `npm run build`
- [ ] Publish directory: `frontend/.next`
- [ ] Environment variable set: `NEXT_PUBLIC_API_URL`

### Deployment

```bash
cd frontend
netlify deploy --prod
```

- [ ] Build succeeds
- [ ] Site accessible
- [ ] Backend connection works
- [ ] No CORS issues

---

## 🚢 shiper.app Deployment Checklist

### Image Preparation

```bash
# Build
docker build -t your-username/jarvis-2v:latest .

# Test locally
docker run -p 8000:8000 -p 3000:3000 your-username/jarvis-2v:latest

# Push
docker push your-username/jarvis-2v:latest
```

- [ ] Image built successfully
- [ ] Image tested locally
- [ ] Image pushed to registry

### shiper.app Configuration

- [ ] App created on shiper.app
- [ ] Image URL configured
- [ ] Ports mapped: 8000, 3000
- [ ] Environment variables set
- [ ] Health check configured

### Post-Deployment

- [ ] App accessible via shiper.app URL
- [ ] Both services running
- [ ] Logs show no errors
- [ ] All functionality works

---

## 🔒 Production Checklist

### Security

- [ ] HTTPS enabled
- [ ] CORS configured for specific domains
- [ ] No debug mode in production
- [ ] Secrets stored in environment variables
- [ ] Rate limiting configured (if needed)

### Performance

- [ ] Frontend build optimized
- [ ] Backend using production ASGI server
- [ ] Caching configured
- [ ] CDN configured for static assets (optional)

### Monitoring

- [ ] Health checks configured
- [ ] Uptime monitoring set up
- [ ] Error tracking enabled (e.g., Sentry)
- [ ] Log aggregation configured

### Backup & Recovery

- [ ] Adapter storage backed up
- [ ] Quantum artifacts backed up
- [ ] Configuration backed up
- [ ] Recovery procedure documented

---

## 📊 Post-Deployment Verification

### Functionality Tests

- [ ] Health endpoint: `GET /health` returns 200
- [ ] List adapters: `GET /api/adapters` works
- [ ] Create adapter: `POST /api/adapters` works
- [ ] Run inference: `POST /api/infer` works
- [ ] Run experiment: `POST /api/quantum/experiment` works
- [ ] List artifacts: `GET /api/artifacts` works

### Frontend Tests

- [ ] Dashboard loads and shows correct data
- [ ] Adapters page loads and lists adapters
- [ ] Can create new adapter from UI
- [ ] Quantum Lab runs experiments
- [ ] Console chat interface works
- [ ] Settings page loads and saves changes

### Integration Tests

- [ ] Frontend → Backend communication works
- [ ] CORS headers correct
- [ ] API responses match expected format
- [ ] Error handling works correctly

---

## 🐛 Troubleshooting Checklist

### Backend Issues

If backend won't start:
- [ ] Check Python version (3.10+)
- [ ] Verify dependencies installed
- [ ] Check port 8000 is available
- [ ] Review error logs
- [ ] Verify config.yaml exists

### Frontend Issues

If frontend won't start:
- [ ] Check Node version (18+)
- [ ] Verify dependencies installed
- [ ] Check port 3000 is available
- [ ] Clear `.next` directory
- [ ] Reinstall node_modules

### Connection Issues

If frontend can't reach backend:
- [ ] Verify backend is running
- [ ] Check `NEXT_PUBLIC_API_URL` is correct
- [ ] Verify CORS settings
- [ ] Check firewall rules
- [ ] Test with curl

---

## ✅ Deployment Success Criteria

Your deployment is successful when:

- ✅ Health check returns `{"status": "ok"}`
- ✅ Dashboard shows system metrics
- ✅ Can create adapters via UI
- ✅ Can run quantum experiments
- ✅ Console chat works
- ✅ All pages load without errors
- ✅ No CORS errors in browser console
- ✅ Backend API docs accessible
- ✅ All endpoints respond correctly

---

## 📞 Getting Help

If you encounter issues:

1. Check logs
2. Review documentation
3. Search GitHub issues
4. Create new issue with details

---

## 🎉 Deployment Complete!

Once all checkboxes are checked, your JARVIS-2v deployment is complete!

**Next steps:**
- Monitor system health
- Set up alerts
- Plan for scaling
- Collect user feedback

---

**Good luck with your deployment! 🚀**
