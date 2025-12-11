# ✅ Vercel Deployment Checklist

Use this checklist to ensure your J.A.R.V.I.S. chatbot deploys successfully to Vercel.

## Pre-Deployment

- [ ] You're on the `deploy/vercel-clean-webapp-no-lfs` branch
  ```bash
  git branch --show-current
  # Should output: deploy/vercel-clean-webapp-no-lfs
  ```

- [ ] All changes are committed and pushed
  ```bash
  git status
  # Should be clean
  ```

- [ ] Required files are present:
  - [ ] `index.html` (web interface)
  - [ ] `server.js` (Node.js backend)
  - [ ] `package.json` (dependencies)
  - [ ] `vercel.json` (Vercel config)
  - [ ] `jarvis-core.js` (LLM engine)

- [ ] Large files are excluded:
  - [ ] No `*.gguf` files in git index
  - [ ] No `quantacap/artifacts/` in git index
  - [ ] `.gitattributes` has LFS disabled
  - [ ] `.gitignore` excludes model files

## Vercel Setup

- [ ] Create Vercel account at [vercel.com](https://vercel.com)

- [ ] Connect GitHub account to Vercel

- [ ] Import repository: `Cyberisthename/chatbot`

- [ ] Select branch: `deploy/vercel-clean-webapp-no-lfs`

- [ ] Configure project:
  - [ ] Framework: Other (or auto-detect)
  - [ ] Root Directory: `./`
  - [ ] Build Command: `npm run build` (or default)
  - [ ] Output Directory: (leave empty)

## Deployment

- [ ] Click "Deploy" in Vercel dashboard

- [ ] Wait for build to complete (~30-90 seconds)

- [ ] Build should succeed without errors

- [ ] No Git LFS errors in logs

- [ ] No missing file errors

## Post-Deployment Testing

- [ ] Visit deployment URL: `https://your-project.vercel.app`

- [ ] Web interface loads correctly

- [ ] Chat UI is visible and styled properly

- [ ] Test API endpoints:
  - [ ] `/api/health` returns JSON
  - [ ] `/api/status` returns system status
  - [ ] `/api/model` returns model info

- [ ] Try sending a message in chat:
  - [ ] Input field works
  - [ ] Send button works
  - [ ] Response appears (even if mock/demo)

## Troubleshooting

If deployment fails, check:

- [ ] Vercel build logs for errors
- [ ] You're on the correct branch
- [ ] No LFS files in the commit
- [ ] `package.json` has all dependencies
- [ ] `vercel.json` is valid JSON

## Common Issues

### "Pointer file error"
- **Cause**: Wrong branch or LFS files present
- **Fix**: Ensure you're on `deploy/vercel-clean-webapp-no-lfs`

### "Module not found"
- **Cause**: Missing dependencies
- **Fix**: Run `npm install` locally, commit `package-lock.json`

### "404 Not Found"
- **Cause**: Routing issue
- **Fix**: Check `vercel.json` routes configuration

### Build times out
- **Cause**: Too many files or large files
- **Fix**: Check `.vercelignore` is excluding unnecessary files

## Success Criteria

Your deployment is successful when:

- ✅ Build completes in < 2 minutes
- ✅ No LFS errors in logs
- ✅ Web interface loads at Vercel URL
- ✅ Chat UI is functional
- ✅ API endpoints respond
- ✅ No console errors in browser

## Optional Enhancements

- [ ] Add custom domain in Vercel settings

- [ ] Enable automatic deployments on push

- [ ] Set up preview deployments for PRs

- [ ] Configure environment variables (if needed)

- [ ] Add monitoring/analytics

## Branch Management

**For future updates:**

```bash
# Always work on the deployment branch
git checkout deploy/vercel-clean-webapp-no-lfs

# Make changes
# ... edit files ...

# Commit and push
git add .
git commit -m "Update web interface"
git push origin deploy/vercel-clean-webapp-no-lfs

# Vercel will auto-deploy
```

## Getting Help

If stuck:

1. Check [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for detailed guide
2. Review Vercel build logs
3. Check GitHub repo structure
4. Open issue on GitHub

---

**Last Updated**: When you created this branch  
**Vercel Docs**: https://vercel.com/docs  
**Support**: https://vercel.com/support
