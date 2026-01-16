# 🚀 Hugging Face Spaces - Quick Start Card

## 5-Minute Deployment Checklist ✅

### 1. Create Space (1 min)
- [ ] Go to [huggingface.co/spaces](https://huggingface.co/spaces)
- [ ] Click "Create new Space"
- [ ] Name: `jarvis-quantum-ai`
- [ ] SDK: **Gradio**
- [ ] Click "Create Space"

### 2. Upload Files (2-3 min)

#### Option A: Git (Fastest)
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/jarvis-quantum-ai
git push hf main
```

#### Option B: Web UI (No Git)
1. Go to your Space → Files tab
2. Click "Upload files"
3. Upload these:
   - `app.py`
   - `requirements.txt`
   - `README_HF.md` (rename to `README.md`)
   - `src/` folder (drag & drop)
   - `gradio_quantum_cancer_demo.py`
   - `jarvis_v1_gradio_space.py`

### 3. Wait for Deploy (1-2 min)
- [ ] Watch build progress in "Logs" tab
- [ ] Wait for "Running" status
- [ ] Test the deployed Space!

### 4. Access Your Space 🎉
```
https://huggingface.co/spaces/YOUR_USERNAME/jarvis-quantum-ai
```

---

## 📋 Required Files (Minimum)

```
app.py                          # ✅ Main app
requirements.txt                 # ✅ Dependencies
README.md                       # ✅ Description
src/                            # ✅ Source code
  ├── quantum_llm/
  └── thought_compression/
gradio_quantum_cancer_demo.py    # ✅ Cancer demo
jarvis_v1_gradio_space.py       # ✅ Jarvis demo
```

---

## 🚨 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Build fails | Check "Logs" tab |
| Import error | Ensure `src/` is uploaded |
| Blank page | Refresh browser |
| Out of memory | Use demo mode (no weights) |

---

## 💡 Pro Tips

1. **Test locally first**: `python app.py`
2. **Use git for updates**: `git push hf main`
3. **Monitor logs**: Check "Logs" tab regularly
4. **Start simple**: Deploy minimal version first

---

## 📞 Need Help?

- [Full Guide](DEPLOYMENT_GUIDE.md)
- [Documentation](README_HF.md)
- [Hugging Face Docs](https://huggingface.co/docs/hub/spaces)

---

**That's it! You're ready to deploy! 🚀**
