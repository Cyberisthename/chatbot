# 🚀 Hugging Face Deployment Guide

This guide walks you through deploying the JARVIS Quantum AI Suite to Hugging Face Spaces.

---

## ✅ Quick Deployment (5 minutes)

### Step 1: Create a Hugging Face Account

If you don't have one yet:
1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up"
3. Verify your email

### Step 2: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Owner**: Your username
   - **Space name**: `jarvis-quantum-ai` (or your choice)
   - **SDK**: Select **"Gradio"**
   - **Space License**: Choose a license (e.g., MIT)
   - **Public/Private**: Choose **Public** for free hosting
4. Click **"Create Space"**

### Step 3: Upload Your Code

There are two ways to do this:

#### Option A: Using Git (Recommended)

```bash
# Clone your repository
git clone <your-repository-url>
cd <your-repository>

# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/your-username/jarvis-quantum-ai

# Push to Hugging Face
git push hf main
```

#### Option B: Using Web Interface

1. Go to your new Space on Hugging Face
2. Click the **"Files"** tab
3. Click **"Upload files"**
4. Upload these files/folders:
   - `app.py`
   - `requirements.txt`
   - `README_HF.md` (rename to `README.md`)
   - `src/` folder (entire folder)
   - `gradio_quantum_cancer_demo.py`
   - `jarvis_v1_gradio_space.py`
   - Any documentation you want to include

### Step 4: Wait for Automatic Deployment

Hugging Face will automatically:
- Install dependencies from `requirements.txt`
- Build the Gradio interface
- Launch your application

This typically takes 2-5 minutes.

### Step 5: Access Your Space

Your deployed Space will be available at:
```
https://huggingface.co/spaces/your-username/jarvis-quantum-ai
```

---

## 📋 Files to Include

### Essential Files (Required)

```
app.py                          # Main application
requirements.txt                 # Dependencies
README.md                       # Space description
src/                            # Source code
  ├── quantum_llm/
  ├── thought_compression/
  └── ...
gradio_quantum_cancer_demo.py    # Cancer demo
jarvis_v1_gradio_space.py       # Jarvis demo
```

### Optional Files (Nice to Have)

```
README_HF.md                    # Comprehensive docs
README_QUANTUM_LLM.md          # Technical documentation
JARVIS_V1_MISSION_COMPLETE.md  # Jarvis guide
CANCER_HYPOTHESIS_COMPLETE.md  # Cancer research docs
LICENSE                         # License file
```

### Files to Exclude (Don't Upload)

```
.git/                          # Git metadata
.gitignore                     # Git ignore file
.venv/                        # Virtual environment
__pycache__/                  # Python cache
*.pyc                         # Compiled Python files
node_modules/                  # Node modules (if any)
.env                          # Environment variables with secrets
*.log                         # Log files
```

---

## 🔧 Troubleshooting

### Issue: Build Fails

**Symptom**: Your Space shows "Build Error" or "Runtime Error"

**Solutions**:
1. Check the **"Logs"** tab in your Space for error details
2. Verify `requirements.txt` has correct dependencies
3. Make sure `app.py` is in the root directory
4. Ensure all imports in your code are available

### Issue: App Doesn't Load

**Symptom**: Page loads but interface is blank

**Solutions**:
1. Check browser console for JavaScript errors
2. Verify Gradio version compatibility (>= 4.0.0)
3. Try refreshing the page
4. Check that all required files are uploaded

### Issue: Imports Fail

**Symptom**: "ModuleNotFoundError" in logs

**Solutions**:
1. Ensure `src/` folder is uploaded
2. Check that `sys.path.insert(0, str(Path(__file__).parent))` is in `app.py`
3. Verify all dependencies are in `requirements.txt`

### Issue: Out of Memory

**Symptom**: "OutOfMemoryError" in logs

**Solutions**:
1. Demo mode works without model weights (reduced memory)
2. For full model, upgrade to "CPU Upgrade" or GPU space
3. The free tier should work fine for demo mode

---

## 🎯 Best Practices

### 1. Use Git for Deployment

```bash
# Make changes locally
git add .
git commit -m "Update demo interface"

# Push to Hugging Face
git push hf main
```

### 2. Monitor Logs

- Check the **"Logs"** tab regularly
- Watch for warnings or errors
- Deployments show build progress

### 3. Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### 4. Keep Requirements Minimal

Only include what's needed:
```txt
gradio>=4.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### 5. Write Good README.md

Your Space's README should include:
- Title and description
- What your app does
- How to use it
- Example prompts/inputs
- Screenshots (optional)

---

## 📊 Performance Tips

### Free Tier (CPU Basic)
- ✅ Demo mode works perfectly
- ✅ Cancer research demo (no model needed)
- ✅ Jarvis demo mode (no model needed)
- ⚠️ Full Jarvis model (optional upgrade)

### Upgraded Tiers
- 🚀 Faster response times
- 🚀 Can run full trained model
- 🚀 Better for production use

### Optimization
- Reduce `num_universes` parameter in cancer demo
- Use lower resolution plots
- Cache model weights (if using full model)

---

## 🔐 Security

### Never Upload
- API keys
- Passwords
- Secrets
- Personal data

### Environment Variables
```python
import os

api_key = os.environ.get("MY_API_KEY")
```

Set in Hugging Face Space settings (Settings → Variables and secrets)

---

## 📈 Monitoring Your Space

### View Metrics
- **Visits**: Number of users
- **Likes**: Community engagement
- **GPU Usage**: If applicable

### Analytics
- Check user interaction patterns
- See which demos are popular
- Monitor error rates

---

## 🔄 Updating Your Space

### Make Changes Locally
```bash
# Edit files
vim app.py

# Test locally
python app.py

# Commit and push
git add .
git commit -m "Update interface"
git push hf main
```

### Automatic Redeploy
- Hugging Face detects changes
- Automatically rebuilds and deploys
- Takes 2-5 minutes

---

## 📚 Additional Resources

### Hugging Face Documentation
- [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Integration](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Community Spaces](https://huggingface.co/spaces) (for examples)

### Repository Documentation
- [README_HF.md](README_HF.md) - Complete documentation
- [README_QUANTUM_LLM.md](README_QUANTUM_LLM.md) - Quantum LLM details
- [JARVIS_V1_MISSION_COMPLETE.md](JARVIS_V1_MISSION_COMPLETE.md) - Jarvis guide

---

## 🎉 Success Checklist

- [x] Created Hugging Face account
- [x] Created a new Space with Gradio SDK
- [x] Uploaded `app.py` and `requirements.txt`
- [x] Uploaded `src/` folder with all source code
- [x] Uploaded demo files (`gradio_quantum_cancer_demo.py`, etc.)
- [x] Uploaded or updated `README.md`
- [x] Waited for automatic deployment
- [x] Tested the deployed Space
- [x] Shared with community

---

## 🤝 Community

### Share Your Space
- Tweet about it
- Share on Discord/Reddit
- Add to Hugging Face collections
- Write a blog post

### Get Feedback
- Enable comments on your Space
- Ask for user experiences
- Monitor issues and suggestions

---

## 💡 Pro Tips

1. **Start Small**: Deploy basic version first, add features later
2. **Version Control**: Use git for all changes
3. **Test Locally**: Always test before deploying
4. **Monitor Logs**: Check regularly for issues
5. **Iterate Fast**: Deploy often, improve incrementally
6. **Engage Community**: Share and get feedback

---

## 🎓 Learning Resources

- [Gradio Documentation](https://gradio.app/docs)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Python Best Practices](https://docs.python-guide.org/)

---

**Ready to deploy?** Follow the Quick Deployment guide above! 🚀

**Need help?** Check the Troubleshooting section or open an issue.

Good luck with your deployment! 🌌
