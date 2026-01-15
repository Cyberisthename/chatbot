# 🚀 Deploy to Hugging Face Spaces - Step by Step

This guide walks you through deploying the **Quantum Time-Entangled Cancer Cure Demo** to Hugging Face Spaces.

## 📋 Prerequisites

- Hugging Face account (free at [huggingface.co](https://huggingface.co))
- Git installed
- Python 3.8+

## 🎯 Quick Deploy (5 Minutes)

### Step 1: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Name**: `quantum-cancer-cure-demo` (or your choice)
   - **License**: MIT
   - **SDK**: Gradio
   - **Visibility**: **Public** (no bullshit, make it look pro)
   - **Hardware**: CPU Basic (free tier is fine)

### Step 2: Clone the Space Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/quantum-cancer-cure-demo
cd quantum-cancer-cure-demo
```

### Step 3: Upload Files

Create these files in your Space directory:

```bash
# 1. Main app file
cp /path/to/project/app.py .

# 2. Demo code
cp /path/to/project/gradio_quantum_cancer_demo.py .

# 3. Requirements
cat > requirements.txt << 'EOF'
gradio>=4.0.0
numpy>=1.24.0
matplotlib>=3.7.0
EOF

# 4. README (copy from project)
cp /path/to/project/README_SPACES.md README.md
```

### Step 4: Push to Hugging Face

```bash
git add .
git commit -m "Deploy quantum time-entangled cancer cure demo"
git push
```

Hugging Face will automatically build and deploy your Space!

### Step 5: Access Your Demo

Wait 2-5 minutes for deployment, then access:
```
https://huggingface.co/spaces/YOUR_USERNAME/quantum-cancer-cure-demo
```

## 🎨 Customizing the Demo

### Add More Genes/Mutations

Edit `gradio_quantum_cancer_demo.py` and add to `GENE_MUTATION_DB`:

```python
GENE_MUTATION_DB = {
    "PIK3CA": {
        "H1047R": GeneMutation("PIK3CA", "H1047R", "...", "...", True, 0.72, 0.85),
        # Add more mutations...
    },
    # Add more genes...
}
```

### Change the Number of Universes

In `run_experiment()` function, change `num_universes=50`:

```python
experiment = engine.run_post_selection_experiment(
    gene=gene,
    mutation=mutation,
    coercion_strength=coercion_strength,
    num_universes=100  # More universes = more accurate but slower
)
```

### Adjust Simulation Speed

Remove the `time.sleep()` calls in the `run_experiment()` function for faster execution:

```python
def run_experiment(gene: str, mutation: str, coercion_strength: float, progress=gr.Progress()):
    # Remove these lines for speed:
    # time.sleep(0.1)
    # time.sleep(0.2)
    ...
```

## 🔧 Troubleshooting

### Build Fails

1. Check requirements.txt has correct versions
2. Try `pip install -r requirements.txt` locally first
3. Check Space logs for specific errors

### App Runs But Shows Errors

1. Check Space logs: "Logs" tab in your Space
2. Common issues:
   - Missing dependencies
   - Import errors
   - Syntax errors in code

### Visualizations Don't Show

1. Ensure matplotlib is using non-interactive backend:
   ```python
   matplotlib.use('Agg')
   ```
2. Check for memory issues (reduce `num_universes`)

## 📊 Performance Optimization

### Faster Experiments (Free Tier)

```python
# In gradio_quantum_cancer_demo.py
num_universes=20  # Reduce from 50
```

### More Accurate Results (Paid Tier)

```python
# In gradio_quantum_cancer_demo.py
num_universes=100  # Increase from 50
```

### Memory Optimization

If you hit memory limits:
1. Reduce `num_universes`
2. Reduce plot sizes: `figsize=(8, 4)` instead of `(10, 6)`

## 🔐 Privacy & Safety

### Important Disclaimers Already Included

The demo includes these disclaimers:
- "SCIENTIFIC DISCLAIMER: All biology is real. All physics is real."
- "This is a computational demonstration of time-entangled quantum effects."
- "Hypotheses require experimental validation. Not for clinical use."

### Additional Safety Notes

1. **This is a scientific demonstration**, not medical advice
2. **Hypotheses require validation** through wet-lab experiments
3. **Results are computational predictions**, not guaranteed treatments
4. **Share responsibly** with appropriate disclaimers

## 📈 Analytics & Usage

### Track Usage

Hugging Face provides built-in analytics:
- Visit your Space dashboard
- View visitor stats, CPU usage, uptime

### Custom Analytics

Add Google Analytics or similar:

```html
<!-- In gradio_quantum_cancer_demo.html header -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
```

## 🎓 Educational Use Cases

This demo is perfect for:

- **University courses**: Quantum computing, bioinformatics
- **Science communication**: Making quantum effects accessible
- **Public engagement**: Interactive science demos
- **Research presentations**: Visualizing complex concepts

## 🌟 Going Public

### Share Your Demo

Once deployed:

1. **Twitter**: "Check out our quantum time-entangled cancer cure demo! 🔬🌌"
2. **Reddit**: r/QuantumComputing, r/bioinformatics
3. **Academic**: Share on ResearchGate, academic mailing lists
4. **Hugging Face**: Request feature in "Trending Spaces"

### Get Featured

1. Add comprehensive README.md
2. Add screenshots/gifs
3. Use clear, scientific language
4. Submit for "Community Spotlight"

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Docs](https://gradio.app/docs)
- [Quantum Biology Papers](https://arxiv.org/list/quant-ph/recent)

## 🎯 Success Metrics

Your demo is successful when:

- ✅ Deployed and accessible publicly
- ✅ Runs without errors
- ✅ Visualizations render correctly
- ✅ All gene/mutation combinations work
- ✅ Coercion slider produces expected behavior
- ✅ Performance is acceptable (< 10s per experiment)

---

**🚀 Deploy now and make the internet crash with quantum science!**
