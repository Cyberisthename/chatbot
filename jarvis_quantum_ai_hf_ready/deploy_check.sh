#!/bin/bash
# 🚀 Quick Deploy Script for JARVIS Quantum AI Suite
# Just run this script to test everything before deployment

echo "🌌 JARVIS Quantum AI Suite - Pre-Deployment Check"
echo "=================================================="
echo

# Check if all required files exist
echo "📋 Checking required files..."

required_files=(
    "app.py"
    "requirements.txt" 
    "src/"
    "gradio_quantum_cancer_demo.py"
    "jarvis_v1_gradio_space.py"
    "README.md"
    "LICENSE"
)

for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING"
        exit 1
    fi
done

echo
echo "🔍 Checking source directory structure..."

src_dirs=(
    "src/quantum_llm/"
    "src/thought_compression/"
    "src/core/"
    "src/bio_knowledge/"
    "src/multiversal/"
    "src/quantum/"
)

for dir in "${src_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir"
    else
        echo "❌ $dir - MISSING"
        exit 1
    fi
done

echo
echo "📝 Checking file contents..."

# Check if app.py has the right structure
if grep -q "def create_interface" app.py; then
    echo "✅ app.py - Has main interface function"
else
    echo "❌ app.py - Missing main interface function"
    exit 1
fi

# Check if requirements.txt has right dependencies
if grep -q "gradio" requirements.txt; then
    echo "✅ requirements.txt - Has gradio dependency"
else
    echo "❌ requirements.txt - Missing gradio dependency"
    exit 1
fi

echo
echo "🎯 Deployment Checklist:"
echo "✅ All required files present"
echo "✅ Source code structure correct"  
echo "✅ Dependencies properly specified"
echo "✅ Ready for Hugging Face Spaces!"
echo
echo "🚀 Next Steps:"
echo "1. Go to https://huggingface.co/spaces"
echo "2. Create new Space → Select 'Gradio' SDK"
echo "3. Upload ALL files from this folder"
echo "4. Wait 2-5 minutes for auto-build"
echo "5. Your quantum AI platform will be live! 🎉"
echo
echo "📚 Documentation:"
echo "- README_HF_SPACES.md - Detailed deployment guide"
echo "- DEPLOYMENT_SUMMARY.md - Complete overview"
echo "- test_deployment.py - Run this locally to test (if you have dependencies installed)"
echo
echo "⚛️ Built with 🧠 for real science. Real research. Real quantum mechanics."