"""
Demo: The World-Breaking Digital Pipeline

This script demonstrates the complete flow from DNA construction to multiversal cell simulation.
"""

from src.bio_knowledge.digital_pipeline import run_pipeline

if __name__ == "__main__":
    try:
        run_pipeline()
        print("\n✨ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY! ✨")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
