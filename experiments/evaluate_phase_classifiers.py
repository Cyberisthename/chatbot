#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis5090x import (
    CentroidPhaseClassifier,
    MLPPhaseClassifier,
    PhaseDataset,
    SimplePhaseClassifier,
)


def evaluate_classifiers(dataset_path: str, train_ratio: float = 0.8) -> None:
    print(f"Loading dataset from {dataset_path}...")
    dataset = PhaseDataset.load_json(dataset_path)
    print(f"Loaded {len(dataset)} examples")
    
    train_dataset, test_dataset = dataset.split(train_ratio)
    print(f"Split into {len(train_dataset)} training and {len(test_dataset)} test examples")
    
    classifiers = [
        ("Simple k-NN (k=5)", SimplePhaseClassifier(k=5)),
        ("Centroid Classifier", CentroidPhaseClassifier()),
    ]

    mlp_classifier = MLPPhaseClassifier()
    if mlp_classifier.is_available():
        classifiers.append(("MLP Neural Net", mlp_classifier))
    else:
        print("⚠ PyTorch not available - skipping MLP neural net classifier")
    
    results = []
    
    for name, classifier in classifiers:
        print(f"\n=== Training {name} ===")
        train_report = classifier.train(train_dataset)
        print(f"Training complete: {train_report}")
        
        print(f"Evaluating {name}...")
        eval_results = classifier.evaluate(test_dataset)
        results.append((name, eval_results))
        
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Mean Confidence: {eval_results['mean_confidence']:.4f}")
    
    print("\n=== Comparison Summary ===")
    print(f"{'Classifier':<25} {'Accuracy':<12} {'Mean Confidence':<18} {'Correct/Total'}")
    print("=" * 80)
    
    for name, eval_results in results:
        accuracy = eval_results["accuracy"]
        mean_conf = eval_results["mean_confidence"]
        correct = eval_results["correct"]
        total = eval_results["total"]
        print(f"{name:<25} {accuracy:<12.4f} {mean_conf:<18.4f} {correct}/{total}")
    
    print("\n=== Confusion Matrices ===")
    for name, eval_results in results:
        print(f"\n{name}:")
        confusion = eval_results["confusion_matrix"]
        if not confusion:
            print("  No confusion matrix available")
            continue
        
        labels = sorted(confusion.keys())
        print(f"{'True \\ Pred':<25}", end="")
        for label in labels:
            print(f"{label[:10]:<12}", end="")
        print()
        
        for true_label in labels:
            print(f"{true_label[:24]:<25}", end="")
            for pred_label in labels:
                count = confusion.get(true_label, {}).get(pred_label, 0)
                print(f"{count:<12}", end="")
            print()


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate phase classifiers")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/phase_dataset.json",
        help="Path to the phase dataset JSON file",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (default: 0.8)",
    )
    
    args = parser.parse_args()
    
    evaluate_classifiers(args.dataset, args.train_ratio)


if __name__ == "__main__":
    main()
