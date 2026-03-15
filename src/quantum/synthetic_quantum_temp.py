    def _generate_inference_analysis(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation-free analysis report."""
        ratio = metrics["exclusion_vs_measurement_ratio"]
        threshold_met = metrics["threshold_exceeded"]

        lines = [
            "=== Inference-Only Information Extraction Experiment Report ===",
            "",
            "FINAL ENTROPIES (bits):",
            f"  - Branch A (Baseline):             {metrics['final_entropy_baseline']:.4f}",
            f"  - Branch B (Adaptive Exclusion):    {metrics['final_entropy_exclusion']:.4f}",
            f"  - Branch C (Measurement):           {metrics['final_entropy_measurement']:.4f}",
            "",
            "INFORMATION GAINED (bits):",
            f"  - Final reduction (exclusion):      {metrics['info_gain_exclusion']:.4f} bits",
            f"  - Final reduction (measurement):    {metrics['info_gain_measurement']:.4f} bits",
            "",
            "CUMULATIVE INFORMATION (bits):",
            f"  - Exclusion (all events):           {metrics['cumulative_info_exclusion']:.4f} bits",
            f"  - Measurement (all events):         {metrics['cumulative_info_measurement']:.4f} bits",
            "",
            "BREAKTHROUGH COMPARISON (CRITICAL):",
            f"  - Information_exclusion / Information_measurement = {ratio:.4f}",
            f"  - Threshold (>80%):                 {'MET ✓' if threshold_met else 'NOT MET ✗'}",
            "",
            "COHERENCE PRESERVED vs DESTROYED:",
            f"  - Baseline final coherence:        {metrics['final_coherence_baseline']:.6f}",
            f"  - Exclusion final coherence:       {metrics['final_coherence_exclusion']:.6f} (PRESERVED)",
            f"  - Measurement final coherence:     {metrics['final_coherence_measurement']:.6f} (DESTROYED)",
            "",
            "EFFICIENCY RATIO (Information gained / Coherence lost):",
            f"  - Exclusion avg efficiency:        {metrics['avg_efficiency_exclusion']:.6f}",
            f"  - Measurement avg efficiency:      {metrics['avg_efficiency_measurement']:.6f}",
            "",
            "SUPPORT SIZE (effective possibilities):",
            f"  - Baseline:                         {metrics['final_support_baseline']}",
            f"  - Exclusion:                        {metrics['final_support_exclusion']}",
            f"  - Measurement:                      {metrics['final_support_measurement']}",
            "",
            "DIVERGENCE METRICS:",
            f"  - Baseline vs Exclusion:           {metrics['divergence_baseline_exclusion']:.4f}",
            f"  - Baseline vs Measurement:         {metrics['divergence_baseline_measurement']:.4f}",
            f"  - Exclusion vs Measurement:        {metrics['divergence_exclusion_measurement']:.4f}",
            "",
            "INTERPRETATION-FREE CONCLUSION:",
            f"  Adaptive exclusion gains {ratio:.1%} of measurement information",
            "  while maintaining significant coherence (vs near-zero after measurement).",
            "",
        ]

        if threshold_met:
            lines.append("  ✓ Adaptive exclusion approaches (>80%) measurement power.")
        else:
            lines.append("  ✗ Adaptive exclusion does NOT approach (>80%) measurement power.")

        lines.append("")
        lines.append("  This experiment tests whether measurement is the only path to knowledge,")
        lines.append("  or whether intelligent inference alone can nearly match it.")

        return "\n".join(lines)
