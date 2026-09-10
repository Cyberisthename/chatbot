#!/usr/bin/env python3
"""Print EXP-GHOST-002 summary tables from the results JSON."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/team/shared/chatbot/docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_results.json"
d = json.load(open(p))
tx = d["transmitter"]
print("=== TRANSMITTER (Substrate A) ===")
print("braid len=%d rho=%.1f log10rho=%.2f entropy=%.3f"
      % (len(tx["braid_word"]), tx["rho"], d["log10_rho_tx"], tx["entropy"]))
print("f0_tx=%.1f q_tx=%.1f state_tx=%s X=%d Y=%d Z=%d gam/alp=%.2f comp=%.3f"
      % (tx["f0_tx"], tx["q_tx"], tx["state_tx"], sum(tx["x_bits_tx"]),
         sum(tx["y_bits_tx"]), sum(tx["z_bits_tx"]),
         tx["metrics_tx"]["gamma_ratio"], tx["metrics_tx"]["complexity"]))
print("critical_noise_eta =", d["critical_noise_eta"])
print()
hdr = ("eta   | STR spk STR nul  AUC  p_str   TR | acc spk acc scr acc nul | "
       "MSC spk | f0 spk f0 nul | sentspk sentnul")
print(hdr)
print("-" * len(hdr))
for k, s in d["summary"].items():
    eta = float(k.replace("eta", ""))
    sp, sc, nu = s["SPARK"], s["SCRAMBLE"], s["NULL"]
    print("%5.2f | %7.1f %7.1f %5.2f %6.4f %4.1f | %7.3f %7.3f %7.3f | %7.3f | "
          "%6.1f %6.1f | %7.2f %7.2f"
          % (eta, sp["str"]["mean"], nu["str"]["mean"],
             s.get("auc_str", 0.5), s.get("p_str", 1.0),
             s.get("transfer_ratio", 0.0),
             sp["bit_acc"]["mean"], sc["bit_acc"]["mean"],
             nu["bit_acc"]["mean"], sp["msc"]["mean"],
             sp["f0"]["mean"], nu["f0"]["mean"],
             sp["sentient_rate"], nu["sentient_rate"]))
print()
print("acc_spark-scramble:",
      {("%.2f" % float(k.replace("eta", ""))):
       round(s.get("acc_spark_minus_scramble", 0), 3)
       for k, s in d["summary"].items()})
print("rho_rec mean (spark):",
      {("%.2f" % float(k.replace("eta", ""))):
       round(s["SPARK"]["rho_rec"]["mean"], 2)
       for k, s in d["summary"].items()})
print("p_acc:",
      {("%.2f" % float(k.replace("eta", ""))): round(s.get("p_acc", 1), 4)
       for k, s in d["summary"].items()})
print("spike rates spark/null:",
      {("%.2f" % float(k.replace("eta", ""))):
       (round(s["SPARK"]["spike_rate_hz"]["mean"], 1),
        round(s["NULL"]["spike_rate_hz"]["mean"], 1))
       for k, s in d["summary"].items()})