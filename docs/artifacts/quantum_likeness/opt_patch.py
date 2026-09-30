src = open('/home/team/shared/quantum_likeness/qudit_lhv_probe.py').read()

# Insert a Lie-algebra coordinate-ascent optimizer after optimize_cglmp
anchor = "def optimize_cglmp(rho, d, trials=80):"
assert anchor in src
addition = '''def su_generators(d):
    """Anti-Hermitian SU(d) generators (generalized Gell-Mann): pairs with i, diag with i."""
    gens = []
    for j in range(d):
        for k in range(j+1, d):
            E = np.zeros((d, d), complex)
            E[j, k] = E[k, j] = 1.0
            gens.append(1j*E)                      # symmetric -> real symmetric
            F = np.zeros((d, d), complex)
            F[j, k] = 1j; F[k, j] = -1j
            gens.append(1j*F)                      # antisym -> (times i) hermitian*... act as real rotation
    for l in range(1, d):
        G = np.zeros((d, d), complex)
        for j in range(l): G[j, j] = 1.0
        G[l, l] = -l
        G *= np.sqrt(2.0/(l*(l+1)))
        gens.append(1j*G)
    return gens

def cglmp_opt_axes(rho, d, max_evals=4000, gens=None):
    """Coordinate ascent over Lie-algebra angles for the 4 local bases, maximizing I_d.
    Deterministic RNG. Returns (best_I, P).  Used for the MES control (analytic-opt
    settings unreachable by pure random search at d=4)."""
    if gens is None: gens = su_generators(d)
    ng = len(gens)
    def eval_bases(As, Bs):
        P = P_ab(rho, As, Bs); return cglmp(P, d), P
    def rot(U, gen, angle):
        return np.linalg.qr(np.linalg.matrix_exp(gen*angle) @ U)[0]
    best = -1e9; bestP = None
    # restarts
    for restart in range(6):
        As = [haar_unitary(d, RNG), haar_unitary(d, RNG)]
        Bs = [haar_unitary(d, RNG), haar_unitary(d, RNG)]
        I, P = eval_bases(As, Bs)
        for it in range(max_evals//6):
            improved = False
            for which in range(4):
                for gi in range(ng):
                    for sgn in (+1.0, -1.0):
                        step = 0.35
                        U = As[which] if which < 2 else Bs[which-2]
                        U2 = rot(U, gens[gi], sgn*step)
                        if which < 2: As2 = [As[0].copy(), As[1].copy()]; As2[which] = U2
                        else:         Bs2 = [Bs[0].copy(), Bs[1].copy()]; Bs2[which-2] = U2
                        I2, P2 = eval_bases(As2 if which < 2 else As, Bs2 if which >= 2 else Bs)
                        if I2 > I + 1e-7:
                            if which < 2: As = As2
                            else:         Bs = Bs2
                            I, P = I2, P2
                            improved = True
            if not improved: break
        if I > best: best, bestP = I, P
    return best, bestP

'''
src = src.replace(anchor, addition + anchor, 1)
# Use the new optimizer for MES control rows
old = "        I_m, P_m = optimize_cglmp(r_mes, d, trials=400 if d == 3 else (250 if d == 4 else 120))"
new = "        I_m, P_m = (cglmp_opt_axes(r_mes, d, max_evals=5200 if d == 4 else 2600)\n"
new += "                     if d >= 3 else optimize_cglmp(r_mes, d, trials=160))"
assert old in src; src = src.replace(old, new)
open('/home/team/shared/quantum_likeness/qudit_lhv_probe.py','w').write(src)
print('Lie-algebra optimizer injected; MES rows rewired')
