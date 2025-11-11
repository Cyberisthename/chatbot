"""
Infinite Compute Lab – Unified App (Streamlit)
================================================

A thin wrapper around experimental physics simulations:
- 3D atom-from-constants (imaginary-time Schrödinger solver)
- Double-slit interference
- Field interference (2D wave dynamics)
- CHSH Bell inequality violation
- Relativistic task graph
- Holographic entropy scaling

All computations run locally on CPU with deterministic seed 424242.
"""

import json
import io

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set deterministic seed globally
SEED = 424242

def new_rng():
    return np.random.default_rng(SEED)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Infinite Compute Lab",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Infinite Compute Lab")
st.caption("Physics simulations from first principles — deterministic seed 424242")
st.divider()

# ============================================================
# TABS
# ============================================================

atom_tab, slit_tab, field_tab, chsh_tab, rel_tab, holo_tab = st.tabs([
    "Atom 3D",
    "Double-slit",
    "Field interference",
    "CHSH",
    "Relativistic graph",
    "Holographic entropy",
])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def laplacian3d(psi, dx):
    """3D finite-difference Laplacian using 6-point stencil."""
    lap = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) +
        np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2) -
        6.0 * psi
    ) / (dx * dx)
    return lap


def normalize(psi, dx):
    """Normalize wavefunction so that ∫|ψ|² dV = 1."""
    dV = dx**3
    norm = np.sqrt((np.abs(psi)**2).sum() * dV)
    if norm <= 0:
        return psi, 0.0
    return psi / norm, norm


def compute_energy(psi, V, dx):
    """Compute total energy E = <ψ| -½∇² + V |ψ>."""
    dV = dx**3
    lap = laplacian3d(psi, dx)
    kinetic = -0.5 * (np.conj(psi) * lap).real.sum() * dV
    potential = (np.conj(psi) * V * psi).real.sum() * dV
    return float(kinetic + potential)


def potential_field(shape, box, centers, Z, softening):
    """Build 3D potential field V(x,y,z) for nuclear centers."""
    N = shape[0]
    L = box
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    V = np.zeros(shape, dtype=np.float64)
    for xc, yc, zc in centers:
        r_sq = (X - xc)**2 + (Y - yc)**2 + (Z_grid - zc)**2
        V -= Z / np.sqrt(r_sq + softening**2)
    return V


def solve_atom_3d(N, box, steps, dt, Z=1.0, softening=0.3):
    """Solve 3D hydrogen atom via imaginary-time evolution."""
    centers = [[0.0, 0.0, 0.0]]
    V = potential_field((N, N, N), box, centers, Z, softening)
    dx = box / N
    
    # Auto-scale dt for stability
    dt_scaled = dt * (dx / (box / 64))**2
    
    # Initialize wavefunction
    x = np.linspace(-box/2, box/2, N)
    y = np.linspace(-box/2, box/2, N)
    z = np.linspace(-box/2, box/2, N)
    X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
    r_sq = X**2 + Y**2 + Z_grid**2
    np.random.seed(SEED)
    psi = np.exp(-r_sq / 4.0) * (1.0 + 0.05 * np.random.randn(*X.shape))
    psi, _ = normalize(psi, dx)
    
    energies = []
    
    # Evolve in imaginary time
    for step in range(steps):
        lap = laplacian3d(psi, dx)
        dpsi = 0.5 * lap - V * psi
        psi = psi + dt_scaled * dpsi
        psi, _ = normalize(psi, dx)
        
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            E = compute_energy(psi, V, dx)
            energies.append(E)
    
    density = np.abs(psi)**2
    
    # Compute radial density for comparison with analytic 1s
    r_vals = np.linspace(0, box/2, 100)
    radial_density = []
    for r in r_vals:
        mask = (r_sq >= r**2) & (r_sq < (r + dx)**2)
        radial_density.append(density[mask].sum())
    radial_density = np.array(radial_density)
    # Normalize radial density
    if radial_density.max() > 0:
        radial_density /= radial_density.max()
    
    # Analytic 1s orbital: ψ(r) = 2 * exp(-r)
    analytic_1s = 4 * r_vals**2 * np.exp(-2 * r_vals)
    if analytic_1s.max() > 0:
        analytic_1s /= analytic_1s.max()
    
    return {
        "psi": psi,
        "density": density,
        "energies": energies,
        "final_energy": energies[-1],
        "radial_r": r_vals,
        "radial_density": radial_density,
        "analytic_1s": analytic_1s,
    }


def double_slit_sim(N=512, k=20.0):
    """Simulate double-slit interference with adapters."""
    x = np.linspace(-1.5, 1.5, N)
    
    # Path A: gaussian at -0.4
    width_A, center_A = 0.25, -0.4
    psi_A = np.exp(-(x - center_A)**2 / (2 * width_A**2))
    
    # Path B: gaussian at +0.4
    width_B, center_B = 0.25, +0.4
    psi_B = np.exp(-(x - center_B)**2 / (2 * width_B**2))
    
    # Interference: add phase to B
    phi = k * x
    psi_A_complex = psi_A.astype(complex)
    psi_B_complex = psi_B * np.exp(1j * phi)
    psi_combined = psi_A_complex + psi_B_complex
    intensity_interference = np.abs(psi_combined)**2
    
    # Control: no phase
    psi_control = psi_A + psi_B
    intensity_control = psi_control**2
    
    # Visibility in central region
    center_mask = (x > -0.8) & (x < 0.8)
    I_max = float(np.max(intensity_interference[center_mask]))
    I_min = float(np.min(intensity_interference[center_mask]))
    visibility = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0
    
    return {
        "x": x,
        "intensity_interference": intensity_interference,
        "intensity_control": intensity_control,
        "visibility": visibility,
    }


def field_interference_sim(N=256, T=400, src=2):
    """Simulate 2D field interference."""
    phi = np.zeros((N, N), dtype=np.complex128)
    
    # Place sources
    local_rng = new_rng()
    source_locs = []
    source_phases = []
    for i in range(src):
        x = local_rng.integers(N // 4, 3 * N // 4)
        y = local_rng.integers(N // 4, 3 * N // 4)
        source_locs.append((x, y))
        source_phases.append(2 * np.pi * local_rng.random())
    
    # Place detectors
    det_locs = [
        (N // 4, N // 4),
        (3 * N // 4, N // 4),
        (N // 4, 3 * N // 4),
        (3 * N // 4, 3 * N // 4),
    ]
    
    dt = 0.01
    diffusion = 0.5
    detector_readings = []
    
    x_grid, y_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    gaussian_scale = 2 * (N / 20) ** 2
    
    # Time evolution
    for t in range(T):
        # Add source terms
        for (sx, sy), phase in zip(source_locs, source_phases):
            amplitude = 0.1 * np.exp(1j * (phase + 0.05 * t))
            dist_sq = (x_grid - sx) ** 2 + (y_grid - sy) ** 2
            source_profile = amplitude * np.exp(-dist_sq / gaussian_scale)
            phi += source_profile * dt
        
        # Laplacian for diffusion
        phi_padded = np.pad(phi, 1, mode="constant")
        laplacian = (
            phi_padded[:-2, 1:-1] + phi_padded[2:, 1:-1] +
            phi_padded[1:-1, :-2] + phi_padded[1:-1, 2:] -
            4 * phi
        )
        
        # Update
        phi += dt * diffusion * laplacian
        phi *= np.exp(-0.001 * dt)  # damping
        
        # Record detectors
        det_intensities = [np.abs(phi[x, y]) ** 2 for x, y in det_locs]
        detector_readings.append(det_intensities)
    
    detector_readings = np.array(detector_readings)
    
    # Compute visibility
    visibilities = []
    for i in range(len(det_locs)):
        I = detector_readings[:, i]
        I_max = np.max(I)
        I_min = np.min(I)
        if I_max + I_min > 1e-12:
            vis = (I_max - I_min) / (I_max + I_min)
        else:
            vis = 0.0
        visibilities.append(float(vis))
    
    mean_visibility = float(np.mean(visibilities))
    
    return {
        "phi": phi,
        "intensity": np.abs(phi) ** 2,
        "source_locs": source_locs,
        "det_locs": det_locs,
        "visibility": mean_visibility,
    }


def chsh_sim(shots=50000, depol=0.0):
    """Simulate CHSH Bell inequality violation."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    
    if depol > 0:
        I = np.eye(4, dtype=complex)
        rho = (1 - depol) * rho + depol * I / 4
    
    def ry(angle):
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def measurement_probs(rho_matrix, angle_a, angle_b):
        Ua = ry(-angle_a)
        Ub = ry(-angle_b)
        U = np.kron(Ua, Ub)
        rotated = U @ rho_matrix @ U.conj().T
        probs = np.real_if_close(np.diag(rotated))
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            return np.full(4, 0.25)
        return probs / total
    
    def correlation_from_probs(probs):
        expectation = 0.0
        for idx, prob in enumerate(probs):
            parity = -1 if (idx.bit_count() % 2) else 1
            expectation += parity * float(prob)
        return expectation
    
    angles = {
        "A": 0.0,
        "A'": np.pi / 2,
        "B": np.pi / 4,
        "B'": -np.pi / 4,
    }
    
    pairs = {
        "AB": ("A", "B"),
        "AB'": ("A", "B'"),
        "A'B": ("A'", "B"),
        "A'B'": ("A'", "B'"),
    }
    
    local_rng = new_rng()
    expectations = {}
    counts = {}
    
    for label, (a_key, b_key) in pairs.items():
        probs = measurement_probs(rho, angles[a_key], angles[b_key])
        expectations[label] = correlation_from_probs(probs)
        outcomes = local_rng.choice(4, size=max(1, shots), p=probs)
        bucket = {}
        for idx in outcomes:
            bitstring = format(int(idx), "02b")
            bucket[bitstring] = bucket.get(bitstring, 0) + 1
        counts[label] = bucket
    
    S = expectations["AB"] + expectations["AB'"] + expectations["A'B"] - expectations["A'B'"]
    
    return [expectations[k] for k in ("AB", "AB'", "A'B", "A'B'")], float(S)


def relativistic_graph(n=64, beta=0.6):
    """Build relativistic task graph with time dilation."""
    edges = min(256, n * (n - 1) // 2)
    
    local_rng = new_rng()
    
    # Generate DAG
    dag = []
    selected = set()
    for _ in range(edges):
        u = int(local_rng.integers(0, n - 1))
        v = int(local_rng.integers(u + 1, n))
        if (u, v) not in selected:
            weight = float(local_rng.uniform(0.5, 2.0))
            dag.append((u, v, weight))
            selected.add((u, v))
    
    # Gamma factors
    velocities = local_rng.uniform(0.0, beta, size=n)
    velocities = np.clip(velocities, 0, 0.999)
    gamma = 1.0 / np.sqrt(1.0 - velocities ** 2)
    
    # Longest path - Newtonian
    times_newton = np.zeros(n)
    adjacency = [[] for _ in range(n)]
    for u, v, w in dag:
        adjacency[v].append((u, w))
    
    for v in range(n):
        candidates = []
        for u, w in adjacency[v]:
            t = times_newton[u] + w
            candidates.append(t)
        if candidates:
            times_newton[v] = max(candidates)
        elif v > 0:
            times_newton[v] = times_newton[v - 1]
    
    # Longest path - Relativistic
    times_rel = np.zeros(n)
    for v in range(n):
        candidates = []
        for u, w in adjacency[v]:
            t = times_rel[u] + w / gamma[u]
            candidates.append(t)
        if candidates:
            times_rel[v] = max(candidates)
        elif v > 0:
            times_rel[v] = times_rel[v - 1]
    
    duration_newton = float(times_newton[-1])
    duration_rel = float(times_rel[-1])
    
    return dag, duration_newton, times_newton, times_rel


def holo_entropy(cube):
    """Compute holographic entropy scaling from 3D voxel cube."""
    N = cube.shape[0]
    max_radius = max(2, N // 2 - 1)
    radii = np.linspace(2, max_radius, 50, dtype=int)
    radii = np.unique(radii[radii > 1])
    
    local_rng = new_rng()
    
    entropy_values = []
    areas = []
    
    def binary_entropy(p):
        if p in (0.0, 1.0) or p < 0.0 or p > 1.0:
            return 0.0
        return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    
    for r in radii:
        low = int(r)
        high = int(N - r)
        if high <= low:
            continue
        center = local_rng.integers(low, high, size=3)
        sub = cube[
            center[0] - r : center[0] + r,
            center[1] - r : center[1] + r,
            center[2] - r : center[2] + r,
        ]
        
        entropy_values.append(binary_entropy(np.mean(sub)))
        side = 2 * r
        area = 6 * (side ** 2)
        areas.append(float(area))
    
    if not areas:
        return [1.0], [1.0], [0.0, 0.0]
    
    entropy_values = np.array(entropy_values, dtype=float)
    areas = np.array(areas, dtype=float)
    
    # Linear fit: H = c * A + b
    coeff = np.polyfit(areas, entropy_values, 1)
    
    return areas, entropy_values, coeff


# ============================================================
# TAB: ATOM 3D
# ============================================================

with atom_tab:
    st.subheader("3D Atom from Constants")
    st.markdown("""
    True imaginary-time Schrödinger solver (no hand-tuned orbitals).
    Solves `-½∇²ψ + V(r)ψ = Eψ` on a 3D grid.
    """)
    
    N_atom = st.slider("Grid size N (N³ cells)", 32, 128, 64, 8)
    box_atom = st.slider("Box size (atomic units)", 8.0, 20.0, 12.0, 1.0)
    steps_atom = st.slider("Imaginary-time steps", 100, 1000, 400, 50)
    
    if st.button("Solve Atom"):
        with st.spinner("Solving 3D Schrödinger equation..."):
            result = solve_atom_3d(N_atom, box_atom, steps_atom, dt=0.002)
        
        st.success(f"✅ Final energy: {result['final_energy']:.6f} Hartree (exact H 1s: -0.5)")
        
        # Plot energy convergence
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.plot(result["energies"], 'b-', linewidth=2)
        ax1.axhline(-0.5, color='r', linestyle='--', label='Analytic 1s')
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Energy (Hartree)")
        ax1.set_title("Energy Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, clear_figure=True)
        
        # Plot 3 MIPs (max-intensity projections)
        density = result["density"]
        mip_xy = density.max(axis=2)
        mip_xz = density.max(axis=1)
        mip_yz = density.max(axis=0)
        
        fig2, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(15, 4))
        
        im_xy = ax_xy.imshow(mip_xy.T, origin='lower', cmap='inferno', interpolation='bilinear')
        ax_xy.set_title('XY projection')
        ax_xy.axis('off')
        plt.colorbar(im_xy, ax=ax_xy, fraction=0.046)
        
        im_xz = ax_xz.imshow(mip_xz.T, origin='lower', cmap='inferno', interpolation='bilinear')
        ax_xz.set_title('XZ projection')
        ax_xz.axis('off')
        plt.colorbar(im_xz, ax=ax_xz, fraction=0.046)
        
        im_yz = ax_yz.imshow(mip_yz.T, origin='lower', cmap='inferno', interpolation='bilinear')
        ax_yz.set_title('YZ projection')
        ax_yz.axis('off')
        plt.colorbar(im_yz, ax=ax_yz, fraction=0.046)
        
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        
        # Plot radial check vs analytic
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4))
        ax3.plot(result["radial_r"], result["radial_density"], 'b-', linewidth=2, label='Computed')
        ax3.plot(result["radial_r"], result["analytic_1s"], 'r--', linewidth=2, label='Analytic 1s')
        ax3.set_xlabel("Radius (a.u.)")
        ax3.set_ylabel("Radial density (normalized)")
        ax3.set_title("Radial Density vs Analytic 1s Orbital")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, clear_figure=True)
        
        # Download links
        st.markdown("### Download results")
        
        # density.npy
        density_bytes = io.BytesIO()
        np.save(density_bytes, result["density"])
        density_bytes.seek(0)
        st.download_button("Download density.npy", density_bytes, "density.npy")
        
        # psi.npy
        psi_bytes = io.BytesIO()
        np.save(psi_bytes, result["psi"])
        psi_bytes.seek(0)
        st.download_button("Download psi.npy", psi_bytes, "psi.npy")
        
        # energy.json
        energy_json = json.dumps({
            "energies": result["energies"],
            "final_energy": result["final_energy"],
            "N": N_atom,
            "box": box_atom,
            "steps": steps_atom,
        }, indent=2)
        st.download_button("Download energy.json", energy_json, "energy.json")


# ============================================================
# TAB: DOUBLE-SLIT
# ============================================================

with slit_tab:
    st.subheader("Adapter Double-Slit")
    st.markdown("Interference with visibility readout.")
    
    k_slit = st.slider("Phase parameter k (fringes)", 10.0, 40.0, 20.0, 1.0)
    
    if st.button("Run Double-Slit"):
        result = double_slit_sim(k=k_slit)
        
        st.metric("Visibility", f"{result['visibility']:.4f}")
        if result['visibility'] > 0.2:
            st.success("Quantum-like behavior: high visibility")
        else:
            st.warning("Low visibility: interference weak")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(result["x"], result["intensity_interference"], 'b-', linewidth=1.5)
        ax1.set_xlabel("Position x")
        ax1.set_ylabel("Intensity")
        ax1.set_title("Interference Pattern (with phase)")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(result["x"], result["intensity_control"], 'r-', linewidth=1.5)
        ax2.set_xlabel("Position x")
        ax2.set_ylabel("Intensity")
        ax2.set_title("Control Pattern (no phase)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


# ============================================================
# TAB: FIELD INTERFERENCE
# ============================================================

with field_tab:
    st.subheader("2D Field Interference")
    st.markdown("Wave field dynamics with multiple sources.")
    
    N_field = st.slider("Grid size", 128, 512, 256, 64)
    T_field = st.slider("Time steps", 100, 800, 400, 100)
    src_field = st.slider("Number of sources", 1, 5, 2, 1)
    
    if st.button("Run Field Simulation"):
        with st.spinner("Computing field evolution..."):
            result = field_interference_sim(N=N_field, T=T_field, src=src_field)
        
        st.metric("Mean Visibility", f"{result['visibility']:.4f}")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(result["intensity"].T, origin='lower', cmap='viridis', interpolation='bilinear')
        
        # Mark sources
        for idx, (x, y) in enumerate(result["source_locs"]):
            ax.plot(x, y, "r*", markersize=15, label="Source" if idx == 0 else "")
        
        # Mark detectors
        for idx, (x, y) in enumerate(result["det_locs"]):
            ax.plot(
                x, y, "wo", markersize=10,
                markeredgecolor="red", markeredgewidth=2,
                label="Detector" if idx == 0 else "",
            )
        
        plt.colorbar(im, ax=ax, label='Intensity |φ|²')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Field Interference Pattern (Final)')
        ax.legend()
        st.pyplot(fig, clear_figure=True)


# ============================================================
# TAB: CHSH
# ============================================================

with chsh_tab:
    st.subheader("CHSH Bell inequality")
    st.markdown("Quantum entanglement test: S > 2 violates classical bound.")
    
    shots = st.slider("Shots per setting", 1000, 100000, 20000, 1000)
    depol = st.slider("Depolarizing noise p", 0.0, 0.5, 0.0, 0.01)
    
    if st.button("Run CHSH"):
        E, S = chsh_sim(shots=shots, depol=depol)
        st.write({"correlators": [float(e) for e in E], "S": float(S)})
        if S > 2.0:
            st.success(f"Quantum violation achieved (S={S:.3f} > 2)")
        else:
            st.warning(f"No violation (S={S:.3f}) — noise too high or settings off")


# ============================================================
# TAB: RELATIVISTIC GRAPH
# ============================================================

with rel_tab:
    st.subheader("Relativistic task graph")
    st.markdown("Time dilation in parallel computation.")
    
    n = st.slider("Nodes", 8, 64, 16, 1)
    beta = st.slider("Velocity β (v/c)", 0.0, 0.9, 0.6, 0.01)
    
    if st.button("Build graph"):
        G, dur_n, Tn, Tp = relativistic_graph(n=n, beta=beta)
        
        # Plot timelines
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(list(Tn), label='Newtonian', linewidth=2)
        ax.plot(list(Tp), label='Relativistic (proper)', linewidth=2)
        ax.set_title('Completion time profiles')
        ax.set_xlabel('Node index')
        ax.set_ylabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)


# ============================================================
# TAB: HOLOGRAPHIC ENTROPY
# ============================================================

with holo_tab:
    st.subheader("Holographic entropy (toy)")
    st.markdown("Entropy ~ Area scaling test.")
    
    N_holo = st.slider("Cube size", 32, 96, 64, 8)
    corr = st.slider("Correlation σ (voxels)", 0.0, 5.0, 2.0, 0.25)
    
    if st.button("Generate cube"):
        with st.spinner("Generating voxel cube..."):
            local_rng = new_rng()
            cube = local_rng.normal(size=(N_holo, N_holo, N_holo))
            if corr > 0:
                try:
                    from scipy.ndimage import gaussian_filter
                except ImportError:
                    st.error("scipy is required for correlated smoothing. Install with `pip install scipy`. Using uncorrelated noise instead.")
                else:
                    cube = gaussian_filter(cube, corr)
            cube = (cube > 0).astype(np.uint8)
        
        zmid = cube[N_holo//2]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(zmid, cmap='inferno', origin='lower')
        ax.set_title('Voxel density slice (mid-plane)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig, clear_figure=True)
        
        areas, ent, coeff = holo_entropy(cube)
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
        ax2.scatter(areas, ent, label='samples', alpha=0.7)
        ax2.plot(areas, coeff[0]*areas + coeff[1], 'r-', linewidth=2, label='fit')
        ax2.set_xlabel('Boundary area A(r)')
        ax2.set_ylabel('Entropy H(r) [bits]')
        ax2.set_title('Entropy ~ c·Area + b (holographic scaling)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)
        
        st.write(f"Fit: H = {coeff[0]:.6f} * A + {coeff[1]:.6f}")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("© 2025 Infinite Compute Lab — deterministic seed 424242. This UI is a thin wrapper; all math runs locally on CPU.")
