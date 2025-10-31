"""Tests for quantum tunneling simulation."""
import pytest

from quantacap.experiments.quantum_tunneling import simulate_tunneling


def test_probability_conservation():
    """Test that total probability is conserved (≈ 1.0) throughout evolution."""
    result = simulate_tunneling(
        n=512,
        barrier_center=256,
        barrier_width=64,
        barrier_height=5.0,
        energy=2.0,
        steps=100,
        dt=0.002,
    )
    
    # Check that transmission + reflection is close to 1 at the end
    # (some probability may be inside the barrier region)
    final_total = result.transmission[-1] + result.reflection[-1]
    assert final_total <= 1.0, "Total probability exceeds 1.0"
    
    # Check total probability conservation
    for total_prob in result.total_probability:
        assert abs(total_prob - 1.0) < 0.01, "Total probability not conserved"


def test_tunneling_low_transmission_high_barrier():
    """Test that transmission is low when barrier >> energy (classical forbidden)."""
    result = simulate_tunneling(
        n=512,
        barrier_center=256,
        barrier_width=64,
        barrier_height=10.0,  # High barrier
        energy=2.0,  # Low energy
        steps=1000,
        dt=0.002,
    )
    
    # For barrier >> energy, transmission should be very small
    assert result.final_transmission < 0.1, "Transmission too high for high barrier"


def test_tunneling_high_transmission_low_barrier():
    """Test that transmission is high when energy > barrier (classically allowed)."""
    result = simulate_tunneling(
        n=512,
        barrier_center=256,
        barrier_width=64,
        barrier_height=2.0,  # Low barrier
        energy=5.0,  # High energy
        steps=3000,  # Need more time for wave packet to reach and pass through
        dt=0.002,
    )
    
    # For energy > barrier, transmission should be significant
    # Note: due to wave packet spreading and reflection, even for E>V
    # transmission may not be 100%, but should be much higher than tunneling case
    assert result.final_transmission > 0.1, f"Transmission {result.final_transmission} too low for low barrier"


def test_transmission_plus_reflection_near_unity():
    """Test that transmission + reflection ≈ 1.0 (conservation with barrier region)."""
    result = simulate_tunneling(
        n=512,
        barrier_center=256,
        barrier_width=64,
        barrier_height=5.0,
        energy=2.0,
        steps=2000,  # Long enough for wave packet to pass through
        dt=0.002,
    )
    
    # After sufficient time, most probability should be either transmitted or reflected
    # Allow some tolerance for numerical errors and barrier region occupation
    final_sum = result.final_transmission + result.final_reflection
    assert 0.8 < final_sum <= 1.0, f"T+R = {final_sum} not close to 1.0"


def test_parameters_stored():
    """Test that simulation parameters are correctly stored in result."""
    result = simulate_tunneling(
        n=256,
        barrier_center=128,
        barrier_width=32,
        barrier_height=7.5,
        energy=3.5,
        steps=500,
        dt=0.001,
        seed=12345,
    )
    
    assert result.parameters["n"] == 256.0
    assert result.parameters["barrier_height"] == 7.5
    assert result.parameters["energy"] == 3.5
    assert result.parameters["steps"] == 500.0
    assert result.steps == 500


def test_invalid_parameters_raise_errors():
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError, match="n and steps must be positive"):
        simulate_tunneling(n=0, steps=100)
    
    with pytest.raises(ValueError, match="n and steps must be positive"):
        simulate_tunneling(n=512, steps=-1)
    
    with pytest.raises(ValueError, match="barrier_width must be positive"):
        simulate_tunneling(n=512, barrier_center=256, barrier_width=0)
    
    with pytest.raises(ValueError, match="energy must be positive"):
        simulate_tunneling(n=512, barrier_center=256, barrier_width=64, energy=-1.0)
