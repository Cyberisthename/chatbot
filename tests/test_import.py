def test_imports():
    import atomsim
    import atomsim.cli
    import atomsim.hydrogen.solver
    import atomsim.helium.solver
    import atomsim.fields.perturb
    import atomsim.inverse.tomo
    import atomsim.render.viz3d
    import atomsim.numerics.grids
    import atomsim.numerics.splitop


def test_version():
    import atomsim

    assert hasattr(atomsim, "__version__")
    assert atomsim.__version__ == "0.1.0"
