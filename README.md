# pyMicroDose
Repository for analytical calculations in microdosimetry.

Instalation:
--------------
1. Download the file pymicrodose-1.1.1.tar.gz
2. Install it on the Terminal using pip install pymicrodose-1.1.1.tar.gz

Examples of use:
----------------
    # Importing it into your code
    import pyMicroDose

    # Microdosimetry for protons
    p = pyMicroDose.Proton() # Optional: site diameter (d = 1 um by default). For d = x um, use pyMicroDose.Proton(x)
    # Quantities to get: meanE, stdE, meanS, sD, yF, yD, z1F, z1D, stdDevZ1 for each energy (in MeV). For example for, 1 MeV
    p.meanE(1)

    # Microdosimetry for alpha particles
    a = pyMicroDose.AlphaParticle() # Optional: site diameter (d = 1 um by default). For d = x um, use pyMicroDose.Proton(x)
    # Quantities to get: meanE, stdE, meanS, sD, yF, yD, z1F, z1D, stdDevZ1 for each energy (in MeV). For example for, 1 MeV
    a.meanE(1)

    # Stopping power from PSTAR and ASTAR databases are also obtainable
    protonStp = pyMicroDose.ProtonStoppingPower()
    protonStp.getStoppingPower(5)

    # It is possible to get also the coefficients for the microdosimetric models
    alphaCoeffs = pyMicroDose.AlphaCoefficients()
    alphaCoeffs.C_e
