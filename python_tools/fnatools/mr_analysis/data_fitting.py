#
# This file is part of the mr_analysis package
#

import numpy
# import pandas
# import scipy.optimize
# import scipy.special
from lmfit import Model


def hall_voltage_function(phi, phi_0=0, I_0=1, R_PHE=1, R_AHE=1, V_0=0,
                          H_A=0, H=1, phi_E=0, theta_M=90, theta_M0=0):
    """
    Calculate the (first harmonic) hall voltage based on the equation from
    MacNeill et al. (2017) PRB 96, 054450.

    Parameters
    ----------
    phi : float or numpy.ndarray
        in-plane field angle (degrees)
    phi_0 : float
        in-plane angle offset (degrees)
    I_0 : float
        Measurement current (A)
    R_PHE : float
        Planar-Hall resistance (Ohm)
    R_AHE : float
        Anomalous-Hall resistance (Ohm)
    V_0: float
        Offset voltage (V)
    H : float
        Applied external magnetic field strength (A/m)
    H_A : float
        In-plane uni-axial anisotropy field (A/m)
    theta_M : float or numpy.ndarray
        out-of-plane magnetization angle (degrees)
    theta_M0 : float
        out-of-plane magnetization angle offset (degrees)

    Returns
    -------
    V_H : float or numpy.ndarray
        First-harmonic Hall voltage
    """
    phi = numpy.radians(phi) - numpy.radians(phi_0)
    phi_M = phi - H_A / (2 * H) * numpy.sin(2 * (phi - phi_E))

    theta_M = numpy.radians(theta_M) - numpy.radians(theta_M0)

    C_PHE = numpy.sin(2 * phi_M) * numpy.sin(theta_M)**2
    C_AHE = numpy.cos(theta_M)

    V_H = I_0 * (R_PHE * C_PHE + R_AHE * C_AHE) + V_0

    return V_H


def hall_voltage_2nd_harmonic_function(phi, phi_0=0, I_0=1, R_PHE=1,
                                       R_AHE=1, V_0=0, V_ANE=0, Ms=0,
                                       H=1, H_A=0, phi_E=90, gamma=1,
                                       tau_ofl=1, tau_iad=1,
                                       tau_oad=0, tau_ifl=0, ):
    """
    Calculate the second harmonic hall voltage based on equation (2) from
    MacNeill et al. (2017) PRB 96, 054450.

    Parameters
    ----------
    phi : float or numpy.ndarray
        in-plane field angle (degrees)
    phi_0 : float
        in-plane angle offset (degrees)
    I_0 : float
        Measurement current (A)
    R_PHE : float
        Planar-Hall resistance (Ohm)
    R_AHE : float
        Anomalous-Hall resistance (Ohm)
    V_0: float
        Offset voltage (V)
    V_ANE : float
        Anomalous-Nernst voltage (V)
    Ms : float
        Saturation magnetization (A/m)
    H : float
        Applied external magnetic field strength (A/m)
    H_A : float
        In-plane uni-axial anisotropy field (A/m)
    phi_E : float
        Angle of anisotropy with the current flow direction (degrees)
    gamma : float
        Gyromagnetic ratio (unit unknown)
    tau_ofl : float
        out-of-plane field-like torque (tau_A in paper)
    tau_iad : float
        in-plane damping-like torque (tau_B in paper)
    tau_oad : float
        out-of-plane damping-like torque (not specified in paper)
    tau_ifl : float
        in-plane field-like torque (tau_S in paper)

    Returns
    -------
    V2_H : float or numpy.ndarray
        Second-harmonic Hall voltage
    """
    phi = numpy.radians(phi) - numpy.radians(phi_0)
    phi_M = phi - H_A / (2 * H) * numpy.sin(2 * (phi - phi_E))

    C_PHE = numpy.cos(2 * phi_M) * (tau_ofl * numpy.cos(phi_M) + tau_oad) / \
        (gamma * (H + H_A * numpy.cos(2 * phi_M - 2 * phi_E)))
    C_AHE = (tau_iad * numpy.cos(phi_M) + tau_ifl) / \
        (2 * gamma * (H + Ms + H_A * numpy.cos(2 * phi_M - 2 * phi_E)**2))
    C_ANE = numpy.cos(phi_M)

    V2_H = I_0 * (R_PHE * C_PHE + R_AHE * C_AHE) + V_ANE * C_ANE + V_0
    return V2_H


def fit_both_harmonics(data, fit_background=True, angle_col="Angle",
                       Vh1_col="Lock_In_2_X", Vh2_col="Lock_In_1_Y",
                       field_col="Magnetic_Field", current_col="Current",
                       paramsH1=None, paramsH2=None,
                       transfer_params={
                           "fixed": ["R_PHE", "phi_0"],
                           "free": ["H_A", "phi_E"]}):
    """
    Fit the first and second harmonic hall measurements subsequently to the
    hall-voltage function. This function generates the models, parameters.
    The first harmonic is fitted to obtain the PHE resistance, angular offset
    and (possibly) the anisotropy field and axis.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset that is to be fitted.
    fit_background : bool
        Whether or not to include a constant background in the fit.
        only valid if no parameters are given. Default is True.
    angle_col : string
        Name of the column containing the angular data (in degrees).
        Default is "Angle".
    Vh1_col : string
        Name of the column containing the first harmonic voltage data (in V).
        Default is "Lock_In_2_X".
    Vh2_col : string
        Name of the column containing the second harmonic voltage  data (in V).
        Default is "Lock_In_1_Y".
    field_col : string
        Name of the column containing the magnetic field data (in Tesla).
        Default is "Magnetic_Field".
    current_col : string
        Name of the column containing the measurement current data (in Amps).
        Default is "Current".
    paramsH1 : lmfit.Parameters
        If given, use these parameters for the first harmonic voltage fit.
    paramsH2 : lmfit.Parameters
        If given, use these parameters for the second harmonic voltage fit.
    transfer_params : dict of lists
        These parameters will be transferred from the first harmonic voltage
        fit to the second harmonic voltage fit. The dict can have two keys:
        "fixed" for the parameters that will be fixed in the second fit, and
        "free" for the parameters that will be copied but free to be fitted.

    Returns
    -------
    resultsH1 : lmfit.ModelResult
        Result of the first harmonic voltage fit.
    resultsH2 : lmfit.ModelResult
        Result of the second harmonic voltage fit.
    """
    # Make models
    H1 = Model(hall_voltage_function,
               independent_vars=["phi", "I_0", "H"])
    H2 = Model(hall_voltage_2nd_harmonic_function,
               independent_vars=["phi", "I_0", "H"])

    # make parameters of not existing
    if paramsH1 is None:
        H1pars = H1.make_params()
        H1pars["theta_M"].vary = False
        H1pars["theta_M"].value = 90
        H1pars["theta_M0"].vary = False
        H1pars["theta_M0"].value = 0
        H1pars["R_AHE"].vary = False
        H1pars["R_AHE"].value = 1
        H1pars["H_A"].vary = False
        H1pars["H_A"].value = 0
        H1pars["phi_E"].vary = False
        H1pars["phi_E"].value = 0
        if not fit_background:
            H1pars["V_0"].value = 0
            H1pars["V_0"].vary = False

    if paramsH2 is None:
        H2pars = H2.make_params()
        H2pars["Ms"].value = 0.800
        H2pars["Ms"].vary = False
        H2pars["gamma"].value = 1
        H2pars["gamma"].vary = False
        H2pars["R_AHE"].value = 1
        H2pars["R_AHE"].vary = False
        H2pars["tau_ifl"].value = 0
        H2pars["tau_ifl"].vary = False
        H2pars["tau_oad"].value = 0
        H2pars["tau_oad"].vary = False
        if not fit_background:
            H2pars["V_0"].value = 0
            H2pars["V_0"].vary = False

    try:
        I_0 = data[current_col]
    except KeyError:
        I_0 = 1

    resultsH1 = H1.fit(data[Vh1_col],
                       H1pars,
                       phi=data[angle_col],
                       I_0=I_0,
                       H=data[field_col])

    if transfer_params is not None and "free" in transfer_params:
        for par in transfer_params["free"]:
            H2pars[par].value = resultsH1.params[par].value
            H2pars[par].vary = True

    if transfer_params is not None and "fixed" in transfer_params:
        for par in transfer_params["fixed"]:
            H2pars[par].value = resultsH1.params[par].value
            H2pars[par].vary = False

    resultsH2 = H2.fit(data[Vh2_col],
                       H2pars,
                       phi=data[angle_col],
                       I_0=I_0,
                       H=data[field_col])

    return resultsH1, resultsH2


def fit_hall_voltage(data, V_column="Lock_In_1_X", I0=1e-3, harmonic=1):
    ydata = data[V_column]

    # Make model
    if harmonic == 1:
        model = Model(hall_voltage_function)
    elif harmonic == 2:
        model = Model(hall_voltage_2nd_harmonic_function)
    else:
        raise ValueError("The parameter harmonic received an invalid value")

    # Generate parameters
    params = model.make_params()
    params.pretty_print()

    # Set initial guess and fix params
    # assert False, "Need to fix the initial values"
    params["phi_M0"].value = 0
    params["theta_M"].value = 90
    params["theta_M"].vary = False
    params["I_0"].value = I0
    params["I_0"].vary = False
    params["R_PHE"].value = ydata.max() - ydata.min()
    params["R_AHE"].value = 0
    params["R_AHE"].vary = False
    params["R_0"].value = ydata.mean()

    # Fit the model
    result = model.fit(ydata, params, phi_M=data["Angle"])
    print(result.fit_report())

    # # Initial Guess
    # phi_M0 = 0
    # R_PHE = ydata.max() - ydata.min()
    # R_AHE = 0
    # theta_M = 90
    # C = ydata.mean()

    # # Fitting the function
    # pout, pcov = scipy.optimize.curve_fit(
    #     hall_resistance_function,
    #     data["Angle"], ydata,
    #     [phi_M0, R_PHE, R_AHE, theta_M, C],)

    return result


def simplified_1st_harmonic(phi, phi_0=0, theta=90, theta_0=0,
                            I_0=1, R_PHE=1, R_AHE=1, V_0=0):
    """
    Calculate the first harmonic hall voltage based on
    MacNeill et al. (2017) PRB 96, 054450.

    Assumes the magnetization and field direction are identical.

    Parameters
    ----------
    phi : float or numpy.ndarray
        in-plane field angle (degrees)
    phi_0 : float
        in-plane angle offset (degrees)
    theta : float or numpy.ndarray
        out-of-plane field angle (degrees, 0 == along z-axis)
    theta_0 : float
        out-of-plane angle offset (degrees)
    I_0 : float
        Measurement current (A)
    R_PHE : float
        Planar-Hall resistance (Ohm)
    R_AHE : float
        Anomalous-Hall resistance (Ohm)
    V_0: float
        Offset voltage (V)

    Returns
    -------
    V1_H : float or numpy.ndarray
        Second-harmonic Hall voltage
    """
    phi_M = numpy.radians(phi) - numpy.radians(phi_0)
    theta_M = numpy.radians(theta) - numpy.radians(theta_0)

    C_PHE = numpy.sin(phi_M * 2) * numpy.sin(theta_M)**2
    C_AHE = numpy.cos(theta_M)

    V1_H = V_0 + I_0 * (R_PHE * C_PHE +
                        R_AHE * C_AHE)
    return V1_H


def simplified_2nd_harmonic(phi, phi_0=0, I_0=1, R_PHE_FL=1, R_PHE_DL=1,
                            R_AHE_DL=1, V_0=0):
    """
    Calculate the second harmonic hall voltage based on equation (2) from
    MacNeill et al. (2017) PRB 96, 054450.

    Assumes the magnetization and field direction are identical.

    Parameters
    ----------
    phi : float or numpy.ndarray
        in-plane field angle (degrees)
    phi_0 : float
        in-plane angle offset (degrees)
    I_0 : float
        Measurement current (A)
    R_PHE_FL : float
        Planar-Hall resistance for FL torque (Ohm)
    R_PHE_DL : float
        Planar-Hall resistance for DL torque (Ohm)
    R_AHE_DL : float
        Anomalous-Hall resistance for DL torque (Ohm)
    V_0: float
        Offset voltage (V)

    Returns
    -------
    V2_H : float or numpy.ndarray
        Second-harmonic Hall voltage
    """
    phi_M = numpy.radians(phi) - numpy.radians(phi_0)

    C_PHE_FL = numpy.cos(2 * phi_M) * numpy.cos(phi_M)
    C_PHE_DL = numpy.cos(2 * phi_M)
    C_AHE_DL = numpy.cos(phi_M)

    V2_H = V_0 + I_0 * (R_PHE_FL * C_PHE_FL +
                        R_PHE_DL * C_PHE_DL +
                        R_AHE_DL * C_AHE_DL * 0.5)
    return V2_H


def simplified_torque_field_dependence(H, Ms=0, tau=0, tau_0=0):
    """
    Generalized and simplified field dependence of the torques based on
    equation (2) from MacNeill et al. (2017) PRB 96, 054450.

    Assumes the magnetization and field direction are identical.

    Parameters
    ----------
    H : float or numpy.ndarray
        Magnetic field strength (A/m)
    Ms : float
        Saturation magnetization (A/m)
    tau : float
        Torque (A/m)
    tau_0 : float
        Effective torque offset (1)

    Returns
    -------
    tau_1 : float or numpy.ndarray
        Effective torque (1)
    """

    tau_1 = tau_0 + tau / (H + Ms)

    return tau_1
