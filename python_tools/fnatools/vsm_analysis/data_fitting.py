import pandas
import numpy
import scipy.optimize
import scipy.special


def tanh_function(B: numpy.ndarray, B0: float=0, B1: float=1, Ms: float=1,
                  M0: float=0, dM0dB: float=0):
    """
    The function that is used for fitting a hysteresis curve:
    tanh function + linear background
    """
    x = (B - B0) / B1
    langevin = Ms * numpy.tanh(x)
    background = M0 + dM0dB * B
    return langevin + background


def langevin_function(B: numpy.ndarray, B0: float=0, B1: float=1, Ms: float=1,
                      M0: float=0, dM0dB: float=0):
    """
    The function that is used for fitting a hysteresis curve:
    Langevin function + linear background
    """
    x = (B - B0) / B1
    langevin = Ms * (1 / numpy.tanh(x) - 1 / x)
    background = M0 + dM0dB * B
    return langevin + background


def fit_hysteresis(data: pandas.DataFrame):
    """
    Function to fit a hysteresis curve using a Langevin function (with linear
    background). The data is separated in ramping up and down parts and fitted
    separately.

    Parameters
    ----------
    data : pandas.DataFrame
        Measurement that is to be fitted

    Returns
    -------
    dict
        Fit parameters for the up and down ramped curves
    dict
        Fit errors for the up and down ramped curves
    """
    data = data.copy()
    data.sort_values('Time_Stamp', inplace=True)

    datasets = {}
    datasets[+1] = data[numpy.gradient(data["Induction"]) > 0].copy()
    datasets[-1] = data[numpy.gradient(data["Induction"]) < 0].copy()

    output_params = {}
    output_errors = {}
    for key, dataset in datasets.items():
        if len(dataset) == 0:
            continue

        dataset.sort_values("Induction", inplace=True)
        """
        Initialize parameters for fitting using the Langevin function:
        M = Ms * L((B - B0) / B1) + M0 + B * dM0dB
        B0: peak in derivative dM/dB
        B1:
        Ms: amplitude of total measurement
        M0: obtained from background_subtraction
        dM0dB: obtained from background_subtraction
        """
        gradient = numpy.gradient(dataset["Moment"], dataset["Induction"])

        dataset, BG = background_subtraction(dataset, keep_uncorrected=True)

        M0 = BG["offset"]
        dM0dB = BG["slope"]

        gradient = numpy.gradient(dataset["Moment"], dataset["Induction"])
        max_gradient_idx = numpy.argmax(gradient)
        try:
            B0 = float(numpy.array(dataset["Induction"])[max_gradient_idx])
        except TypeError:
            print(max_gradient_idx)

        B1 = numpy.mean(numpy.abs(numpy.diff(data["Induction"])))
        Ms = (numpy.max(dataset["Moment"]) -
              numpy.min(dataset["Moment"])) / 2

        input_param = [B0, B1, Ms, M0, dM0dB]

        """
        Actual curve-fitting
        """

        input_param2, pcov = scipy.optimize.curve_fit(
            tanh_function,
            dataset["Induction"], dataset["Moment_uncorrected"],
            input_param, dataset["M_Std_Err"])

        popt, pcov = scipy.optimize.curve_fit(
            langevin_function,
            dataset["Induction"], dataset["Moment_uncorrected"],
            input_param2, dataset["M_Std_Err"], xtol=1e-12, ftol=1e-12)

        p_sigma = numpy.sqrt(numpy.diag(pcov))

        output_params[key] = popt
        output_errors[key] = p_sigma

    return output_params, output_errors


def background_subtraction(data, sort_column="Time_Stamp",
                           x_column="Induction", y_column="Moment",
                           keep_uncorrected: bool = False, order=1,
                           slope_error: float = 1e-2, edge_points: int = 4,
                           use_field_weights: bool = True):
    """
    Function to correct an hysteresis loop for the (linear) background signal
    by looking at the straight parts of the measurement at the high (positive
    and negative) fields. Function assumes at least 4 data-points (ramping up
    and down combined) at either end of the measurement to be linear. This
    can be changed by passing the edge_points argument.

    Parameters
    ----------
    data : pandas.DataFrame
        Measurement that is to be corrected.
    sort_column : str
        column name used for sorting prior and after the background subtraction
    x_column: str
        column name used as x-axis for background subtraction
    y_column: str
        column name that is to be background subtracted
    keep_uncorrected : bool
        Keep the uncorrected moment as a new column ("Moment_uncorrected") in
        the DataFrame.
    slope_range : float
        The maximum error (part of the slope at the ends of the measurement)
        that is accepted for the linear regions in the measurement.
    edge_points : int
        The number of points on the edge that is assumed to be part of the
        linear regime. Default is 4, to account for double points on changing
        from ramping up to ramping down.
    use_field_weights : bool
        Use the induction (tanh(abs(B/250mT))) as weights for fitting a linear
        curve to the linear parts of the measurement. This makes the outer most
        edges the most influential for the subtracted background. Default is
        True.

    Returns
    -------
    pandas.DataFrame
        Background subtracted measurement.
    dict
        Offset (key is "offset") and slope (key is "slope") of the linear
        background, among other intermediate results.
    """

    if order not in [0, 1, 2]:
        raise NotImplementedError("order can only be 0, 1, or 2.")

    data = data.copy()
    background_parameters = {}
    offset = 0
    slope = 0

    data.sort_values(sort_column, inplace=True)

    field = numpy.array(data[x_column])
    moment = numpy.array(data[y_column])
    d2moment = numpy.abs(numpy.gradient(numpy.gradient(moment, field), field))
    d2moment = numpy.nan_to_num(d2moment)

    # Sort the data
    sort_idx = numpy.argsort(field)
    field = field[sort_idx[::+1]]
    moment = moment[sort_idx[::+1]]
    d2moment = d2moment[sort_idx[::+1]]

    # Determine maximum value for 2nd derivative
    maxval = numpy.nanmax(d2moment[edge_points: -edge_points]) * slope_error
    background_parameters["maxval"] = maxval

    # Linear at lower field
    idx = numpy.argmax(d2moment[edge_points:] > maxval) + edge_points
    print(idx)

    if use_field_weights:
        w = numpy.tanh(numpy.abs(field[:idx]) / 250)
    else:
        w = None

    p1 = numpy.polyfit(field[:idx], moment[:idx], order, w=w)
    # slope1, offset1 = p[0], p[1]

    background_parameters.update({"f1": field, "m1": moment, "d1": d2moment})
    background_parameters.update(
        {"F1": field[:idx], "M1": moment[:idx], "D1": d2moment[:idx]})
    background_parameters.update({"poly_params1": p1})
    # background_parameters.update({"offset1": offset1, "slope1": slope1})

    # Invert arrays
    field = field[::-1]
    moment = moment[::-1]
    d2moment = d2moment[::-1]

    # Linear at higher field
    idx = numpy.argmax(d2moment[edge_points:] > maxval) + edge_points
    print(idx)

    if use_field_weights:
        w = numpy.tanh(numpy.abs(field[:idx]) / 250)
    else:
        w = None

    p2 = numpy.polyfit(field[:idx], moment[:idx], order, w=w)
    # slope2, offset2 = p[0], p[1]

    background_parameters.update({"f2": field, "m2": moment, "d2": d2moment})
    background_parameters.update(
        {"F2": field[:idx], "M2": moment[:idx], "D2": d2moment[:idx]})
    # background_parameters.update({"offset2": offset2, "slope2": slope2})
    background_parameters.update({"poly_params2": p2})

    # Average slopes and offsets
    # slope = (slope1 + slope2) / 2
    # offset = (offset1 + offset2) / 2
    p_avg = numpy.mean([p1, p2], axis=0)

    # background_parameters.update({"offset": offset, "slope": slope})
    background_parameters.update({"poly_params": p_avg})

    # Correct the measured moment
    if keep_uncorrected:
        data[y_column + "_uncorrected"] = data[y_column]
    data[y_column] -= numpy.polyval(p_avg, data[x_column])
    # data[y_column] -= offset + data[x_column] * slope

    data.sort_values(sort_column, inplace=True)

    return data, background_parameters
