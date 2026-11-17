import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as st
from scipy.integrate import simpson
from scipy.signal import savgol_filter
from lmfit import models
from matplotlib.ticker import FuncFormatter


# ============================================================
# Build one composite model containing all components
# ============================================================

def generate_composite_model(json_file):
    """
    Build one composite lmfit model from all components in the JSON file.

    All peaks are fitted simultaneously. Peaks are not fitted or screened
    individually before construction of the composite model.
    """
    with open(json_file, "r") as f:
        model_spec = json.load(f)

    composite_model = None
    params = None

    for i, basis_func in enumerate(model_spec["models"]):
        model_type = basis_func["type"]
        params_dict = basis_func["params"]
        prefix = f"m{i}_"

        try:
            model_class = getattr(models, model_type)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown lmfit model type: {model_type}"
            ) from exc

        model = model_class(prefix=prefix)

        # Set parameter constraints before creating parameters
        for param_name, options in params_dict.items():

            hint_kwargs = {}

            if "value" in options:
                hint_kwargs["value"] = options["value"]

            if "min" in options:
                hint_kwargs["min"] = options["min"]

            if "max" in options:
                hint_kwargs["max"] = options["max"]

            if "vary" in options:
                hint_kwargs["vary"] = options["vary"]

            # Gaussian amplitude should not be negative
            if param_name == "amplitude":
                hint_kwargs["min"] = max(
                    0,
                    hint_kwargs.get("min", 0)
                )

            model.set_param_hint(param_name, **hint_kwargs)

        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model

    if composite_model is None:
        raise ValueError("No fitting components were found in the JSON file.")

    # Make all parameters only after the complete composite model is built
    params = composite_model.make_params()

    return composite_model, params, model_spec


# ============================================================
# Spectrum preprocessing
# ============================================================

def preprocess_spectrum(
    spectra,
    wavelength_start=950,
    wavelength_end=1800,
    fit_start=None,
    fit_end=None,
    smooth_window=11,
    smooth_polyorder=3
):
    """
    Smooth the spectrum, select the fitting range, and shift the minimum
    intensity to zero.
    """
    spectra = np.asarray(spectra, dtype=float).squeeze()

    if spectra.ndim != 1:
        raise ValueError(
            f"Expected a 1D spectrum, but received shape {spectra.shape}."
        )

    if not np.all(np.isfinite(spectra)):
        raise ValueError("Spectrum contains NaN or infinite values.")

    full_x = np.linspace(
        wavelength_start,
        wavelength_end,
        spectra.size
    )

    # Savitzky-Golay window must be odd and smaller than spectrum length
    if smooth_window >= spectra.size:
        smooth_window = spectra.size - 1

    if smooth_window % 2 == 0:
        smooth_window -= 1

    minimum_window = smooth_polyorder + 2

    if minimum_window % 2 == 0:
        minimum_window += 1

    smooth_window = max(smooth_window, minimum_window)

    if smooth_window >= spectra.size:
        smoothed_y = spectra.copy()
    else:
        smoothed_y = savgol_filter(
            spectra,
            window_length=smooth_window,
            polyorder=smooth_polyorder
        )

    if fit_start is None:
        fit_start = wavelength_start

    if fit_end is None:
        fit_end = wavelength_end

    fit_mask = (
        (full_x >= fit_start) &
        (full_x <= fit_end)
    )

    x_fit = full_x[fit_mask]
    y_fit = smoothed_y[fit_mask]

    if x_fit.size == 0:
        raise ValueError(
            f"No spectral points were found between "
            f"{fit_start} and {fit_end} cm^-1."
        )

    # Baseline shift used in the previous fitting workflow
    y_fit = y_fit - np.min(y_fit)

    return {
        "x": x_fit,
        "y": y_fit,
        "raw_x": full_x,
        "raw_y": spectra,
        "smoothed_full_y": smoothed_y
    }


# ============================================================
# Simultaneous fitting
# ============================================================

def fit_spectrum(spec, json_file):
    """
    Fit all JSON-defined components simultaneously.
    """
    model, params, model_spec = generate_composite_model(json_file)

    output = model.fit(
        data=spec["y"],
        params=params,
        x=spec["x"],
        nan_policy="omit",
        method="least_squares"
    )

    components = output.eval_components(x=spec["x"])

    return output, components, model_spec


# ============================================================
# Plot formatting
# ============================================================

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def plot_results(
    spec,
    output,
    components,
    component_names,
    component_colors,
    file_name,
    save_path
):
    """
    Plot measured spectrum, total fit, and all fitted components.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    x = spec["x"]
    y = spec["y"]

    # Experimental/smoothed spectrum
    ax.plot(
        x,
        y,
        "o",
        label="Data",
        markersize=4,
        alpha=0.65
    )

    # Total fitted curve
    ax.plot(
        x,
        output.best_fit,
        linewidth=3,
        label="Total fit"
    )

    # Individual fitted components
    for i, name in enumerate(component_names):
        prefix = f"m{i}_"
        component_y = components.get(prefix)

        if component_y is None:
            continue

        color = component_colors[i]

        ax.plot(
            x,
            component_y,
            linewidth=2,
            label=name,
            color=color
        )

        ax.fill_between(
            x,
            0,
            component_y,
            color=color,
            alpha=0.25
        )

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=28)
    ax.set_ylabel("Intensity", fontsize=28)

    formatter = FuncFormatter(lambda value, position: f"{value:.1f}")
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=24)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=17,
        frameon=False
    )

    clean_axes(ax)

    fig.tight_layout()

    output_file = os.path.join(
        save_path,
        f"{os.path.splitext(file_name)[0]}_fitting.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Process all spectra
# ============================================================

def process_folder(
    input_folder,
    output_csv,
    summary_csv,
    json_file,
    save_plots_folder,
    wavelength_start=950,
    wavelength_end=1800,
    fit_start=950,
    fit_end=1800,
    smooth_window=11,
    smooth_polyorder=3
):
    """
    Process every .npy spectrum in a folder using simultaneous composite
    fitting.
    """
    os.makedirs(save_plots_folder, exist_ok=True)

    component_names = [
        "Symmetric PO2",
        "Phosphate band/Collagen",
        "CO stretching",
        "Amide III",
        "Phosphate I/Amide III",
        "Amide III",
        "CH2 wagging/collagen",
        "Symmetric CH3 bending",
        "Asymmetric CH3 bending",
        "Amide II",
        "Amide I beta-sheet",
        "Amide I alpha-helix",
        "Amide I beta-turn"
    ]

    component_colors = plt.cm.tab20.colors[:len(component_names)]

    # Store integrals separately for each component
    component_integrals = {
        i: [] for i in range(len(component_names))
    }

    result_rows = []
    fit_quality_rows = []

    for root, _, files in os.walk(input_folder):

        for file_name in sorted(files):

            if not file_name.lower().endswith(".npy"):
                continue

            file_path = os.path.join(root, file_name)
            # st()
            print(f"Processing: {file_path}")

            try:
                spectra = np.load(file_path)

                spec = preprocess_spectrum(
                    spectra=spectra,
                    wavelength_start=wavelength_start,
                    wavelength_end=wavelength_end,
                    fit_start=fit_start,
                    fit_end=fit_end,
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder
                )

                output, components, model_spec = fit_spectrum(
                    spec,
                    json_file
                )

                best_values = output.best_values

                for i, basis_func in enumerate(model_spec["models"]):
                    prefix = f"m{i}_"

                    amplitude = best_values.get(
                        prefix + "amplitude",
                        np.nan
                    )

                    center = best_values.get(
                        prefix + "center",
                        np.nan
                    )

                    sigma = best_values.get(
                        prefix + "sigma",
                        np.nan
                    )

                    component_y = components.get(prefix)

                    if component_y is None:
                        integral_value = np.nan
                    else:
                        integral_value = simpson(
                            component_y,
                            x=spec["x"]
                        )

                    component_name = (
                        component_names[i]
                        if i < len(component_names)
                        else f"Component {i + 1}"
                    )

                    component_integrals.setdefault(i, []).append(
                        integral_value
                    )

                    result_rows.append({
                        "File Name": file_name,
                        "Component": component_name,
                        "Component Index": i + 1,
                        "Amplitude": amplitude,
                        "Center Value": center,
                        "Sigma": sigma,
                        "Integral Value": integral_value,
                        "Fit Success": output.success,
                        "Reduced Chi-Square": output.redchi,
                        "AIC": output.aic,
                        "BIC": output.bic
                    })

                fit_quality_rows.append({
                    "File Name": file_name,
                    "Fit Success": output.success,
                    "Chi-Square": output.chisqr,
                    "Reduced Chi-Square": output.redchi,
                    "AIC": output.aic,
                    "BIC": output.bic,
                    "Number of Evaluations": output.nfev
                })

                plot_results(
                    spec=spec,
                    output=output,
                    components=components,
                    component_names=component_names,
                    component_colors=component_colors,
                    file_name=file_name,
                    save_path=save_plots_folder
                )

                print(
                    f"Done: {file_name}; "
                    f"redchi={output.redchi:.6g}; "
                    f"success={output.success}"
                )

            except Exception as exc:
                print(f"Error processing {file_name}: {exc}")

    # Save detailed component fitting results
    output_fields = [
        "File Name",
        "Component",
        "Component Index",
        "Amplitude",
        "Center Value",
        "Sigma",
        "Integral Value",
        "Fit Success",
        "Reduced Chi-Square",
        "AIC",
        "BIC"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_fields
        )
        writer.writeheader()
        writer.writerows(result_rows)

    # Save summary statistics
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Component",
            "Component Index",
            "Average Integral",
            "Standard Deviation",
            "Number of Spectra"
        ])

        for i in sorted(component_integrals):

            values = np.asarray(
                component_integrals[i],
                dtype=float
            )

            values = values[np.isfinite(values)]

            component_name = (
                component_names[i]
                if i < len(component_names)
                else f"Component {i + 1}"
            )

            if values.size > 0:
                average = np.mean(values)
                standard_deviation = np.std(values, ddof=1) \
                    if values.size > 1 else 0.0
            else:
                average = np.nan
                standard_deviation = np.nan

            writer.writerow([
                component_name,
                i + 1,
                average,
                standard_deviation,
                values.size
            ])

    # Optional separate fitting-quality CSV
    quality_csv = os.path.join(
        os.path.dirname(output_csv),
        "fit_quality.csv"
    )

    with open(quality_csv, "w", newline="") as f:
        quality_fields = [
            "File Name",
            "Fit Success",
            "Chi-Square",
            "Reduced Chi-Square",
            "AIC",
            "BIC",
            "Number of Evaluations"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=quality_fields
        )

        writer.writeheader()
        writer.writerows(fit_quality_rows)

    print(f"Detailed fitting results saved to: {output_csv}")
    print(f"Summary statistics saved to: {summary_csv}")
    print(f"Fit-quality statistics saved to: {quality_csv}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    path = "../res/Caf2_03252026/org/mean_spec/"

    input_folder = path

    output_folder = (
        "../res/Caf2_03252026/org/"
        "spectrum_fitting_results"
    )

    os.makedirs(output_folder, exist_ok=True)

    output_csv = os.path.join(
        output_folder,
        "subspectrum_fitting_results.csv"
    )

    summary_csv = os.path.join(
        output_folder,
        "summary_statistics.csv"
    )

    save_plots_folder = os.path.join(
        output_folder,
        "plots_col_new"
    )

    json_file = "model_specification_caf2_col.json"

    process_folder(
        input_folder=input_folder,
        output_csv=output_csv,
        summary_csv=summary_csv,
        json_file=json_file,
        save_plots_folder=save_plots_folder,
        wavelength_start=950,
        wavelength_end=1800,
        fit_start=950,
        fit_end=1800,
        smooth_window=11,
        smooth_polyorder=3
    )