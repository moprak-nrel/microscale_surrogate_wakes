import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import wrf_functions

# Specify a matplotlib style
plt.style.use("./project.mplstyle")
# Default directory for data
data_dir = "./data/"
plot_dir = "./plots/"

data = pd.read_pickle(f"{data_dir}/wrf_df.pkl")
data["time"] = data.index.get_level_values(
    0
)  # This is just because multiindex is painful
data["height"] = data.index.get_level_values(
    1
)  # This is just because multiindex is painful

## Convert wind speed/direction into u,v values
data["u"] = np.sin(np.radians(data.wind_direction)) * data.wind_speed
data["v"] = np.cos(np.radians(data.wind_direction)) * data.wind_speed

criteria_col = "is_neutral"
neutral = data.groupby(level=0).apply(wrf_functions.is_neutral)
neutral.name = criteria_col
neutral = pd.DataFrame(neutral)
data = data.join(neutral)

duration_col = "neutral_duration"
data[duration_col] = 0
data = wrf_functions.add_duration(data, criteria_col, duration_col)

neutral_durations = sorted(data[duration_col].unique())


## This is a hand selected time to avoid LLJs
d = neutral_durations[2]
selected_time = data[data[duration_col] == d].time.unique()[0]
df = data.loc[selected_time : selected_time + pd.Timedelta(d, "s")]
wrf_functions.plot_slice(df, 'wind_speed', 'Wind Speed (m/s)', plot_dir)
wrf_functions.plot_slice(df, 'temperature', r'$\theta (\mathrm{K})$', plot_dir, "magma")
mean_df = df.groupby(level=1).mean()

## Averaged line plots
plot_labels = {
    "wind_speed": "Wind speed (m/s)",
    "tke": "TKE",
    "temperature": r"$\theta (\mathrm{K})$",
}
with PdfPages(f"{plot_dir}/mean_line_plots.pdf") as pdf:
    for k in plot_labels:
        plt.figure(k)
        plt.plot(mean_df[k].values, mean_df.height.values, label=plot_labels[k])
        plt.ylabel("Height (m AMSL)")
        plt.xlabel(plot_labels[k])
        pdf.savefig()
plt.close("all")
