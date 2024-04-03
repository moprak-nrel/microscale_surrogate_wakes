import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("./project.mplstyle")


# Setting basic criteria for neutral profile detection
# Current criteria 7 m/s < windspeed < 10 m/s
# Monin-Obukhov Length > 1000
def is_neutral(d, vmin=7, vmax=10, MOL_min=1000):
    profile = d.loc[d.name]
    is_valid_velocity = vmax > profile.wind_speed[140.0] > vmin
    is_valid_MOL = abs(profile.invMOL[140.0]) < 1.0 / MOL_min
    is_neutral = int(is_valid_velocity and is_valid_MOL)
    return is_neutral


# Check for contiguous occurrence of criteria and add the duration to an output column
def add_duration(data, criteria_col, output_col):
    neutral_times = data[data[criteria_col] == 1].index.get_level_values("time")
    seen = []
    dt = pd.Timedelta("5m")
    for time in neutral_times:
        if time not in seen:
            current = time
            seen.append(time)
            current_duration = dt
            if (current + dt) in neutral_times:
                while data.loc[current + dt, criteria_col].values[0] != 0:
                    current_duration += dt
                    current += dt
                    seen.append(current)
            else:
                current_duration += dt
                current += dt
            data.loc[time, output_col] = current_duration.total_seconds()
    return data


##### Plotting functions #####
def plot_slice(df, key, label,  plot_dir, cmap = "viridis"):
    N = len(df)
    na = len(df.index.levels[1])
    nt = N // na
    x = ((df.time.values - df.time.values[0]) / 60e9).reshape((nt, na))
    y = df.height.values.reshape((nt, na))
    z = df[key].values.reshape((nt, na))
    f = plt.figure()
    c = plt.contourf(x, y, z, cmap = cmap)
    f.colorbar(c, label=label)
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Height (m AMSL)")
    f.savefig(f"{plot_dir}/{key}_contour_{df.time.iloc[0].date()}.pdf")
