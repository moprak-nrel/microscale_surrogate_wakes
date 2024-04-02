import numpy as np
import xarray as xr
from rex import WindX

## Define the heights that are available in the NOW23 dataset
alts = [0, 2, 10]
alts.extend(np.arange(20, 320, 20))
alts.extend([400, 500])

year = 2020
print(alts)
buoy4 = (39.27166, -73.93917)
df = None
for i, alt in enumerate(alts):
    wtk_file = f"/kfs2/pdatasets/WIND/Mid_Atlantic/{year}/Mid_Atlantic_{year}_{alt}m.h5"
    with WindX(wtk_file) as f:
        if alt == 2:
            invMOL = np.array(
                f.get_lat_lon_ts(f"inversemoninobukhovlength_{alt}m", buoy4)
            )
            ustar = np.array(f.get_lat_lon_ts(f"friction_velocity_{alt}m", buoy4))
        elif alt == 0:
            sst = np.array(f.get_lat_lon_ts("surface_sea_temperature", buoy4)) + 273.15
            ssf = np.array(f.get_lat_lon_ts("surface_heat_flux", buoy4))
        else:
            if alt == 10:
                # Wind Speed
                ws = np.array(f.get_lat_lon_ts(f"windspeed_{alt}m", buoy4))
                # Wind Direction
                wd = np.array(f.get_lat_lon_ts(f"winddirection_{alt}m", buoy4))
                # Temperature
                t = f.get_lat_lon_ts(f"temperature_{alt}m", buoy4)
                t += 273.15  # convert to Kelvin
                # TKE
                tke = f.get_lat_lon_ts(f"turbulent_kinetic_energy_{alt}m", buoy4)
                times = f.get_lat_lon_ts("time_index", buoy4)
            else:
                ws = np.dstack([ws, f.get_lat_lon_ts(f"windspeed_{alt}m", buoy4)])
                wd = np.dstack([wd, f.get_lat_lon_ts(f"winddirection_{alt}m", buoy4)])
                t = np.dstack(
                    [t, f.get_lat_lon_ts(f"temperature_{alt}m", buoy4) + 273.15]
                )
                vt = np.dstack(
                    [
                        t,
                        f.get_lat_lon_ts(f"virtual_potential_temperature_{alt}m", buoy4)
                        + 273.15,
                    ]
                )
                tke = np.dstack(
                    [tke, f.get_lat_lon_ts(f"turbulent_kinetic_energy_{alt}m", buoy4)]
                )

data = xr.Dataset(
    data_vars=dict(
        wind_speed=(["time", "height"], ws[0]),
        wind_direction=(["time", "height"], wd[0]),
        temperature=(["time", "height"], t[0]),
        virtual_temperature=(["time", "height"], t[0]),
        tke=(["time", "height"], tke[0]),
        sst=(["time"], sst),
        surface_heat_flux=(["time"], ssf),
        invMOL=(["time"], invMOL),
        ustar=(["time"], ustar),
    ),
    coords=dict(
        time=times,
        height=np.array(alts[2:]),
    ),
)
data_pd = data.to_dataframe()
data_pd.to_pickle("wrf_df.pkl", protocol=-1)
