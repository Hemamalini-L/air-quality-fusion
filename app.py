"""
Streamlit AQI Multi-Source Capture & Fusion Demo
Features:
 - Fetches official station data from OpenAQ (public)
 - Simulates mobile/edge IoT sensors around a selected location
 - Fuses multiple sources using inverse-variance weighting
 - Map visualization, time-series charts, CSV export, simple forecasting & anomaly detection
Author: Generated for hackathon by ChatGPT
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pydeck as pdk
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="AQI Multi-Source Capture (Demo)", layout="wide")

# -------------------------
# Helpers
# -------------------------
OPENAQ_BASE = "https://api.openaq.org/v2/measurements"

@st.cache_data(ttl=300)
def fetch_openaq(lat, lon, radius=10000, limit=100):
    """
    Fetch latest measurements around a lat/lon using OpenAQ (no API key).
    radius in meters.
    """
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius,
        "limit": limit,
        "sort": "desc",
        "order_by": "date"
    }
    r = requests.get(OPENAQ_BASE, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    records = []
    for m in data.get("results", []):
        records.append({
            "location": m.get("location"),
            "parameter": m.get("parameter"),
            "value": m.get("value"),
            "unit": m.get("unit"),
            "date_utc": m.get("date", {}).get("utc"),
            "latitude": m.get("coordinates", {}).get("latitude"),
            "longitude": m.get("coordinates", {}).get("longitude"),
            "source": "openaq"
        })
    return pd.DataFrame(records)

def simulate_sensors(center_lat, center_lon, n=8, spread_m=800):
    """
    Simulate n sensors randomly distributed around center point.
    Each sensor returns PM2.5 and PM10 with small random bias and noise.
    spread_m ~ typical dispersion in meters
    """
    sensors = []
    for i in range(n):
        # convert meters to deg approx (1 deg ~ 111km)
        dx = np.random.normal(scale=spread_m/111000)
        dy = np.random.normal(scale=spread_m/111000)
        lat = center_lat + dx
        lon = center_lon + dy
        # baseline values (simulate near real-world range)
        base_pm25 = np.random.uniform(10, 120)  # µg/m3
        base_pm10 = base_pm25 + np.random.uniform(5, 40)
        # per-sensor bias and noise
        bias25 = np.random.normal(scale=3)
        bias10 = np.random.normal(scale=5)
        noise25 = np.random.normal(scale=2)
        noise10 = np.random.normal(scale=4)
        sensors.append({
            "sensor_id": f"sim_{i+1}",
            "latitude": lat,
            "longitude": lon,
            "pm25": max(0, base_pm25 + bias25 + noise25),
            "pm10": max(0, base_pm10 + bias10 + noise10),
            "timestamp_utc": datetime.utcnow().isoformat(),
            "source": "sim_sensor"
        })
    return pd.DataFrame(sensors)

def pm25_to_aqi(pm25):
    """
    Simple approximate conversion (US EPA breakpoints).
    Not exact — used for demo only.
    """
    c = float(pm25)
    # breakpoints (concise)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (cl, ch, il, ih) in breakpoints:
        if cl <= c <= ch:
            aqi = ((ih - il) / (ch - cl)) * (c - cl) + il
            return round(aqi)
    return 500

def fuse_sources(openaq_df, sensors_df):
    """
    Fuse PM2.5 readings from official sources (openaq) and sensors.
    Use inverse-variance weighting: weight = 1 / variance_estimate.
    For demo, we assign variances heuristically: official stations more trusted.
    """
    records = []
    now = datetime.utcnow().isoformat()
    # gather pm25 readings from openaq (filter parameter pm25)
    oa_pm25 = openaq_df[openaq_df['parameter']=='pm25'] if not openaq_df.empty else pd.DataFrame()
    # convert openaq rows to unified format
    oa_rows = []
    for _, r in oa_pm25.iterrows():
        oa_rows.append({
            "id": f"openaq_{r.location}_{_}",
            "lat": r.latitude,
            "lon": r.longitude,
            "pm25": r.value,
            "source": "openaq",
            "var": max(1.0, abs(r.value)*0.05 + 2.0)  # heuristic variance
        })
    # sensors
    sensor_rows = []
    for _, s in sensors_df.iterrows():
        sensor_rows.append({
            "id": s.sensor_id,
            "lat": s.latitude,
            "lon": s.longitude,
            "pm25": s.pm25,
            "source": "sim_sensor",
            "var": max(2.0, abs(s.pm25)*0.10 + 1.0)  # sensors less trusted
        })
    all_rows = oa_rows + sensor_rows
    if not all_rows:
        return None, pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # compute inverse-variance weights
    df['inv_var'] = 1.0 / (df['var'] ** 2)
    wsum = df['inv_var'].sum()
    df['weight'] = df['inv_var'] / wsum
    fused_pm25 = float((df['pm25'] * df['weight']).sum())
    fused_aqi = pm25_to_aqi(fused_pm25)
    fused_record = {
        "timestamp_utc": now,
        "fused_pm25": fused_pm25,
        "fused_aqi": fused_aqi,
        "component_count": len(df)
    }
    return fused_record, df

def detect_anomalies(series, z_thresh=2.5):
    mean = series.mean()
    std = series.std(ddof=0) if series.size>1 else 0.0
    if std==0:
        return pd.Series([False]*len(series))
    z = (series - mean) / std
    return z.abs() > z_thresh

# -------------------------
# UI - Sidebar
# -------------------------
st.title("AQI Multi-Source Capture & Fusion — Streamlit Demo 🛰️📡")
st.markdown("Prototype showing how to combine official stations + mobile IoT sensors + fusion & simple forecasting.")

with st.sidebar:
    st.header("Configuration")
    city_lat = st.number_input("Center latitude", value=13.0827, format="%.6f", help="Default: Chennai")
    city_lon = st.number_input("Center longitude", value=80.2707, format="%.6f")
    radius = st.slider("Search radius (meters) for official stations", min_value=1000, max_value=50000, value=8000, step=1000)
    n_sensors = st.slider("Simulated IoT sensors", min_value=3, max_value=30, value=8)
    refresh_sec = st.number_input("Auto-refresh interval (seconds)", min_value=0, max_value=60, value=10)
    st.write(" ")
    if st.button("Fetch latest & simulate once"):
        st.session_state["fetch_now"] = True
    st.markdown("---")
    st.write("Export & Settings")
    enable_export = st.checkbox("Show CSV export", value=True)

# -------------------------
# Main
# -------------------------
col1, col2 = st.columns((1.2, 1))

with col1:
    st.subheader("Map: sensors + official stations")
    # Fetch data
    try:
        openaq_df = fetch_openaq(city_lat, city_lon, radius=radius, limit=100)
    except Exception as e:
        st.error(f"OpenAQ fetch failed: {e}")
        openaq_df = pd.DataFrame()
    sensors_df = simulate_sensors(city_lat, city_lon, n=n_sensors, spread_m=radius/4)

    fused_record, fused_df = fuse_sources(openaq_df, sensors_df)
    # Map layer assembly
    points = []
    # official
    if not openaq_df.empty:
        # pick latest pm25 per location
        latest = openaq_df[openaq_df['parameter']=='pm25'].dropna(subset=['latitude','longitude']).copy()
        if not latest.empty:
            latest = latest.drop_duplicates(subset=['location'])
            for _, r in latest.iterrows():
                points.append({
                    "lat": r.latitude, "lon": r.longitude,
                    "type": "Official station",
                    "label": f"{r.location}: {r.value} {r.unit}"
                })
    # sensors
    for _, s in sensors_df.iterrows():
        points.append({
            "lat": s.latitude, "lon": s.longitude,
            "type": "Sensor",
            "label": f"{s.sensor_id}: PM2.5 {s.pm25:.1f}"
        })

    if points:
        map_df = pd.DataFrame(points)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_fill_color='[255, 140, 0, 160]',
            get_radius=200,
            pickable=True
        )
        view = pdk.ViewState(latitude=city_lat, longitude=city_lon, zoom=11, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{type}\n{label}"}))
    else:
        st.info("No geolocated points to map.")

    # show table of fused components
    st.subheader("Fused components (weights)")
    if not fused_df.empty:
        display_df = fused_df[['id','lat','lon','pm25','source','var','weight']].copy()
        display_df['weight'] = display_df['weight'].round(3)
        st.dataframe(display_df)
    else:
        st.write("No components available (no PM2.5 data found).")

with col2:
    st.subheader("Live fused AQI & Trend")
    if fused_record:
        st.metric("Fused PM2.5 (µg/m³)", f"{fused_record['fused_pm25']:.1f}")
        st.metric("Fused AQI (approx)", fused_record['fused_aqi'])
    else:
        st.write("No fused record available yet.")

    # simulate a time-series buffer in session state
    if "timeseries" not in st.session_state:
        st.session_state.timeseries = pd.DataFrame({
            "timestamp_utc": [ (datetime.utcnow() - timedelta(minutes=i)).isoformat() for i in range(30,0,-1)],
            "fused_pm25": np.nan,
            "fused_aqi": np.nan
        })
    # append latest fused
    if fused_record:
        new_row = {"timestamp_utc": fused_record['timestamp_utc'], "fused_pm25": fused_record['fused_pm25'], "fused_aqi": fused_record['fused_aqi']}
        st.session_state.timeseries = pd.concat([st.session_state.timeseries, pd.DataFrame([new_row])], ignore_index=True).tail(120)

    ts = st.session_state.timeseries.copy()
    ts['ts_dt'] = pd.to_datetime(ts['timestamp_utc'])

    # Plot PM2.5 time series
    fig1 = px.line(ts, x='ts_dt', y='fused_pm25', title="Fused PM2.5 over time", labels={"ts_dt":"Time","fused_pm25":"PM2.5 (µg/m³)"})
    st.plotly_chart(fig1, use_container_width=True)

    # Anomaly detection
    ts_nonnull = ts.dropna(subset=['fused_pm25'])
    if not ts_nonnull.empty:
        anomalies = detect_anomalies(ts_nonnull['fused_pm25'])
        if anomalies.any():
            anom_times = ts_nonnull['ts_dt'][anomalies].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            st.warning(f"Anomaly detected at: {', '.join(anom_times)} (z-score threshold).")

    # Simple forecast: rolling mean for next 3 points (demo)
    if not ts_nonnull.empty:
        rolling = ts_nonnull['fused_pm25'].rolling(window=5, min_periods=1).mean().iloc[-1]
        forecast = [rolling * (1 + 0.01*i) for i in range(1,4)]
        st.subheader("Short-term forecast (rolling mean)")
        st.write(pd.DataFrame({
            "t+1": [forecast[0]],
            "t+2": [forecast[1]],
            "t+3": [forecast[2]],
        }).T.rename(columns={0:"predicted_pm25(µg/m3)"}))

# -------------------------
# Export & Data table
# -------------------------
st.markdown("---")
st.header("Fused Data Export & Raw Sources")
if fused_record:
    fused_df_out = pd.DataFrame([fused_record])
    st.dataframe(pd.concat([fused_df_out.reset_index(drop=True)], axis=1))
    if enable_export:
        csv = fused_df_out.to_csv(index=False)
        st.download_button("Download fused data (CSV)", csv, file_name="fused_aqi.csv", mime="text/csv")
else:
    st.write("No fused result to export.")

st.subheader("Raw simulated sensor readings")
st.dataframe(sensors_df[['sensor_id','latitude','longitude','pm25','pm10','timestamp_utc']])

st.subheader("OpenAQ raw hits (pm25 filtered)")
if not openaq_df.empty:
    st.dataframe(openaq_df[openaq_df['parameter']=='pm25'][['location','value','unit','date_utc','latitude','longitude']].head(20))
else:
    st.write("No OpenAQ results in this radius / location.")

# -------------------------
# Auto refresh
# -------------------------
if refresh_sec > 0:
    st.experimental_rerun() if st.session_state.get("fetch_now", False) else time.sleep(0.001)
