from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np

def build_synthetic_timeline(
    num_timesteps: int,
    start_datetime: str,
    points_per_hour: int,
) -> List[Tuple[str, str]]:
\
\
\
\
       
    start = datetime.strptime(start_datetime.strip(), "%Y-%m-%d %H:%M")
    interval_minutes = max(1, 60 // max(points_per_hour, 1))
    step = timedelta(minutes=interval_minutes)
    timeline = []
    for i in range(num_timesteps):
        dt = start + i * step
        timeline.append((dt.strftime("%Y%m%d"), "{}:{}".format(dt.hour, dt.strftime("%M"))))
    return timeline

def load_event_timeline(event_labels_filename: str) -> Tuple[List[Tuple[str, str]], np.ndarray, List[str]]:
\
\
\
\
\
\
\
       
    with open(event_labels_filename, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    sensor_names = header[2:]
    timeline = [(row[0], row[1]) for row in rows]
    event_matrix = np.array(
        [[int(float(v)) for v in row[2:]] for row in rows],
        dtype=np.int8,
    )
    return timeline, event_matrix, sensor_names

def _format_time_label(date: str, time: str) -> str:
    date_text = str(date)
    if len(date_text) == 8 and date_text.isdigit():
        date_text = "{}-{}-{}".format(date_text[:4], date_text[4:6], date_text[6:8])
    return "{} {}".format(date_text, time)

def _parse_datetime(date: str, time: str) -> Optional[datetime]:
    date_text = str(date)
    if len(date_text) == 8 and date_text.isdigit():
        year = int(date_text[:4])
        month = int(date_text[4:6])
        day = int(date_text[6:8])
    else:
        return None

    time_text = str(time).strip()
    if ":" in time_text:
        hour_str, minute_str = time_text.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_text)
        minute = 0
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None

def _weekday_label(date: str, time: str) -> str:
    dt = _parse_datetime(date, time)
    if dt is None:
        return "unknown"
    return dt.strftime("%A")

def _format_duration_minutes(total_minutes: int) -> str:
    if total_minutes % 60 == 0:
        hours = total_minutes // 60
        return "{} hour{}".format(hours, "" if hours == 1 else "s")
    if total_minutes >= 60:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return "{} h {} min".format(hours, minutes)
    return "{} min".format(total_minutes)

def _build_period_text(
    timeline: Sequence[Tuple[str, str]],
    input_start: int,
    input_end: int,
) -> str:
    start_date, start_time = timeline[input_start]
    end_idx = min(input_end - 1, len(timeline) - 1)
    end_date, end_time = timeline[end_idx]
    period_text = _format_time_label(start_date, start_time)
    if (end_date, end_time) != (start_date, start_time):
        period_text += " to " + _format_time_label(end_date, end_time)
    return period_text

def build_prompt_for_window(
    timeline: Sequence[Tuple[str, str]],
    event_matrix: np.ndarray,
    sensor_names: Sequence[str],
    input_start: int,
    input_end: int,
    max_sensors: int = 10,
    num_vertices: int = 0,
    num_for_predict: int = 0,
    points_per_hour: int = 4,
    dataset_display_name: str = "Traffic",
) -> str:
\
\
\
\
\
       
    if input_start < 0 or input_end <= input_start:
        return ""

    interval_minutes = max(1, 60 // max(points_per_hour, 1))
    len_input = input_end - input_start
    forecast_minutes = num_for_predict * interval_minutes

    start_date, start_time = timeline[input_start]
    period_text = _build_period_text(timeline, input_start, input_end)
    weekday = _weekday_label(start_date, start_time)

    base_prompt = (
        "{} traffic forecast. Input period: {} ({} steps, {}-min interval, {} sensors). "
        "Forecast horizon: next {} steps ({}). Day: {}."
    ).format(
        dataset_display_name,
        period_text,
        len_input,
        interval_minutes,
        num_vertices,
        num_for_predict,
        _format_duration_minutes(forecast_minutes),
        weekday,
    )

    window = event_matrix[input_start:input_end]
    if window.size == 0 or window.sum() == 0:
        return base_prompt + " Event: none."

    affected = set()
    for row in window:
        cols = np.where(row > 0)[0]
        for col in cols:
            affected.add(sensor_names[int(col)])

    if not affected:
        return base_prompt + " Event: none."

    sensors_sorted = sorted(affected)
    shown = sensors_sorted[:max_sensors]
    sensors_text = ", ".join(shown)
    if len(sensors_sorted) > max_sensors:
        sensors_text += ", and {} more sensors".format(len(sensors_sorted) - max_sensors)

    event_prompt = (
        " Event: traffic incident affecting {} sensors, including {}."
    ).format(len(sensors_sorted), sensors_text)
    return base_prompt + event_prompt

def build_sample_event_prompts(
    sample_timestamps: np.ndarray,
    timeline: Sequence[Tuple[str, str]],
    event_matrix: np.ndarray,
    sensor_names: Sequence[str],
    len_input: int,
    points_per_hour: int,
    num_of_hours: int,
    max_sensors: int = 10,
    num_vertices: int = 0,
    num_for_predict: int = 0,
    dataset_display_name: str = "Traffic",
) -> np.ndarray:
\
\
\
\
       
    offset = points_per_hour * num_of_hours
    prompts = []
    for ts in sample_timestamps.reshape(-1):
        idx = int(ts)
        input_start = idx - offset
        input_end = input_start + len_input
        prompt = build_prompt_for_window(
            timeline,
            event_matrix,
            sensor_names,
            input_start,
            input_end,
            max_sensors=max_sensors,
            num_vertices=num_vertices,
            num_for_predict=num_for_predict,
            points_per_hour=points_per_hour,
            dataset_display_name=dataset_display_name,
        )
        prompts.append(prompt)
    return np.array(prompts, dtype=object)

def count_prompts_with_events(prompts: np.ndarray) -> int:
    return int(np.sum(["traffic incident" in str(p) for p in prompts]))

def resolve_event_labels_path(config, base_dir: str) -> Optional[str]:
    if not config.has_option("Data", "event_labels_filename"):
        return None
    path_value = config.get("Data", "event_labels_filename").strip()
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(base_dir, path_value))

def resolve_context_prompt_settings(config, base_dir: str, dataset_name: str):
\
\
\
\
       
    data_config = config["Data"]
    enabled = False
    if config.has_option("Data", "use_context_prompt"):
        enabled = data_config.getboolean("use_context_prompt", fallback=False)
    elif config.has_option("Data", "use_event_prompt"):
        enabled = data_config.getboolean("use_event_prompt", fallback=False)

    start_datetime = data_config.get("prompt_start_datetime", "").strip()
    if not enabled or not start_datetime:
        return None

    display_name = data_config.get("prompt_dataset_name", dataset_name).strip() or dataset_name
    return {
        "start_datetime": start_datetime,
        "dataset_display_name": display_name,
    }
