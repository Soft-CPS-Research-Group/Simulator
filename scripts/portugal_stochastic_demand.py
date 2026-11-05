"""Activity-conditioned aggregate demand with persistent, finite load episodes.

Parameters are developed against household-level statistics of raw SHEERM
meters, not by copying measured traces. End-use budgets remain inventory priors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from portugal_activity import CONFIG, N23, rng_for


def activity_intensity(root, mid, year, size):
    cfg = CONFIG["demand"]
    rng = rng_for(root, mid, year, "daily_intensity")
    rho, sigma = cfg["daily_persistence"], cfg["daily_log_sigma"]
    innovations = rng.normal(0, sigma*np.sqrt(1-rho*rho), size)
    innovations[0] = rng.normal(0, sigma)
    state = lfilter([1], [1, -rho], innovations)
    return np.exp(state-.5*sigma*sigma)


def residential_shape(root, member, original, calendar, daily, occupancy):
    """Random load episodes retain chronology, presence and seasonal priors."""
    cfg = CONFIG["demand"]
    clock = pd.DatetimeIndex(calendar.timestamp_utc).tz_convert("Europe/Lisbon")
    slot = clock.hour*4+clock.minute//15
    original = np.asarray(original, dtype=float)
    # Remove inherited day templates and modulo schedules while retaining each
    # household's seasonal/civil-hour prior from its inventory model.
    means = pd.Series(original).groupby([clock.year, clock.month, slot]).mean()
    patterns = {key: series.droplevel([0,1]).reindex(range(96)).interpolate(limit_direction="both").to_numpy()
                for key, series in means.groupby(level=[0,1])}
    smooth = np.empty(len(clock))
    intensity = np.ones(len(clock))
    for year, group in daily.groupby("year"):
        rng = rng_for(root, member.member_id, year, "demand_timing")
        for d in group.itertuples():
            a,b=int(d.start_time_step),int(d.end_time_step)
            month=pd.Timestamp(d.date).month
            prior=patterns[(year,month)]
            shift=rng.normal(0,cfg["timing_jitter_hours"]*4)
            smooth[a:b]=np.interp((slot[a:b]-shift)%96,np.arange(97),np.r_[prior,prior[0]])
            intensity[a:b]=d.activity_intensity
    result=np.empty(len(clock))
    for year,lo,hi in ((2023,0,N23),(2024,N23,len(clock))):
        rng=rng_for(root,member.member_id,year,"load_episodes")
        prior=smooth[lo:hi]
        rate=cfg["episodes_per_day"]/96
        # Finite persistent episodes: compound-Poisson arrivals and a decaying
        # activity state. This is a 15-minute aggregate model, not device physics.
        weights=prior/max(prior.mean(),1e-12)
        counts=rng.poisson(rate*weights)
        impulses=rng.gamma(counts*1.5,1/1.5)/rate
        rho=cfg["episode_persistence"]
        episodes=lfilter([1-rho],[1,-rho],impulses)
        fraction=cfg["episode_fraction"]
        result[lo:hi]=((1-fraction)*prior+fraction*prior.mean()*episodes)*intensity[lo:hi]
    return result*(.25+.75*occupancy)


def bounded_energy_allocation(values, cap, total):
    """Conserve energy under a physical envelope; no silent energy clipping."""
    values=np.asarray(values,dtype=float)
    cap=np.maximum(np.asarray(cap,dtype=float),0.)
    if total <= 0: return np.zeros_like(values)
    assert cap.sum() >= total-1e-8, "Annual energy exceeds available envelope"
    weights=np.maximum(values,1e-12)
    low,high=0.,max(1.,total/weights.sum())
    while np.minimum(high*weights,cap).sum()<total:
        high*=2
    for _ in range(55):
        mid=(low+high)/2
        if np.minimum(mid*weights,cap).sum()<total: low=mid
        else: high=mid
    result=np.minimum(high*weights,cap)
    assert abs(result.sum()-total)<1e-6
    return result
