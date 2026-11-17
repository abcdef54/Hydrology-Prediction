"""Stage 2 Visualization and Report Generation.

Generates:
- 7 comprehensive figures (DEM map, spatial correlation matrix, lag cross-correlations,
  lag-window heatmaps, conditional analysis, leave-one-out event stability, station selection).
- Machine-readable CSV tables.
- Comprehensive markdown report adhering to EDA2.md master plan.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def generate_stage2_figures(
    dem_df: pd.DataFrame,
    rain_stations: pd.DataFrame,
    water_stations: pd.DataFrame,
    p_corr: pd.DataFrame,
    clusters_df: pd.DataFrame,
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    cond_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    multi_eval_df: pd.DataFrame,
    output_dir: Path
) -> List[Path]:
    """Generate all 7 required Stage 2 visualization figures."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    # Styling defaults
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    # -------------------------------------------------------------
    # Figure 1: DEM Spatial Map of Rain and Water Stations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Plot rain stations colored by elevation
    rain_meta = rain_stations.copy()
    rain_meta['station_id'] = rain_meta['station_id'].astype(str)
    elev_map = dem_df.set_index(dem_df['station_id'].astype(str))['elevation_m'].to_dict()
    rain_meta['elevation_m'] = rain_meta['station_id'].map(elev_map).fillna(0.0)

    sc = ax.scatter(
        rain_meta['longitude'], rain_meta['latitude'],
        c=rain_meta['elevation_m'], cmap='terrain',
        s=80, edgecolors='black', linewidth=0.8, alpha=0.9, zorder=3,
        label='Rain Gauge (55)'
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Terrain Elevation (FABDEM, meters)', fontsize=10)

    # Plot water level stations
    ax.scatter(
        water_stations['longitude'], water_stations['latitude'],
        marker='s', s=100, color='crimson', edgecolors='white', linewidth=1.2,
        zorder=4, label='Water Level Gauge (19)'
    )

    # Annotate key stations
    key_annotations = {
        '0074647403': ('Dã Viên (Huế)', (10, 5)),
        '0074647705': ('Ngã Ba Sình', (10, -10)),
        '0074647404': ('Tứ Phú (Sông Bồ)', (-70, 5)),
        '0074647606': ('Cầu Huỳnh Liên (Ô Lâu)', (-90, 5)),
        '0074647402': ('Thuận An (Tidal)', (10, 5)),
        '750062': ('Hồng Thái (A Lưới)', (-95, -5)),
        '750007': ('Nhâm (A Lưới)', (-80, 5)),
        '750041': ('Bạch Mã (750041)', (10, -5)),
        '750014': ('Hồ Phú Bài', (10, 5)),
        '750061': ('Bình Điền', (-60, -10)),
    }

    all_stations = pd.concat([
        rain_stations[['station_id', 'longitude', 'latitude']],
        water_stations[['station_id', 'longitude', 'latitude']]
    ]).set_index('station_id')

    for sid, (lbl, offset) in key_annotations.items():
        if sid in all_stations.index:
            row = all_stations.loc[sid]
            lon = row['longitude'].iloc[0] if isinstance(row['longitude'], pd.Series) else row['longitude']
            lat = row['latitude'].iloc[0] if isinstance(row['latitude'], pd.Series) else row['latitude']
            ax.annotate(
                lbl, (lon, lat),
                xytext=offset, textcoords='offset points',
                fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=0.6)
            )

    ax.set_title('Thừa Thiên Huế Hydrological Network — Rain Gauges & Water Stations over Topography', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', framealpha=0.9)

    fig1_path = figures_dir / '01_dem_station_network_map.png'
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=160)
    plt.close(fig)
    generated_paths.append(fig1_path)

    # -------------------------------------------------------------
    # Figure 2: 55x55 Spatial Rainfall Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    # Order stations by cluster
    clust_order = clusters_df.sort_values(['cluster_id', 'elevation_m'])['station_id'].tolist()
    valid_order = [s for s in clust_order if s in p_corr.columns]
    corr_ordered = p_corr.loc[valid_order, valid_order]

    im = ax.imshow(corr_ordered.values, cmap='YlGnBu', vmin=0.0, vmax=1.0, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Pearson Correlation (Hourly Rainfall)', fontsize=10)

    # Cluster boundary lines
    clust_sizes = clusters_df.groupby('cluster_id').size()
    boundaries = np.cumsum(clust_sizes)[:-1]
    for b in boundaries:
        ax.axhline(b - 0.5, color='crimson', lw=1.2, linestyle='--')
        ax.axvline(b - 0.5, color='crimson', lw=1.2, linestyle='--')

    ax.set_title('Pairwise Spatial Correlation Matrix across 55 Rain Gauges (Grouped by Hydrological Zone)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Rain Stations (Ordered by Hydrological Zone)', fontsize=10)
    ax.set_ylabel('Rain Stations (Ordered by Hydrological Zone)', fontsize=10)

    fig2_path = figures_dir / '02_spatial_rain_correlation_matrix.png'
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=160)
    plt.close(fig)
    generated_paths.append(fig2_path)

    # -------------------------------------------------------------
    # Figure 3: Lag Cross-Correlation Profiles for Dã Viên
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    lags = list(range(0, 49, 2))
    
    dv_target = targets.get('0074647403')
    key_stations_plot = [
        ('750062', 'Hồng Thái (A Lưới)', '#1f77b4', '-'),
        ('750007', 'Nhâm (A Lưới)', '#aec7e8', '--'),
        ('750041', 'Bạch Mã (750041)', '#2ca02c', '-'),
        ('750030', 'VQG Bạch Mã (750030)', '#98df8a', '--'),
        ('750014', 'Hồ Phú Bài', '#ff7f0e', '-'),
        ('750061', 'Bình Điền', '#9467bd', '-'),
    ]

    if dv_target is not None:
        yh = dv_target['water_level_mm']
        ydh6 = dv_target['dh_p6h']
        ydh12 = dv_target['dh_p12h']

        for sid, sname, color, ls in key_stations_plot:
            if sid not in accum_standard[6].columns:
                continue
            rain_6h = accum_standard[6][sid]

            ch_list = [float(yh.corr(rain_6h.shift(l))) for l in lags]
            c6_list = [float(ydh6.corr(rain_6h.shift(l))) for l in lags]
            c12_list = [float(ydh12.corr(rain_6h.shift(l))) for l in lags]

            axes[0].plot(lags, ch_list, label=sname, color=color, linestyle=ls, lw=1.8)
            axes[1].plot(lags, c6_list, label=sname, color=color, linestyle=ls, lw=1.8)
            axes[2].plot(lags, c12_list, label=sname, color=color, linestyle=ls, lw=1.8)

        axes[0].set_title('Absolute Level H(t)\n[Full period; rainfall window 6h]', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Rainfall Lag L (hours)', fontsize=10)
        axes[0].set_ylabel('Pearson r', fontsize=10)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].set_title('Rise Target ΔH +6h\n[Full-period descriptive correlation]', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Rainfall Lag L (hours)', fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].axvline(0, color='gray', linestyle=':', lw=1.2)

        axes[2].set_title('Rise Target ΔH +12h\n[Full-period descriptive correlation]', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Rainfall Lag L (hours)', fontsize=10)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].axvline(0, color='gray', linestyle=':', lw=1.2)

        axes[2].legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    fig.suptitle('Dã Viên Water Station: Lagged Cross-Correlations with 6-Hour Rainfall Accumulation', fontsize=12, fontweight='bold', y=1.02)
    fig3_path = figures_dir / '03_lag_cross_correlation_profiles.png'
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    generated_paths.append(fig3_path)

    # -------------------------------------------------------------
    # Figure 4: 2D Heatmaps of Correlation (Lag vs Window)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    windows = [1, 3, 6, 12, 24]
    lags_hm = list(range(0, 25, 2))

    if dv_target is not None and '750062' in accum_standard[1].columns:
        # Hong Thai for dh_p6h and dh_p12h
        grid6 = np.zeros((len(windows), len(lags_hm)))
        grid12 = np.zeros((len(windows), len(lags_hm)))

        for i, w in enumerate(windows):
            rw = accum_standard[w]['750062']
            for j, l in enumerate(lags_hm):
                rwl = rw.shift(l)
                grid6[i, j] = float(dv_target['dh_p6h'].corr(rwl))
                grid12[i, j] = float(dv_target['dh_p12h'].corr(rwl))

        im1 = axes[0].imshow(grid6, cmap='Blues', aspect='auto', origin='lower')
        axes[0].set_title('Hồng Thái vs Dã Viên ΔH +6h Correlation', fontsize=11, fontweight='bold')
        axes[0].set_yticks(range(len(windows)), [f'{w}h' for w in windows])
        axes[0].set_xticks(range(len(lags_hm)), lags_hm)
        axes[0].set_ylabel('Accumulation Window W', fontsize=10)
        axes[0].set_xlabel('Lag L (hours)', fontsize=10)
        fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Pearson r')

        im2 = axes[1].imshow(grid12, cmap='Blues', aspect='auto', origin='lower')
        axes[1].set_title('Hồng Thái vs Dã Viên ΔH +12h Correlation', fontsize=11, fontweight='bold')
        axes[1].set_yticks(range(len(windows)), [f'{w}h' for w in windows])
        axes[1].set_xticks(range(len(lags_hm)), lags_hm)
        axes[1].set_ylabel('Accumulation Window W', fontsize=10)
        axes[1].set_xlabel('Lag L (hours)', fontsize=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8, label='Pearson r')

    fig.suptitle('Full-period Correlation by Window and Lag (Hồng Thái -> Dã Viên)', fontsize=12, fontweight='bold', y=1.02)
    fig4_path = figures_dir / '04_lag_heatmaps_dh6_dh12.png'
    fig.tight_layout()
    fig.savefig(fig4_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    generated_paths.append(fig4_path)

    # -------------------------------------------------------------
    # Figure 5: Conditional Relationship Analysis
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sub_cond = cond_df[cond_df['rain_station_id'].isin(['750062', '750007', '750041'])].copy()
    if not sub_cond.empty:
        piv_cond = sub_cond.pivot(index='condition', columns='rain_station_name', values='corr_dh_p12h')
        # Order conditions logically
        order = [
            'All Hours (Unconditional)',
            'Flood Season (Sep-Dec)',
            'Dry Season (Jan-Aug)',
            'Rising Limb (dH_-3h > 0)',
            'Falling Limb (dH_-3h < 0)',
            'Rain >= 1mm',
            'Rain >= 5mm',
            'Rain >= 10mm',
            'Rapid Rise (|dH_+12h| >= 200mm)',
            'Normal Periods (|dH_+12h| < 200mm)',
        ]
        piv_cond = piv_cond.reindex([c for c in order if c in piv_cond.index])
        
        piv_cond.plot(kind='barh', ax=ax, width=0.75, colormap='viridis')
        ax.set_title('Conditional Hydrological Sensitivity: ΔH +12h Correlation Across Regimes', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Pearson Correlation (r with ΔH +12h)', fontsize=10)
        ax.set_ylabel('Condition / Regime', fontsize=10)
        ax.axvline(0, color='black', lw=0.8, linestyle='--')
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')
        ax.legend(title='Candidate Rain Station', loc='lower right', framealpha=0.9)

    fig5_path = figures_dir / '05_conditional_relationship_analysis.png'
    fig.tight_layout()
    fig.savefig(fig5_path, dpi=160)
    plt.close(fig)
    generated_paths.append(fig5_path)

    # -------------------------------------------------------------
    # Figure 6: Leave-One-Out (LOO) Event Stability
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    if not loo_df.empty:
        plot_loo = loo_df.head(6).copy()
        y_pos = np.arange(len(plot_loo))
        
        ax.errorbar(
            plot_loo['loo_mean_r'], y_pos,
            xerr=plot_loo['loo_std_r'], fmt='o', color='#1f77b4',
            ecolor='crimson', elinewidth=2, capsize=4, label='LOO Mean ± 1 Std'
        )
        ax.scatter(
            plot_loo['baseline_r'], y_pos,
            marker='D', color='darkgreen', s=60, zorder=5, label='Baseline Full Series'
        )
        ax.scatter(
            plot_loo['loo_min_r'], y_pos,
            marker='|', color='black', s=100, zorder=5, label='LOO Min / Max'
        )
        ax.scatter(plot_loo['loo_max_r'], y_pos, marker='|', color='black', s=100, zorder=5)

        ax.set_yticks(y_pos, [f"{r['rain_station_name']}\n(Score: {r['stability_score']})" for _, r in plot_loo.iterrows()])
        ax.set_title('Full-period Sensitivity to Removing Flagged Episodes', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Correlation with ΔH +12h', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')
        ax.legend(loc='lower right', framealpha=0.9)

    fig6_path = figures_dir / '06_event_stability_loo.png'
    fig.tight_layout()
    fig.savefig(fig6_path, dpi=160)
    plt.close(fig)
    generated_paths.append(fig6_path)

    # -------------------------------------------------------------
    # Figure 7: Multi-Station Selection & Collinearity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not multi_eval_df.empty:
        df_p = multi_eval_df.copy()
        
        colors = ['#1f77b4' if 'Model B' in s else '#2ca02c' if '3-Station' in s else '#ff7f0e' for s in df_p['subset_name']]
        
        ax.scatter(
            df_p['max_collinearity'], df_p['train_pearson_r'],
            s=df_p['n_stations'] * 70, c=colors, alpha=0.85, edgecolors='black', zorder=4
        )

        for _, row in df_p.iterrows():
            ax.annotate(
                row['subset_name'],
                (row['max_collinearity'], row['train_pearson_r']),
                xytext=(8, 4), textcoords='offset points',
                fontsize=8.5, fontweight='bold'
            )

        ax.set_title('Rain-only Ridge: Common Purged Train Anchors', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Maximum Pairwise Collinearity (Pearson r between stations)', fontsize=10)
        ax.set_ylabel('Training Correlation with ΔH +12h (r)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    fig7_path = figures_dir / '07_station_selection_collinearity.png'
    fig.tight_layout()
    fig.savefig(fig7_path, dpi=160)
    plt.close(fig)
    generated_paths.append(fig7_path)

    return generated_paths



from .stage2_report import write_stage2_report
