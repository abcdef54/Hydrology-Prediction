"""Plot September and October error episodes using the saved forecast snapshot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


OUT = Path(__file__).resolve().parents[1] / "reports/large_error_events_2026-10-15"
TZ = "Asia/Ho_Chi_Minh"


def main():
    data = pd.read_csv(OUT / "forecast_context.csv")
    data["forecast_at"] = pd.to_datetime(data.forecast_at, utc=True).dt.tz_convert(TZ)
    data = data[data.horizon_minutes.eq(1440)].set_index("forecast_at").sort_index()
    cases = [
        ("september_11_14", "11–14/09/2026", "2026-09-11", "2026-09-14 23:00",
         [("2026-09-12 10:00", "2026-09-13 11:00", 26, -1),
          ("2026-09-13 22:00", "2026-09-14 13:00", 16, 1)]),
        ("october_07_08", "07–08/10/2026", "2026-10-07", "2026-10-08 23:00",
         [("2026-10-07 14:00", "2026-10-08 07:00", 18, -1)]),
    ]
    for slug, title, start, end, runs in cases:
        start, end = pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ)
        lo, hi = start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)
        window = data.reindex(pd.date_range(lo, hi, freq="h"))
        fig, (ax, err) = plt.subplots(
            2, 1, figsize=(16, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}, layout="constrained",
        )
        ax.plot(window.index, window.actual, color="#ce4525", lw=2,
                label="Thực tế (trung bình giờ)")
        ax.plot(window.index, window.water_level_mm, color="#1677b8", lw=1.8,
                label="Dự báo trước 24 giờ")
        err.plot(window.index, window.error, color="#1677b8", lw=1.6,
                 label="Sai lệch = dự báo − thực tế")
        err.axhline(0, color="#555555", lw=.8)
        err.axhline(200, color="#ce4525", ls="--", label="Ngưỡng ±200 mm")
        err.axhline(-200, color="#ce4525", ls="--")
        descriptions = []
        for a, b, n, sign in runs:
            a, b = pd.Timestamp(a, tz=TZ), pd.Timestamp(b, tz=TZ)
            run = window.loc[a:b]
            assert len(run) == n and run.ae.gt(200).all()
            assert (run.error * sign).gt(200).all()
            color = "#ffc857" if sign < 0 else "#a8d5a2"
            for panel in (ax, err):
                panel.axvspan(a, b, color=color, alpha=.3)
                panel.axvline(a, color="#777777", ls=":", lw=.8)
                panel.axvline(b, color="#777777", ls=":", lw=.8)
            ax.scatter(run.index, run.water_level_mm, color="#1677b8", s=12, zorder=3)
            label = "Thiếu" if sign < 0 else "Thừa"
            descriptions.append(f"{label}: {n} mốc liên tiếp · MAE {run.ae.mean():.2f} mm · Max {run.ae.max():.2f} mm")
            print(slug, label, a, b, n, run.ae.mean(), run.ae.max())
        ax.set_ylabel("Mực nước (mm)")
        ax.set_ylim(250, 1400)
        err.set_ylabel("Sai lệch (mm)")
        err.set_ylim(-550, 550)
        err.set_xlabel("Thời điểm được dự báo · Giờ Việt Nam (UTC+7)")
        ax.legend(loc="upper left", ncol=2)
        handles, labels = err.get_legend_handles_labels()
        handles += [Patch(facecolor="#ffc857", alpha=.5, label="Chuỗi dự báo thiếu >200 mm")]
        if len(runs) > 1:
            handles += [Patch(facecolor="#a8d5a2", alpha=.5, label="Chuỗi dự báo thừa >200 mm")]
        err.legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
        for panel in (ax, err):
            panel.grid(alpha=.2)
            panel.set_xlim(lo, hi)
        err.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=start.tzinfo))
        err.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M", tz=start.tzinfo))
        fig.suptitle(
            f"Dã Viên · Giai đoạn {title} · Dự báo trước 24 giờ\n"
            + "\n".join(descriptions)
            + "\nMAE: sai số tuyệt đối trung bình trong chuỗi tô màu · Hiển thị thêm 1 ngày trước và sau",
            fontsize=13,
        )
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"{slug}_forecast_error_vi.{ext}", dpi=180)
        plt.close(fig)
        window.to_csv(OUT / f"{slug}_forecast_error_vi.csv", index_label="target_vietnam")


if __name__ == "__main__":
    main()
