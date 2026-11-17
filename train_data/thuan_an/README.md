# train_data/thuan_an

Hai dataset train của hai model Thuận An (`models/thuan_an/`), dựng bằng `src/ai/station_dataset/build.py`.

| Dataset | Dùng cho |
|---|---|
| `ta_water_only_wind_tide30_fc` | model đơn `wind_tide30_fc_gru32x3_seq12_cosine_ep77` |
| `ta_water_only_wind` | thành phần thứ hai của ensemble 34k |

Lệnh dựng lại, số mẫu và sha256: `models/thuan_an/REPRODUCE.md` mục 4. Cấu hình: `models/thuan_an/dataset_config.json`.
