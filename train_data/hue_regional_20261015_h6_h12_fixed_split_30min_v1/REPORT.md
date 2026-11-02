# Bộ dữ liệu khu vực 11 trạm — lưới 30 phút

Dataset mới, tách biệt dataset 1h cũ. Không huấn luyện mô hình trong bước build.

## Cấu hình

- 11 trạm nước và cùng danh mục trạm mưa, cùng mốc phân chia fixed và khoảng purge 24h với dataset 1h.
- Lưới anchor cách nhau 30 phút; anchor thiếu hiện tại, target hoặc không đủ lịch sử được loại, nên có thể có khoảng trống.
- 96 timestep từ t−47,5h đến t. Mỗi giá trị nước là trung bình quan trắc hợp lệ trong (t−30 phút, t], không phải trung bình 1 giờ như dataset cũ.
- Hai nhãn là mực nước tại đúng t+6h và t+12h (12 và 24 bước 30 phút); mỗi nhãn cũng là trung bình 30 phút.
- Các scaler được fit lại chỉ trên train của split fixed; không tái sử dụng scaler dataset 1h.

| Split | Số sample |
|---|---:|
| train | 126,026 |
| validation | 23,760 |
| test_exploratory | 15,895 |
| purged | 1,056 |

## Quan trắc mưa và forecast

Mưa được tính từ các quan trắc 10 phút. Tại từng anchor 30 phút, tổng 1h sử dụng 6/6 mẫu hoặc ước tính 5/6 ×6/5; ít hơn 5 mẫu là thiếu. Quan trắc trùng timestamp làm cửa sổ thiếu. Các tổng 3/6/12/24/48h cộng các khoảng 1h không chồng lấn, không cộng hai lần.

Nguồn forecast có độ phân giải 1 giờ. Tại anchor :30, dùng nguyên vector forecast đã chọn tại giờ tròn trước đó, không dùng tương lai hay nội suy; tuổi forecast tăng thêm 0,5h. Do vậy các cửa sổ forecast 0–3/3–6/6–12/12–24h tại :30 bắt đầu sớm hơn anchor hiện tại 30 phút. Đây là khác biệt quan trọng so với forecast đồng bộ chính xác từng anchor. Các giả định availability hồi cứu từ dataset cũ vẫn áp dụng; chưa chứng minh được dữ liệu thật sự khả dụng theo thời gian thực.

Nước thiếu được giữ NaN trong parquet và có mask; bộ đọc chuẩn hóa bằng scaler train rồi điền 0 trong không gian chuẩn hóa. Hai mẫu liên tiếp có lịch sử chồng lấn nên không thể xem số mẫu tăng gần gấp đôi là số sự kiện độc lập tăng gấp đôi.

## Sử dụng

Các lệnh training cần `--dataset-dir` trỏ tới thư mục này, `--horizon 30min`, `--lstm-seq-len 96` và `--target-horizons 6 12`. Không dùng checkpoint/scaler 1h như thể chúng là mô hình 30 phút.

Xác minh: `PYTHONPATH=. .venv/bin/python research/scripts/validate_regional_30min_dataset.py train_data/hue_regional_20261015_h6_h12_fixed_split_30min_v1`. Xem validation.json để biết những gì đã kiểm tra. Nguồn và hash được lưu trong manifest.json.
