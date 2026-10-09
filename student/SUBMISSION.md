# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Văn Giáp
- MSSV: 2A202602903
- Email: giap8547813@gmail.com
- Link repo (fork): https://github.com/Giappp/K4-L3L4-DAY23-NguyenVanGiap-2A202602903-SensorFusion.git
- Commit hash nộp (`git rev-parse HEAD`): d5f405ef3fd552c78a5aa52a34fe5c75a5333aaf

## Tóm tắt kết quả

Số liệu dưới đây lấy từ [metrics.json](artifacts/metrics.json) hiện có và đã đối chiếu với [grade_run.log](artifacts/grade_run.log). Đây là lần chạy ngắn **0–20**, chưa phải lần chạy chấm điểm đủ **0–198**; cần chạy lại và cập nhật báo cáo trước khi nộp.

- `fusion_mode`: `compare`; `frames`: `[0, 20]` (21 frame mỗi mode); `seed`: `0`.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `detection.precision`: `0.9545454545454546`; `detection.recall`: `1.0`; `detection.tp/fp/fn`: `42/2/0`.

| Metric | LiDAR | LiDAR + camera (fused) |
|---|---:|---:|
| `rmse` (m) | 0.11370319390900692 | 0.15438963381882595 |
| `matches` | 34 | 34 |
| `sum_sq_err` (m²) | 0.4395661543737137 | 0.8104294070441798 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 8 | 8 |
| `mean_confirmed_tracks` | 1.619047619047619 | 1.619047619047619 |
| `precision_track = matches / (matches + ghost_track_frames)` | 1.0 | 1.0 |
| `coverage = matches / det_tp` | 0.8095238095238095 | 0.8095238095238095 |

Camera làm RMSE tăng khoảng **0.04068644 m** trong lần chạy này. Cả hai mode vẫn có RMSE ≤ 0.45 m, precision_track ≥ 0.75 và coverage ≥ 0.70; mức tăng RMSE nằm trong giới hạn 0.05 m của rubric. Hai mode có cùng số cặp ghép và ghost/miss, nên chênh lệch RMSE không đi kèm thay đổi số lượng cặp đánh giá. Tuy nhiên, chưa thể kết luận kết quả cho toàn segment từ 21 frame này.

Trong log, mỗi mode có 21 record; các frame 0, 1, 2, 3 đều có `matches = 0`, `misses = 2`, tạo tổng 8 miss. Từ frame 4 đến 20 không có miss. Điều này phù hợp với giai đoạn tích lũy score trước khi các track được confirmed. Mỗi record thỏa `matches + ghosts == confirmed` và `matches + misses == valid_gt`; bộ kiểm tra của platform xác nhận tổng số liệu khớp metrics.

Lệnh chạy chấm điểm, sau khi đặt `frame_start: 0`, `frame_end: 198` trong cấu hình:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

RMSE = `sqrt(sum_sq_err / matches)`, tính trên vị trí 3D của confirmed tracks ghép một-một với GT xe trong cửa sổ BEV, gate XY 2.0 m. Camera sử dụng tâm hộp ground-truth FRONT 2D có thêm nhiễu theo seed; kết quả này đánh giá fusion với đo camera mô phỏng, chưa chứng minh chất lượng một camera detector.

Các file per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`, `grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo LiDAR 3D và camera 2D trong EKF (`z`, `R`)?**

   LiDAR có `z = [x_s, y_s, z_s]ᵀ` trong hệ cảm biến, kích thước 3×1, đơn vị mét. `R` là ma trận 3×3 với đường chéo `sigma_lidar_x²`, `sigma_lidar_y²`, `sigma_lidar_z²` (mặc định 0.01 m² mỗi trục). Camera có `z = [u, v]ᵀ`, kích thước 2×1, đơn vị pixel; `R` là ma trận 2×2, mặc định đường chéo 25 pixel². Camera dự đoán phép đo bằng chiếu pinhole phi tuyến nên EKF dùng Jacobian `H`. Xem [build_camera_measurement](workspace/camera_fusion.py) và [Sensor/Measurement](../platform/fusion_lab/tracking/sensors.py). `Q` biểu diễn nhiễu mô hình chuyển động, còn `R` biểu diễn nhiễu phép đo.

2. **Vì sao cần gating Mahalanobis trước khi gán?**

   Gating loại các cặp track–đo không phù hợp thống kê trước khi greedy chọn cặp có chi phí nhỏ nhất. Với innovation `gamma = z − h(x)` và `S = HPHᵀ + R`, khoảng cách bình phương là `gammaᵀ S⁻¹ gamma`. Khác khoảng cách Euclidean, Mahalanobis xét cả độ bất định và tương quan của sai số. Trong [association.py](workspace/association.py), cặp chỉ được giữ khi khoảng cách hữu hạn, không âm và nhỏ hơn `chi2.ppf(0.995, sensor.dim_meas)`. FOV được kiểm tra trước phép chiếu để tránh xử lý camera có độ sâu không hợp lệ.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**

   Pipeline là **track-then-fuse**, dùng một tracker cho mỗi lượt chạy. Trong mỗi frame, [run_lab.py](../platform/fusion_lab/scripts/run_lab.py) gọi predict một lần cho các track hiện có, association/update LiDAR và quản lý vòng đời LiDAR, sau đó association/update camera trên các track đó. Không ghép detection LiDAR và camera thành một detection chung trước tracking. Log hiện có chứa `mode = lidar` và `mode = fused` cho cùng khoảng frame 0–20; số liệu của chúng nằm trong hai nhánh `tracking.lidar` và `tracking.fused`. Thứ tự các bước bên trong frame được chứng minh bằng code runtime, vì log JSONL chỉ lưu metric tổng kết mỗi frame.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**

   Calibration sai làm phép đổi hệ tọa độ hoặc phép chiếu `h(x)` sai, nên `z − h(x)` có thể xuất hiện độ lệch có hệ thống, tăng biên độ hoặc thay đổi theo vị trí xe. Nếu Mahalanobis vượt ngưỡng, gating từ chối phép đo. Nếu vẫn nằm trong gate, update có thể kéo track về vị trí sai và tăng RMSE. Đây là giải thích lý thuyết; bài hiện tại chưa có thử nghiệm làm lệch calibration để đo ảnh hưởng.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao LiDAR quyết định score/init/delete còn camera chỉ EKF update?**

   Khi danh sách phép đo rỗng, không thể lấy loại cảm biến từ `meas_list[0]`. Tham số `sensor` giúp kết thúc đúng lượt: LiDAR vẫn phải trừ score các track không ghép nằm trong FOV và kiểm tra xóa, còn camera không thay đổi vòng đời. [associate_and_update](workspace/association.py) luôn gọi `manager.manage_tracks` kể cả khi không có phép đo. Trong [TrackManager](../platform/fusion_lab/tracking/manager.py), chỉ LiDAR tăng score khi hit và quản lý miss/delete/birth; camera chỉ sửa trạng thái/covariance qua EKF. Nhờ vậy track không bị phạt vì nằm ngoài vùng nhìn camera, và phép đo ảnh 2D không tự khởi tạo vị trí 3D chưa biết độ sâu.

6. **Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**

   Trong [track_management.py](workspace/track_management.py), track mới có score `1/6`, state `initialized`, vị trí chuyển từ hệ LiDAR sang hệ xe, vận tốc bằng 0. Covariance vị trí được xoay theo extrinsic; covariance vận tốc khởi tạo bằng các phương sai `sigma_p44²`, `sigma_p55²`, `sigma_p66²`. Mỗi LiDAR hit cộng `1/6`, chặn tối đa 1; track chưa confirmed chuyển sang `tentative`, rồi `confirmed` khi score **> 0.8**. Miss trong FOV trừ `1/6`, giữ nguyên trạng thái confirmed. Track bị xóa nếu `Pxx > 9 m²` hoặc `Pyy > 9 m²`, hoặc nếu confirmed có score **< 0.6**, hoặc chưa confirmed có score **≤ 0**. Tại đúng ngưỡng covariance 9 hoặc score confirmed 0.6, điều kiện tương ứng chưa kích hoạt xóa.

## Bonus (không bắt buộc)

Không. Chưa có bằng chứng bonus trong `student/bonus/`.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): Giải thích mục tiêu lab, Q, innovation và camera fusion; review Part E (`kalman.py`); hoàn thiện Part G (`camera_fusion.py`), Part F (`association.py`), Part H (`track_management.py`); hỗ trợ soạn báo cáo và đối chiếu metrics với log.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Trong phiên hỗ trợ, Codex đã chạy 64 test gốc trong `test_kalman.py`, `test_tracking_regressions.py`, `test_tracking_contract.py`, tất cả pass. Launcher chỉ nạp các module E–H để tránh phụ thuộc detector; không sửa nội dung test. Có kiểm tra số trực tiếp cho extrinsic, covariance khởi tạo và phép chiếu camera. Metrics của lần chạy 0–20 hiện có đã được đối chiếu với log bằng `validate_metrics_records`, không phát hiện sai lệch. Chưa xác nhận toàn bộ `student/tests` và chưa có lần chạy chấm điểm 0–198 trong artifacts hiện tại. Người nộp cần tự đọc, kiểm tra và giải thích được code cùng các câu trả lời trước khi nộp.

## Checklist nộp

- [x] **Part E–H** đã implement và 64 test tracking/Kalman đã pass; còn cần chạy toàn bộ `pytest student/tests -q` để xác nhận không có `failed`/`xfailed`.
- [x] Part A–D sử dụng code có sẵn; phiên hỗ trợ không sửa các phần này.
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198` (artifacts hiện tại chỉ có frame 0–20).
- [x] Commit artifacts của lần chạy chấm điểm cuối; 6 file hiện có đã được git theo dõi, không sửa tay.
- [x] Báo cáo đã có trả lời E–H và khai báo AI; cần cập nhật số liệu sau lần chạy đầy đủ và chốt commit hash.
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key.
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`.
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md)).
