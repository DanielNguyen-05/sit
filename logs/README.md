# logs/ – Slurm log của từng job

Mỗi job ghi hai file: `<tên job>_<job id>.out` (nội dung in ra, gồm kết quả) và `.err` (cảnh báo, lỗi).
Riêng job train in đường loss ra `.err`; `tools/plot_loss.py` đọc từ đó.

| Thư mục | Loại job | Script gửi job |
| --- | --- | --- |
| `train/` | train model | `scripts/train.sh` |
| `fid/` | đo FID một hoặc vài checkpoint | `scripts/fid.sh` |
| `sweep/` | một đợt nhiều cấu hình FID trong một job | `scripts/fid_batch.sh`, `scripts/guidance_sweep.sh` |
| `analysis/` | đo loss theo mức nhiễu và số token | `scripts/budget_loss.sh` |
| `bench/` | đo chi phí: tham số, GFLOPs, tốc độ, bộ nhớ | `scripts/bench.sh` |
| `prepare/` | chuẩn bị dữ liệu (mã hóa ImageNet) | `scripts/encode.sh` |

## Các job đã có (SiT-S/2 trừ khi ghi B/2; FID không CFG trừ khi ghi rõ)

| File | Việc | Kết quả chính |
| --- | --- | --- |
| `train/base_78522` | train baseline S/2, 400K bước | loss cuối 0,7712 |
| `train/method_78588` | train method S/2 (K = 32), 400K bước | loss cuối 0,7847 |
| `train/base_B2_79033` | train baseline B/2, 400K bước | loss cuối 0,7547 |
| `train/method_B2_79034` | train method B/2 (K = 32), 400K bước | loss cuối 0,7661 |
| `train/nest_time_83681` | train nested, gán mức token theo mức nhiễu | 400K bước |
| `train/nest_uni_84177` | train nested, gán mức token ngẫu nhiên | 400K bước |
| `fid/fid_78928` | FID-50K baseline và method ở 400K | 51,91 và 52,28 |
| `fid/fid_79107` | FID-50K baseline, các mốc 50K–350K | 108,11 → 53,81 |
| `fid/fid_79397` | FID-50K method, các mốc 50K–350K | 115,16 → 54,28 |
| `fid/fid_79400` | FID-50K baseline B/2 ở 400K | 26,90 |
| `fid/fid_84283` … `fid_84286` | FID-10K nested_uniform, K = 8 / 16 / 32 / 64 | 70,40 / 62,15 / 57,84 / 54,60 |
| `fid/fid_84287` | FID-10K nested_uniform, lịch K theo t | 60,03 |
| `fid/fid_84288` | FID-10K nested_uniform, 64 token + dẫn hướng 16 token, s = 1,5 | 49,63 |
| `fid/fid_84480` | FID-10K method và baseline ở 400K (chạy trên V100) | 55,03 và 54,21 |
| `fid/fid_nt_84486` | FID-10K nested_time, lịch K theo t | 57,45 |
| `fid/fid_nt_84487` | FID-10K nested_time, 64 + dẫn hướng 16, s = 1,5 | 57,89 |
| `analysis/budget_loss_84488` | loss theo mức nhiễu cho cả 4 checkpoint | bảng trong file; JSON ở `results/*/budget_loss/` |
| `bench/bench_79116` | chi phí S/2 và B/2, baseline và method | method: 0,60× FLOPs, train 1,77×, sinh ảnh 2,03× |

Chỉ có trên server (chưa đồng bộ về máy): `sweep/guid_sweep_85230` (đợt dẫn hướng 1), `sweep/fid_batch_85272`
(đợt 2), `sweep/fid_batch_85273` (đợt 3), `fid/sg_A_85194`, `fid/sg_A_85195` (dẫn hướng s = 1,25 và 2),
`fid/fid_nt_84482` (nested_time K = 8). Tổng hợp mọi kết quả: `results-log.md` trong Project SiT.
