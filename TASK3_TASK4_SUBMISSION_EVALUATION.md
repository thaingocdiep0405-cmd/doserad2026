# DoseRAD2026 Task 3/4 — đánh giá và bộ nộp

> **Snapshot of 2026-08-15, superseded.** The checkpoints assessed below belong to the
> end-to-end dose-prediction networks that were the submission candidates at that date.
> They were later replaced by the hybrid synthetic-CT + analytic pencil-beam method,
> which is what was submitted to the final test phase. See the README for the final
> method and results. Kept as a record of the intermediate stage.

Ngày chốt local: 2026-08-15 (Asia/Bangkok).

## Kết luận

- Task 3 Proton CT và Task 4 Proton MRI đã train xong, chọn checkpoint, kiểm tra
  overfit, đánh giá full-volume và đóng gói thành hai thuật toán riêng.
- Cả hai image cuối đều qua smoke test không mạng: `/health = 200`,
  `/invoke = 201`, đủ 10 output scalar-4D MHA và đúng kiến trúc
  `linux/amd64`.
- Bộ local đủ để tạo **preliminary submission đầu tiên**, nhưng chưa thể khẳng
  định top trước khi có điểm hidden test. Trang chính thức hiện vẫn cảnh báo lỗi
  IDD và yêu cầu chưa submit nếu muốn giữ lượt nộp.

## Dữ liệu và leakage

- 75 bệnh nhân public được chia cố định theo bệnh nhân: 60 train, 15 validation.
- Không có bệnh nhân trùng giữa train và validation.
- Full-volume evaluation dùng 45 beamlet, trải trên đủ 15 bệnh nhân validation.
- Validation đã được dùng để chọn checkpoint/tham số, vì vậy chỉ preliminary
  hidden test mới là đánh giá ngoài mẫu hoàn toàn.

## Checkpoint được chọn

| Task | Checkpoint chính | MAE | NRMSE | IDD | Scale ratio |
|---|---|---:|---:|---:|---:|
| Task 3 — Proton CT | blend 50% v1 + 50% v2 | 0.048287 | 0.006003 | 0.197855 | 0.950037 |
| Task 4 — Proton MRI | v1 teacher-student | 0.047828 | 0.005971 | 0.194345 | 0.944737 |

Các số trên là full-volume validation, `patch=128³`, `overlap=0.25`,
`ray_gate=1e-6`, không relative cutoff.

### Kiểm soát overfit

- Task 3 v1 hoàn thành 30 epoch; train/validation vẫn cùng xu hướng, chưa có
  dấu hiệu overfit cổ điển rõ rệt.
- Task 3 v2 cải thiện IDD 5.30% nhưng MAE xấu 3.42%; epoch 2 tiếp tục làm
  validation xấu hơn nên đã dừng sớm. Blend được chọn để cân bằng rủi ro.
- Task 4 v1 hoàn thành 30 epoch và là checkpoint chính.
- Task 4 v2 cải thiện IDD 4.12% nhưng MAE xấu 3.75% và scale xấu 0.79%;
  không dùng làm submission chính.

## Runtime

Conditioning proton đã được vector hóa trên GPU và kiểm chứng với bản NumPy:

- sai khác tensor lớn nhất: `1.19e-7`;
- sai khác full-volume trung bình: `1.50e-10`;
- giữ `overlap=0.25`, không đổi cấu hình accuracy.

Benchmark trên ảnh validation thật `120×447×449`, 64 condition trải đều:

| Task | Ước tính inference cho 500 beamlet | Peak CUDA với 64 beamlet |
|---|---:|---:|
| Task 3 | 428.69 giây | 22.99 GiB reserved |
| Task 4 | 427.50 giây | 22.99 GiB reserved |

Submission dùng `BEAMLET_CHUNK_SIZE=32`; cấu hình này đã đo peak khoảng
17.14 GiB để có khoảng trống trên A10G 24 GiB. Các con số 428–429 giây là
ước tính inference trên NVIDIA GB10, chưa phải runtime chính thức do Grand
Challenge fit trên A10G và chưa bao gồm mọi khác biệt I/O/hardware. Ngưỡng chính
thức cho proton là 500 giây.

## Archive cuối

### Task 3 — Proton CT

- File: `doserad_proton_ct/dist/proton-ct/doserad2026-proton-ct.tar.gz`
- Dung lượng: 3,934,305,172 byte
- SHA-256: `b84bda60bb88a8b4f2017a6c6738c90ec42db68953511a599bb02cc036adea26`
- Model SHA-256: `b4663434dc763c280c7c2b4b4241a6bbfe48c5ba7de57a2e998373cf4c14b482`
- Docker image: `sha256:399d0ee12b4c9ee4351a5b6c141765369bc27d6b3f2dfdfef1f7e51028a0a008`

Bản này thay bản `82b08e4c…` ngày 2026-08-15 sau khi sửa cách ghi output
(xem "Sửa host memory" bên dưới). Checkpoint không đổi.

### Task 4 — Proton MRI

- File: `doserad_proton_ct/dist/proton-mri/doserad2026-proton-mri.tar.gz`
- Dung lượng: 3,934,315,956 byte
- SHA-256: `9099c2eed0c2304e3ca8f975dc3523beae69cb1a94c5516be1c3f87a58f627b7`
- Model SHA-256: `ffddd2a90642bc9bc8549b21e7434ecedfe8bba14f90650a918b787188e2fa4c`
- Docker image: `sha256:a0a442c0409caa31decb7aa36b3e0e5500e96de4ecdf49e5a4e2827c4ff2bafd`

Bản này thay bản `dcfe6bb6…` ngày 2026-08-15. Checkpoint không đổi.

Hai archive đều đã qua `gzip -t`, đều là `linux/amd64` và mang
`org.grand-challenge.api-method=invoke`, với `TASK=proton-ct` và
`TASK=proton-mri` tương ứng.

## Sửa host memory (2026-08-16)

Bản trước giữ toàn bộ dose map của một run trong RAM rồi mới gọi
`sitk.JoinSeries` ở cuối. Đo trên grid validation thật `120×447×449`
(91.9 MiB mỗi map), 64 map:

| Cách ghi | Peak RSS | Output |
|---|---:|---:|
| Buffer + JoinSeries | 12.143 GiB | 6,165,596,492 byte |
| Streaming writer | 0.660 GiB | 6,165,596,492 byte |

Peak của bản buffer bằng khoảng 2.1× kích thước stack vì `JoinSeries` tạo thêm
một bản copy đầy đủ. Ngoại suy tới 500 beamlet: 44.9 GiB chỉ riêng các frame,
tức khoảng 90 GiB lúc copy. Instance A10G của Grand Challenge cho tối đa 31 GiB
DRAM khả dụng (`ml.g5.2xlarge`, đã trừ 1 GiB reserved), `ml.g5.xlarge` chỉ 15
GiB. Bản streaming ghi thẳng frame vào offset `header + idx_in_output *
frame_bytes` nên peak chỉ còn một frame, và không phụ thuộc thứ tự beamlet hoàn
thành.

Output không đổi: header do chính SimpleITK sinh từ probe 1×1×1×1, chỉ patch
`DimSize`. Đã kiểm chứng byte-identical với `JoinSeries` trên **cả 75** geometry
CT thật của tập train (0 mismatch), cùng 17 test trong
`doserad_proton_ct/tests/`.

Container gate chạy lại offline sau khi sửa, cho **cả hai task**
(`submission/smoke_test.sh proton-ct 4` và `... proton-mri 4`): `/health = 200`,
`/invoke = 201`, slot 1 là stack 4D đúng 4 frame, 9 slot còn lại là placeholder
1×1×1×1.

## Hai lỗi phát hiện thêm khi kiểm tra

**Fixture smoke test trước đây đưa dữ liệu rác vào container.**
`create_submission_smoke_fixture.py` ghi MHA với `CompressedData = True` nhưng
không có `CompressedDataSize`. MetaIO in "Uncompress failed" rồi trả về bộ nhớ
chưa khởi tạo thay vì báo lỗi, nên ảnh input có giá trị tới 1.8e+36. Mọi smoke
test trước đó chỉ chứng minh phần plumbing, không chứng minh phần số học. Đã
chuyển sang ghi uncompressed.

**Phantom MRI của fixture bị suy biến.** Body mask cho MRI là
`image <= bounds[0]` với `bounds[0]` là phân vị 0.5 của các voxel dương. Phantom
cũ đặt toàn bộ foreground bằng đúng 1.0, nên `bounds[0] = 1.0` và mask xoá sạch
cả volume: smoke test Task 4 trả về `peak_dose = 0` bất kể model làm gì. Trên
MRI validation thật, mask chỉ xoá thêm 0.14–0.34% ngoài phần background
(`1ABB039` 0.336%, `1ABB041` 0.136%, `1ABB042` 0.156%), nên đây là lỗi của
fixture, không phải của inference. Sau khi đổi sang phantom có gradient,
Task 4 cho `peak_dose = 7.14e-05`, tương đương Task 3 (`7.61e-05`).

**Cutoff có thể xoá sạch dose map.** Prediction thô đạt cực đại 7.61e-05
(Task 3) và 7.14e-05 (Task 4), trong khi `minimum_cutoff` của ví dụ trong đặc tả
là 0.02, tức lớn hơn khoảng 260–280 lần. `dose_scale` của checkpoint là
1.1307e-03 và audit tập train cho per-map max trong khoảng 4.84e-04 đến
1.64e-03, nên nếu hidden test dùng cutoff 0.02 với cùng đơn vị thì mọi map đều
thành 0. Container vẫn tuân thủ đặc tả (bắt buộc phải zero dưới cutoff), nhưng
nay in cảnh báo kèm max trước cutoff và `peak_dose` của cả run, để log
preliminary phát hiện ngay sai đơn vị thay vì âm thầm nộp toàn số 0. Metadata
tập train không có `minimum_cutoff`, nên không thể chốt câu này bằng dữ liệu
local.

## Đánh giá khả năng cạnh tranh

Local validation cho thấy hai model ổn định và Task 3/4 đã vượt mọi gate kỹ
thuật local. Tuy nhiên không thể suy ra vị trí top vì leaderboard còn dùng
plan-level MAE, local gamma 1%/1 mm, DVH score và runtime A10G; các metric này
không có ground truth hidden để tái tạo local. Lượt preliminary đầu nên dùng
hai archive chính ở trên, sau khi cảnh báo IDD trên trang chính thức biến mất.
Kết quả preliminary sẽ quyết định có dùng lượt tiếp theo cho checkpoint thiên
IDD hay tiếp tục tối ưu accuracy/runtime.
