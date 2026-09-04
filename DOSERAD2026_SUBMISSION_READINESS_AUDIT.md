# DoseRAD2026 — Submission readiness audit

Ngày đánh giá: 2026-08-14

## Kết luận điều hành

Hiện tại chưa có task nào đủ bằng chứng để gọi là ứng viên tranh giải. Task 1
và Task 2 đã có pipeline hoàn chỉnh, checkpoint, container và smoke test nên có
thể dùng cho **một lượt preliminary chẩn đoán** sau khi cảnh báo lỗi IDD trên
trang chính thức được gỡ bỏ. Task 3 và Task 4 chưa sẵn sàng vì dữ liệu proton
chưa tải xong và chưa có project/model/container tương ứng.

Không thể suy ra Mean Position từ validation local. Mean Position chỉ tồn tại
sau khi thuật toán được chấm trên hidden test, trên cả sáu chiều xếp hạng.

## Tiêu chuẩn chính thức dùng để audit

Mỗi task được xếp hạng theo sáu chiều:

1. masked MAE từng beam;
2. IDD curve distance;
3. plan-level stratified MAE;
4. local gamma 1%/1 mm;
5. DVH clinical score;
6. runtime chuẩn hóa.

Runtime được tính trên NVIDIA A10G 24 GiB và có trọng số xếp hạng gấp đôi.
Photon bị loại nếu runtime chuẩn hóa đạt hoặc vượt 181 giây; proton bị loại nếu
đạt hoặc vượt 500 giây. Level-1 phải được tổng hợp beam-to-patient rồi
patient-to-submission.

Nguồn chính thức:

- [Metrics and ranking](https://doserad2026.grand-challenge.org/metrics-and-ranking/)
- [Tasks and getting started](https://doserad2026.grand-challenge.org/tasks-and-getting-started/)
- [Submission instructions](https://doserad2026.grand-challenge.org/submission-instructions/)
- [Timeline and rules](https://doserad2026.grand-challenge.org/timeline-and-rules/)
- [Final submission requirements](https://doserad2026.grand-challenge.org/final-submission-requirements/)

## Readiness theo task

| Task | Dữ liệu | Model/local validation | Container | Trạng thái |
| --- | --- | --- | --- | --- |
| 1 — Photon CT | 75/75, 40.500/40.500 maps | Có; MAE 0,0797, IDD 0,2322 trên 15 bệnh nhân/15 CP | Có, smoke pass | Chỉ nên preliminary chẩn đoán |
| 2 — Photon MRI | 75/75, 40.500/40.500 maps | Có; patient-mean MAE 0,0862, IDD 0,3123 trên 15 bệnh nhân/75 CP | Có, smoke pass | Chỉ nên preliminary chẩn đoán |
| 3 — Proton CT | Đang tải; 44/75 ca hoàn chỉnh, 48.352/81.000 maps tại thời điểm audit | Chưa có | Chưa có | Chưa thể submit |
| 4 — Proton MRI | Dùng cùng kho proton, chưa hoàn chỉnh | Chưa có | Chưa có | Chưa thể submit |

### Task 1 — Photon CT

Điểm tốt:

- split theo bệnh nhân 60 train / 15 validation, không rò rỉ control point;
- CT và density được đưa vào model;
- checkpoint riêng cho MAE/IDD và candidate blend đã được so sánh;
- inference đã batch, chunk, AMP và compile;
- container đúng `linux/amd64`, đúng invoke label, output scalar 4D, cutoff và
  kiểm tra NaN/Inf đều đạt;
- 10/10 unit tests đạt.

Điểm chưa đạt chuẩn ứng viên top:

- chỉ đánh giá một control point trên mỗi bệnh nhân validation; khoảng tin cậy
  MAE 95% xấp xỉ 0,0690–0,0913 còn rộng;
- validation đã được dùng để chọn checkpoint/blend nên có selection bias;
- cuối quá trình train, patch train MAE khoảng 0,0668 nhưng validation khoảng
  0,0904, cho thấy generalization gap khoảng 35%;
- chưa đánh giá ba metric plan-level: plan MAE, gamma và DVH;
- runtime ước tính 83,7 giây là trên GB10 local, chưa phải A10G chính thức;
- đường `torch.compile` trong image amd64 chưa thể test bằng GPU trên host ARM64.

Kết quả hidden leaderboard đang ở vùng beam MAE khoảng 0,0086–0,014 đối với
nhóm đầu. Dù không thể so trực tiếp local và hidden test, chênh lệch nhiều lần
là tín hiệu mạnh rằng model hiện tại mới là baseline kỹ thuật.

### Task 2 — Photon MRI

Điểm tốt:

- split bệnh nhân sạch và dữ liệu hoàn chỉnh;
- đánh giá 75 record phủ đủ 15 bệnh nhân;
- model warm-start từ photon CT, inference đã compile/batch/chunk;
- runtime local ước tính 99,7 giây/181 maps, dưới hard limit local;
- 8/8 unit tests đạt.

Điểm chưa đạt:

- patient-mean MAE đúng quy tắc là 0,0862, không phải record-mean 0,08535;
- khoảng tin cậy MAE 95% xấp xỉ 0,0769–0,0962, IDD còn biến động lớn;
- train/validation gap khoảng 34%, checkpoint tốt xuất hiện sớm rồi xấu đi;
- MRI không chứa electron density trực tiếp nhưng model hiện tại chưa có nhánh
  synthetic-CT/density teacher hoặc physics prior đủ mạnh;
- chưa đánh giá ba metric plan-level và chưa benchmark A10G thật.

Nhóm đầu hidden leaderboard đang ở vùng beam MAE khoảng 0,0158–0,0172. Đây
cũng là khoảng cách quá lớn để coi candidate hiện tại là model tranh giải.

### Task 3 và Task 4 — Proton

Tại lúc audit, tiến trình tải vẫn chạy. Kho proton có 45/75 thư mục bệnh nhân,
44 ca hoàn chỉnh và 48.352/81.000 dose maps. Chưa có `doserad_proton_ct` hay
`doserad_proton_mri`, chưa có split, model, validation, benchmark hay container.
Vì vậy hai task này chưa thể đánh giá accuracy và chưa thể submit.

Proton cần mô hình vật lý khác photon: năng lượng, Bragg peak, water-equivalent
path length, range uncertainty và lateral spot spread phải được conditioning
trực tiếp. Sao chép U-Net photon rồi đổi input không phải phương án đủ chắc chắn.

## Các thiếu hụt khoa học quan trọng

1. **Validation chưa đại diện metric thi.** Local mới có hai trong năm metric
   accuracy; NRMSE không phải metric xếp hạng.
2. **Không có holdout độc lập.** Cùng 15 bệnh nhân được dùng để chọn epoch,
   blend và tham số inference.
3. **Model thiên về image-to-image baseline.** Bài toán là transport vật lý có
   beam geometry, không chỉ là segmentation/regression 3D.
4. **Loss chưa bám đủ ranking.** Cần masked normalized L1, IDD differentiable,
   gradient/gamma surrogate, scale calibration và plan proxy nếu có weights.
5. **Generalization yếu.** Chưa có augmentation hình học nhất quán với beam,
   HU perturbation hoặc MRI bias/noise đủ hệ thống.
6. **Runtime chưa được xác nhận trên A10G.** Smoke test CPU emulation chỉ xác
   nhận contract, không xác nhận CUDA/compile/performance.

## Kiến trúc đề xuất để cạnh tranh

### Photon CT

Dùng physics-prior residual network:

- ray tracing từ source qua CT density để tạo radiological depth;
- explicit aperture/MLC fluence, source distance và beam-axis coordinates;
- primary-dose prior có attenuation/inverse-square;
- mạng 3D dự đoán residual/scatter correction và luôn ép output không âm;
- loss đa mục tiêu bám masked MAE + IDD + dose gradient/gamma surrogate.

### Photon MRI

Tận dụng CT ghép cặp chỉ trong training:

- train CT teacher tốt trước;
- MRI student có auxiliary synthetic density/CT head;
- distill feature và dose từ CT teacher sang MRI student;
- thêm MRI bias field, intensity scaling, noise và artifact augmentation;
- inference cuối vẫn chỉ nhận MRI, đúng task.

### Proton CT/MRI

- conditioning riêng cho energy và spot geometry;
- tính WEPL/radiological path length;
- prior cho range/Bragg peak và lateral Gaussian spread;
- biểu diễn beam's-eye-view theo chuỗi lát cắt; đây là hướng phù hợp với forward
  transport và đã được dùng trong [DoTA](https://arxiv.org/abs/2202.02653),
  thay vì bắt một U-Net toàn cục tự học lại toàn bộ hình học;
- mạng chỉ học heterogeneity correction và residual;
- MRI dùng CT/density teacher khi train tương tự Task 2.

Đề xuất MRI teacher/synthetic-density không chỉ là heuristic: MRI không cung
cấp trực tiếp electron density cần cho tính liều, đúng với mô tả task chính
thức và tổng quan chuyên ngành về
[synthetic CT trong xạ trị MRI-only](https://pubmed.ncbi.nlm.nih.gov/34474325/).
Dataset challenge có CT–MRI đã đăng ký theo cặp, nên supervision chéo modality
là tài sản quan trọng cần tận dụng.

## Thứ tự thực hiện có xác suất thành công cao nhất

1. Chờ tải proton hoàn tất, không làm gián đoạn tiến trình hiện tại.
2. Mở rộng Task 1 validation thành tập cố định nhiều CP mỗi bệnh nhân, phân tầng
   anatomy/beam/gantry; báo cáo patient mean và bootstrap CI.
3. Xây physics-prior Photon CT và chạy ablation một biến mỗi lần.
4. Distill CT teacher sang Photon MRI.
5. Chỉ sau khi proton đủ 75/75 mới dựng chung một proton physics core rồi tách
   CT/MRI head.
6. Test image `linux/amd64` bằng A10G thật; nếu compile fail phải fallback an
   toàn sang eager và vẫn dưới hard limit.
7. Khi cảnh báo IDD chính thức biến mất, dùng một preliminary slot cho baseline
   ổn định để lấy đủ sáu metric hidden; không dùng slot để thử ngẫu nhiên.
8. Dùng các slot sau cho ablation có giả thuyết và khóa hai final candidates
   trước hạn.

## Go/no-go gate trước final submission

Chỉ coi một task là final-ready khi đồng thời đạt:

- dữ liệu và split audit pass, không leakage;
- patient-level evaluation trên cohort cố định đủ rộng và có CI;
- không có xu hướng validation xấu liên tiếp trong khi train tiếp;
- đủ năm accuracy metrics hoặc đã có preliminary feedback tương ứng;
- A10G runtime có ít nhất 20% safety margin so với hard limit;
- container chạy end-to-end nhiều image, mọi output đúng geometry/cutoff;
- checkpoint, config, seed, commit/image digest và report đều được khóa.

Theo gate này: Task 1 và Task 2 chưa final-ready; Task 3 và Task 4 chưa
preliminary-ready.
