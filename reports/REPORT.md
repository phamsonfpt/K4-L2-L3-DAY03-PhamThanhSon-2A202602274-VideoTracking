# Báo cáo Ngày 3 — Tracking Annotation

Họ tên / nhóm: `Phạm Thanh Sơn`  
Ngày: `15/09/2026`

## 1. Quá trình gán nhãn

| Mục                                 | Giá trị     |
| ----------------------------------- | ----------- |
| Công cụ                             | CVAT        |
| Thời gian gán `clip_02` (warm-up)   | `25` phút   |
| Thời gian gán `clip_01`             | `75` phút   |
| Số track trong annotation `clip_01` | `8`         |
| Số keyframe trung bình mỗi track    | khoảng `18` |

Ba tình huống khó nhất: xử lý occlusion bằng cách giữ ID cũ khi xe xuất hiện lại, rà kỹ các đoạn xe cắt ngang nhau và thêm keyframe khi motion blur hoặc vật thể ở xa làm bbox thay đổi nhanh.

## 3. Pre-gold lock và chấm trước/sau rework

Workspace hiện chưa có `evidence/pre-gold/clip_01/manifest.json` hoặc snapshot pre-gold. Vì vậy SHA-256, thời điểm khóa và số liệu pre-gold chưa thể xác nhận từ artifact hiện có.

| Evidence                               | Giá trị          |
| -------------------------------------- | ---------------- |
| SHA-256 manifest                       | Chưa có artifact |
| Thời điểm khóa                         | Chưa có artifact |
| Số row / frame / track trước reference | Chưa có artifact |

| So sánh                     |   HOTA |   DetA |   AssA |   LocA |   IDF1 |   MOTA |   MOTP |  FP |  FN | IDSW |
| --------------------------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | --: | --: | ---: |
| Annotation hiện tại vs gold | 0.8014 | 0.7791 | 0.8262 | 0.8856 | 0.9393 | 0.8726 | 0.8761 |  65 |   8 |    0 |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): **ĐẠT**.

Các frame cần chú ý: track `6` có bbox sớm ở frame `79–100`; track `5` ở frame `64–78`; track `4` kéo dài đến frame `151`; bbox track `6` lệch rõ quanh frame `110–111`.

## 4. Kết quả model: ByteTrack control vs ReID treatment

| Mục                                | Giá trị                                    |
| ---------------------------------- | ------------------------------------------ |
| Python / ultralytics / torch / lap | `3.11.9 / 8.4.145 / 2.6.0+cu124 / 0.5.13`  |
| weights / tracker                  | `yolo26n.pt / ByteTrack / BoT-SORT + ReID` |
| conf / IoU / imgsz / classes       | `0.25 / 0.70 / 960 / [2, 5, 7]`            |
| device / persist / số frame        | `0 / True / 190`                           |

| So sánh                   |   HOTA |   DetA |   AssA |   LocA |   IDF1 |   MOTA |   MOTP |  FP |  FN | IDSW |
| ------------------------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | --: | --: | ---: |
| Bạn vs gold               | 0.8014 | 0.7791 | 0.8262 | 0.8856 | 0.9393 | 0.8726 | 0.8761 |  65 |   8 |    0 |
| ByteTrack control vs gold | 0.7080 | 0.6478 | 0.7762 | 0.8463 | 0.8738 | 0.7469 | 0.8226 |  89 |  54 |    2 |
| BoT-SORT + ReID vs gold   | 0.7630 | 0.7100 | 0.8205 | 0.8721 | 0.8993 | 0.7906 | 0.8595 |  92 |  26 |    2 |
| ReID vs bạn               | 0.7368 | 0.6830 | 0.7961 | 0.8841 | 0.8731 | 0.7492 | 0.8728 |  82 |  73 |    3 |

## 5. Phân tích

**1. MOTA và IDF1:** IDF1 của annotation là `0.9393`, cao hơn MOTA `0.8726`. Annotation có `0` IDSW, đủ `8` track gold và chỉ `8` FN; điểm mất chủ yếu đến từ `65` FP và bbox lệch. MOTA tổng hợp FP, FN và IDSW nên không mô tả riêng chất lượng giữ identity; cần đọc cùng IDF1 và AssA.

**2. ByteTrack so với ReID:** ReID tăng HOTA `0.7080 -> 0.7630`, IDF1 `0.8738 -> 0.8993` và AssA `0.7762 -> 0.8205`. IDSW vẫn là `2` ở cả hai. ByteTrack có switch tại frame `59` của track gold `4` và frame `94` của track `5`; ReID có switch tại frame `87` của track `5` và frame `113` của track `6`. ReID cải thiện association trung bình nhưng chưa loại bỏ lỗi ở đoạn che khuất/cắt nhau. Đây là system comparison, không cô lập causal effect của ReID vì hai tracker implementation khác nhau.

**3. DetA, FP và FN:** ReID tăng DetA `0.6478 -> 0.7100`, giảm FN `54 -> 26`, nhưng FP tăng `89 -> 92`. Cải thiện lớn nhất là detector/recovery; association cũng tốt hơn qua AssA và IDF1 nhưng vẫn còn fragmentation và IDSW. Lỗi còn lại là kết hợp của detector FP và association.

**4. Một chỗ annotation đúng và ReID sai:** ReID có ghost track `7` trong frame `16–116`, dài `44` frame nhưng không khớp track gold nào. So với gold, annotation có FP `65`, FN `8`, IDSW `0`, tốt hơn ReID với FP `92`, FN `26`, IDSW `2`.

**5. Một chỗ cần xem lại annotation:** Đoạn quanh frame `104–115`, track gold `6`, đáng được kiểm tra vì ReID bị fragmentation quanh các ID model `25` và `32` và có bbox IoU thấp. Tuy nhiên chưa có evidence cho thấy annotation sai: annotation đạt IDF1 `0.9393`, không có IDSW, trong khi ReID có IDF1 `0.8993`, `2` IDSW và `26` FN. Vì vậy giữ nguyên annotation là hợp lý.

## 6. Nếu phải gán thêm 10 clip nữa

Bổ sung vào `GUIDELINE_MINI.md` quy tắc đặt keyframe ở frame bắt đầu/kết thúc, kiểm tra sau occlusion, không để bbox tồn tại trước khi xe xuất hiện hoặc sau khi xe ra khỏi khung, và rà riêng đoạn xe cắt nhau. Quy trình nên có lượt kiểm tra theo ID, lượt kiểm tra frame đầu/cuối và lượt rà các frame bbox thay đổi nhanh. Model chỉ dùng để tìm vùng cần xem lại, không dùng để tự động sửa nhãn.

## 7. Tệp đã nộp

- [x] `annotations/clip_01/gt.txt`
- [x] `annotations/clip_02/gt.txt`
- [ ] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json` chưa có
- [ ] `GUIDELINE_MINI.md` chưa xác nhận
- [x] Các file kết quả trong `outputs/`
- [ ] `reports/review_partner.md` chưa có
- [x] `reports/REPORT.md`
