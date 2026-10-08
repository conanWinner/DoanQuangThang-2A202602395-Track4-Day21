# CP6 — Bài nói 3 phút và chuẩn bị vấn đáp

Đoàn Quang Thắng · 2A202602395 · Topic A. Tài liệu chuẩn bị; không phải bằng chứng đã trình bày hoặc tập nói.

## Mở sẵn trước khi trình bày

Mở [slide PDF 4 trang](../output/pdf/day21_topic_a_presentation.pdf), [REPORT.md](REPORT.md), [biểu đồ KITTI](../results/figures/topic_a_completed_20261008/sweep_kitti_mini_angles.png) và [ảnh failure](../results/figures/topic_a_completed_20261008/fail_01_nuscenes_mini_subset_scene-1094_015_yaw_-3.png). Trái là baseline, phải là yaw −3°.

## 0:00–0:35 — Câu hỏi và phương pháp

“Em chọn Topic A: kiểm tra độ nhạy khi calibration giữa LiDAR và camera bị lệch. LiDAR cho vị trí điểm trong không gian; calibration quyết định điểm đó rơi vào pixel nào trên ảnh. Em muốn biết lệch một độ có đáng kể không, và chỉ số đơn giản có phát hiện được không.
Em chạy 20 frame KITTI, 80 frame nuScenes và 5 frame synthetic. Mỗi frame có 43 cấu hình; mỗi lần chỉ đổi một góc hoặc một hướng dịch chuyển, giữ nguyên dữ liệu và seed 2026.”

## 0:35–1:25 — Kết quả chính

“Mời nhìn đường yaw trong biểu đồ. Trên KITTI, lệch +1° làm điểm dịch trung bình 15,42 pixel; tỷ lệ đúng box giảm từ 99,56% xuống 92,85%. Lệch +3° làm dịch 45,82 pixel và tỷ lệ đúng box còn 75,92%.
Trong khi đó tỷ lệ điểm nằm trong ảnh gần như không đổi, khoảng 15,74–15,76%. Do vậy điểm vẫn ở trong ảnh chưa có nghĩa là calibration đúng.
Trên nuScenes, +1° làm dịch 25,88 pixel. Không so trực tiếp hai sensor như cùng điều kiện: camera, tiêu cự theo pixel, cảnh và quy ước trục khác nhau. Độ lệch pixel chỉ đo điểm còn nhìn thấy ở cả hai lần, nên em báo thêm coverage để không che mất điểm rơi khỏi ảnh.”

## 1:25–2:15 — Failure case

“Em thử score khoảng cách từ điểm chiếu đến biên Canny trên ảnh. Ngưỡng lấy từ baseline nhóm calibration, không chỉnh trên nhóm test.
Ảnh failure là frame đêm scene-1094_015. Gây lệch yaw −3° làm điểm dịch trung bình 78,65 pixel, tỷ lệ đúng box còn 72,44%. Nhưng score 0,014708 thấp hơn ngưỡng 0,036728 nên không báo lỗi.
Đây là lỗi của cách đo: gần một biên ảnh chưa chắc là gần đúng biên vật thể. Score Canny và tỷ lệ điểm trong ảnh đều có thể bỏ sót drift. Em không kết luận ánh sáng tối là nguyên nhân duy nhất vì chưa có thí nghiệm tách riêng yếu tố đó.”

## 2:15–3:00 — Kết luận và triển khai

“Với ADAS, em đề xuất ghi log coverage, chất lượng ảnh, timestamp và score cùng nhau. Hai metric cần calibration chuẩn hoặc nhãn box hiện chỉ dùng để benchmark, chưa phải tín hiệu vận hành sẵn có.
Bước tiếp theo là ghép biên độ sâu với biên ảnh, kiểm tra theo thời gian và đánh giá trên scene độc lập. Hiện tại mới chứng minh độ nhạy với sai lệch giả lập.
Code qua 13 test, benchmark đủ 4.515 dòng. Kaggle phiên bản 1 đã COMPLETE; chạy lại từ bản clone GitHub tái tạo cùng số liệu. AI hỗ trợ code và báo cáo được khai báo rõ, không tạo ảnh hay số liệu giả.”

## Trả lời nhanh câu hỏi thường gặp

| Câu hỏi | Trả lời |
|---|---|
| Hai hàm projection làm gì? | `velo_to_cam` đổi điểm LiDAR sang camera rectified bằng biến đổi đồng nhất. `cam_to_image` nhân P2, chia mẫu số rồi bỏ NaN/Inf, depth nhỏ hoặc âm và pixel ngoài ảnh. |
| Test điểm (10,0,0) cho gì? | Với synthetic frame 000000: depth ≈9,7273 m, pixel ≈(613,964;175,007). |
| “Đúng box” lấy mẫu số nào? | Cố định điểm thuộc box 3D và nhìn thấy ở baseline, chỉ xét object có ≥5 điểm. Điểm biến mất khỏi ảnh sau perturb vẫn là miss. |
| Coverage là gì? | Số điểm nhìn thấy ở cả baseline và lần lệch chia số điểm nhìn thấy baseline. Đọc kèm độ lệch pixel. |
| Bracket lệch 1° có tự phát hiện được không? | Chưa đáng tin với score này. Yaw +1°: KITTI phát hiện 3/10 test frame, nuScenes 5/40; đây là độ nhạy giả lập, không phải chứng nhận vận hành. |
| Phát hiện ở khoảng cách nào? | Có demo object 4,88/24,99/59,77 m, nhưng chưa đánh giá recall theo từng khoảng cách; không suy ra phạm vi phát hiện từ ba ảnh. |
| Claim sai khi nào? | Với camera/cảnh/mật độ điểm khác, độ lệch pixel và box ratio có thể khác. Ngưỡng học từ tập mini không bảo đảm cho scene mới. |
| Vì sao nuScenes khác KITTI? | Tiêu cự theo pixel, ảnh, sensor, cảnh và nhãn khác. Box 2D nuScenes sinh từ nhãn 3D nên không độc lập. Không quy mọi khác biệt cho số beam. |
| Có rò rỉ nhóm test khi chọn ngưỡng không? | Ngưỡng chỉ dùng baseline calibration. Tuy vậy chia frame trong cùng scene nên còn tương quan; chưa phải holdout scene độc lập. |
| Khi không đủ score thì làm sao? | Dưới 30 mẫu nhìn thấy: đánh dấu không đủ score và tính là bỏ sót trong tỷ lệ tổng; báo riêng tỷ lệ có điều kiện. Lần chạy này không có trường hợp thiếu score. |
| Latency gồm gì? | Chỉ hai hàm projection trên điểm nạp sẵn, 5 warmup và 30 lần. Không gồm I/O, vẽ hoặc metric; local và Kaggle khác phần cứng/thread. |
| Đã trình bày trước lớp chưa? | Tài liệu này chỉ chuẩn bị; học viên tự xác nhận việc đã tập nói hoặc trình bày. |

## Tự luyện trước khi vấn đáp

- Đọc bài nói, bấm giờ và rút ngắn cho vừa 3 phút; thời gian trên là phân bổ dự kiến, chưa phải thời gian đo.
- Tự giải thích ba số 15,42 pixel, 92,85% và 98,72% ở hàng KITTI yaw +1°.
- Chỉ trên ảnh failure chỗ điểm dịch khỏi vật thể; giải thích vì sao score vẫn thấp.
- Nếu không được gọi trình bày, giảng viên chấm phần này theo REPORT theo rubric; không cần tự tạo bằng chứng đã trình bày.
