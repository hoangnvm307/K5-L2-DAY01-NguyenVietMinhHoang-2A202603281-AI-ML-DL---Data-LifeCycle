# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

    ### 1.1. Kết quả dự đoán Top-1

    | Sample ID | `traffic` |
    | Class ID | `468` |
    | Class Name | `cab` (taxi) |
    | Rank | `1` |
    | Score | `0.510915` |
    | Taxonomy | `ImageNet-1K` |

    **1.2. Prediction mô tả cả ảnh hay một vật thể?**

    Prediction mô tả **toàn bộ ảnh**, vì Image Classification gán nhãn cho ảnh thay vì từng vật thể riêng lẻ. Kết quả không chứa tọa độ bounding box hoặc polygon.

    **1.3. Nguồn gốc danh sách lớp**

    Danh sách lớp được định nghĩa sẵn theo taxonomy đi kèm checkpoint, không do mô hình tự tạo. Checkpoint `yolo11n-cls.pt` sử dụng ImageNet-1K (1.000 lớp).

    **1.4. Ý nghĩa của ba trường thông tin**

    | `class_id` | Mã định danh lớp, phục vụ xử lý dữ liệu |
    | `class_name` | Tên lớp, giúp con người đọc hiểu |
    | `taxonomy_name` | Xác định hệ thống phân loại, tránh nhầm lẫn giữa các bộ nhãn |

    **1.5. Quy tắc chọn nhãn khi có nhiều đối tượng**

    Guideline cần quy định tiêu chí chọn **một lớp đại diện**, như đối tượng chính hoặc chủ đề nổi bật theo mục tiêu bài toán. Đồng thời, cần hướng dẫn xử lý các trường hợp nhiều đối tượng cùng nổi bật hoặc không có lớp phù hợp.

    **1.6. Vì sao score không phải ground truth?**

    `score = 0.510915` chỉ là điểm dự đoán của mô hình đối với lớp `cab`, **không bảo đảm kết quả đúng**. Ground truth phải là nhãn chuẩn được xác định và kiểm tra dựa trên guideline, không lấy trực tiếp từ score của mô hình.

## 2. Phát hiện vật thể

    ### Câu 1. Thông tin một prediction

    Chọn record dự đoán đối tượng `person` có score cao nhất trong ảnh `kitchen`.

    | Thuộc tính | Giá trị |
    |---|---|
    | `class_name` | `person` |
    | `score` | `0.912625` |
    | `bbox_xyxy` | `[385.33, 69.24, 498.92, 348.92]` |
    | `bbox_width` | `113.58` px |
    | `bbox_height` | `279.68` px |

    ### Câu 2. Giải thích Bounding Box

    Bounding Box bao quanh người đang đứng ở khu vực giữa bên phải của ảnh nhà bếp.

    - **Góc trên trái:** `(385.33, 69.24)` pixel.
    - **Góc dưới phải:** `(498.92, 348.92)` pixel.
    - **Chiều rộng:** khoảng `113.58` pixel.
    - **Chiều cao:** `279.68` pixel.

    Tọa độ được tính từ góc trên bên trái của ảnh `(0, 0)`. Bounding Box xác định vị trí và kích thước vùng chứa người mà mô hình phát hiện.

    ### Câu 3. So sánh số lượng prediction theo threshold

    | Confidence threshold | Số prediction |
    |---|---|
    | `0.35` | `11` |
    | `0.60` | `6` (lọc từ JSON) |

    Khi tăng threshold từ `0.35` lên `0.60`, số prediction được giữ lại giảm từ 11 xuống 6 theo dữ liệu JSON hiện có.

    Ngưỡng thấp giữ lại nhiều vật thể hơn nhưng có thể xuất hiện nhiều prediction sai. Ngưỡng cao loại bỏ các prediction có score thấp, tuy nhiên cũng có nguy cơ bỏ sót vật thể thực tế.

    ### Câu 4. Công việc của reviewer khi hạ threshold

    Khi hạ threshold, reviewer cần kiểm tra nhiều bounding box hơn, đặc biệt:

    - **False Positive:** Phát hiện nhầm vật thể hoặc nhận diện sai lớp.
    - **Duplicate Boxes:** Nhiều hộp cùng bao quanh một vật thể.
    - **Box không chính xác:** Hộp quá rộng, quá hẹp hoặc lệch khỏi vật thể.
    - **Vật thể bị bỏ sót:** Kiểm tra xem những prediction mới có giúp phát hiện thêm vật thể thực tế hay không.

    Reviewer cần đối chiếu với ảnh gốc và guideline để quyết định giữ, sửa hoặc loại bỏ các prediction không phù hợp.

    ### Câu 5. Quy tắc gán nhãn Bounding Box

    **Quy tắc đề xuất:**

    Annotator phải vẽ bounding box bao sát phần vật thể nhìn thấy được, hạn chế chứa vùng nền không liên quan và không cắt mất các phần vật thể còn nhìn thấy.

    Bounding box phải có tọa độ nằm trong giới hạn ảnh và tuân thủ cùng một quy tắc cho những vật thể tương tự.

    ### Câu 6. Xử lý vật thể bị che khuất hoặc cắt mép

    **Trường hợp quan sát:** Ở mép trái ảnh `kitchen` có một người chỉ xuất hiện một phần cơ thể, chủ yếu là cánh tay và bàn tay gần khu vực bàn bếp. Mô hình dự đoán một bounding box `person` với score `0.610931`.

    **Hướng xử lý:** Escalation – hỏi Lab Coach để xác nhận quy tắc gán nhãn.

    **Lý do:** Phần lớn cơ thể người nằm ngoài khung hình nên cần làm rõ liệu có đủ thông tin để gán nhãn `person` hay không, và bounding box nên bao phần nhìn thấy hay ước lượng toàn bộ cơ thể.

    Vì guideline hiện chưa được xác nhận cho trường hợp này, cần hỏi Lab Coach trước khi đưa ra quy tắc chung, tránh việc các annotator gán nhãn không nhất quán.

## 3. Phân đoạn theo từng đối tượng 

    **Dữ liệu:** `segmentation_predictions.json` | **Ảnh:** `kitchen` | **Model:** `yolo11n-seg.pt`

    ### Câu 1. Thông tin một prediction

    Chọn record `kitchen-001`, mô hình dự đoán một người đang đứng trong khu vực nhà bếp.

    | Thuộc tính | Giá trị |
    |---|---|
    | `instance_id` | `kitchen-001` |
    | `class_name` | `person` |
    | `score` | `0.899318` |
    | `polygon_point_count` | `348` |
    | `polygon_xy` (5 điểm đầu) | `[[446,70], [445,71], [444,71], [443,72], [442,72]]` |

    ### Câu 2. Polygon cung cấp thêm thông tin gì so với Bounding Box?

    Bounding Box chỉ xác định vùng hình chữ nhật bao quanh vật thể, trong khi Polygon mô tả **đường viền chi tiết theo hình dạng thực tế**.

    Polygon giúp xác định vùng pixel thuộc vật thể, phân biệt vật thể với nền và thể hiện chính xác hơn các phần có hình dạng phức tạp như tay, chân hoặc đầu người.

    ### Câu 3. Ý nghĩa của instance_id

    `instance_id` dùng để **phân biệt từng đối tượng riêng biệt** trong kết quả phân đoạn.

    - **Class ID:** Xác định vật thể thuộc loại nào, ví dụ `person = 0`.
    - **Instance ID:** Phân biệt các cá thể cùng lớp, ví dụ `kitchen-001` và `kitchen-009` đều là người nhưng là hai đối tượng khác nhau.
    - **Tracking ID:** Dùng để theo dõi cùng một đối tượng qua nhiều khung hình video.

    Vì vậy, `instance_id` không phải mã lớp và cũng không thể hiện việc theo dõi vật thể theo thời gian.

    ### Câu 4. Quy tắc vẽ biên Mask

    **Quy tắc đề xuất:**

    Annotator cần vẽ đường biên mask bám sát ranh giới phần vật thể nhìn thấy được, không bao gồm vùng nền hoặc các vật thể khác.

    Đối với vật thể có hình dạng phức tạp, cần sử dụng đủ điểm polygon để mô tả chính xác đường viền, đồng thời tuân thủ quy tắc xử lý vùng bị che khuất trong guideline.

    ### Câu 5. Xử lý biên mờ, vật thể chạm nhau hoặc bị che khuất

    **Trường hợp quan sát:** Trong ảnh `kitchen`, người đứng giữa bên phải có phần tay và thân tiếp giáp với các vật dụng trong nhà bếp. Một số vùng ranh giới giữa cơ thể và vật dụng không rõ ràng, có nguy cơ khiến mask bao nhầm vùng nền hoặc vật thể khác.

    **Hướng xử lý:** Sử dụng guideline nếu đã có quy định rõ; thực hiện **escalation – hỏi Lab Coach** nếu chưa xác định được đường biên cần gán nhãn.

    **Lý do:** Việc tự ước lượng vùng không nhìn thấy hoặc xác định đường biên theo cảm tính có thể khiến dữ liệu không nhất quán. Cần thống nhất cách xử lý để các annotator áp dụng cùng một tiêu chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

    **Quy trình production:**

    `Ảnh thô → Guideline → Ground Truth → Huấn luyện → Prediction → QC/Rework`

    | Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
    |---|---|---|---|---|
    | **Phân loại ảnh** | Một nhãn lớp đại diện cho toàn bộ ảnh (`class_id`, `class_name`). | Ảnh `traffic` được dự đoán là `cab` với score `0.510915`, nhưng ảnh giao thông có thể chứa nhiều loại phương tiện nên khó chọn một lớp đại diện. | Quan sát toàn ảnh, chọn một lớp theo taxonomy và guideline. | Kiểm tra nhãn có đúng chủ đề chính và tuân thủ quy tắc chọn lớp không. |
    | **Phát hiện vật thể** | Mỗi vật thể có một nhãn lớp và bounding box `[x_min, y_min, x_max, y_max]`. | Ảnh `kitchen` có người ở mép trái bị cắt một phần cơ thể (`person`, score `0.610931`), gây khó khăn khi xác định bounding box. | Xác định từng vật thể, gán đúng lớp và vẽ bounding box theo guideline. | Kiểm tra sai lớp, hộp lệch, trùng lặp, bỏ sót và trường hợp vật thể bị cắt mép. |
    | **Instance Segmentation** | Mỗi vật thể có một nhãn lớp và mask/polygon mô tả chính xác vùng pixel của riêng vật thể đó. | Ảnh `kitchen` có người đứng giữa (`person`, score khoảng `0.90`), phần tay và thân tiếp giáp với vật dụng, dễ khiến mask bao nhầm vùng nền. | Gán nhãn từng instance, vẽ mask sát đường viền vật thể theo guideline. | Kiểm tra mask có tràn nền, thiếu vùng, sai biên hoặc gộp nhầm các vật thể không. |

## 5. An toàn dữ liệu

    **1. Quy tắc bảo vệ dữ liệu**

    Chỉ sử dụng ảnh công khai được cung cấp trong notebook, không tải lên hoặc chia sẻ dữ liệu cá nhân, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab và GitHub công khai.

    **2. Xử lý dữ liệu ngoài phạm vi**

    Nếu phát hiện ảnh hoặc dữ liệu không đúng phạm vi cho phép, tôi sẽ dừng xử lý ngay và báo cho **Lab Coach hoặc giảng viên phụ trách** để được hướng dẫn, không tự ý sử dụng hoặc chia sẻ dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
