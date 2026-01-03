# HƯỚNG DẪN TOÀN TẬP: DỰ ÁN MINE (MUTUAL INFORMATION NEURAL ESTIMATION)

- **Môn học:** Lý thuyết thông tin - HUST
- **Nhóm thực hiện:** NHÓM 2
  - Phạm Hồng Duy Minh - 20234025
  - Trần Phi Anh Nhật - 20234029
  - Hoàng Đức Trung - 20234041
  - Trần Độ - 20233999
- **Đề tài:** Ước lượng thông tin tương hỗ bằng mạng Neuron.


---

## MỤC LỤC
1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Cấu trúc mã nguồn](#2-cấu-trúc-mã-nguồn)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Hướng dẫn chạy chương trình](#4-hướng-dẫn-chạy-chương-trình)
5. [Phân tích kết quả](#5-phân-tích-kết-quả)

---

## 1. GIỚI THIỆU DỰ ÁN
Mục tiêu của dự án là xây dựng một mô hình Deep Learning để ước lượng lượng **Thông tin tương hỗ (Mutual Information - MI)** giữa các biến ngẫu nhiên liên tục.
- **Phương pháp:** Sử dụng mạng Neural để tối ưu hóa cận dưới Donsker-Varadhan (MINE).
- **Dữ liệu kiểm chứng:** Dữ liệu phân phối Gaussian đa chiều (vì loại dữ liệu này có thể tính được MI chính xác bằng công thức toán học để so sánh).

---

## 2. CẤU TRÚC MÃ NGUỒN
Trong thư mục dự án, các file có chức năng như sau:

* **`data/gaussian.py`**:
    * Chứa hàm `sample_correlated_gaussian`: Tạo ra dữ liệu giả lập (X, Y) có hệ số tương quan $\rho$.
    * Chứa hàm `true_mi_gaussian`: Tính giá trị MI lý thuyết (Ground Truth) để làm chuẩn so sánh.
* **`models/network.py`**:
    * Định nghĩa mạng Neural (`MineNetwork`).
    * Cấu trúc: Input (X,Y) -> Lớp ẩn 1 -> Lớp ẩn 2 -> Output (Scalar).
* **`utils/mine_loss.py`**:
    * Chứa hàm Loss quan trọng nhất dựa trên công thức Donsker-Varadhan.
    * Thực hiện kỹ thuật **Shuffle (tráo đổi)** biến Y để tạo phân phối biên phục vụ tính toán.
* **`main.py`**:
    * File trung tâm điều khiển mọi thứ: Cài đặt tham số, chạy vòng lặp huấn luyện, tính toán Loss và vẽ biểu đồ kết quả.
* **`requirements.txt`**: Danh sách các thư viện cần cài đặt.

---

## 3. CÀI ĐẶT MÔI TRƯỜNG
Để chạy được code, bạn cần thực hiện đúng các bước sau:

- **Bước 1:** Mở thư mục dự án trong VS Code.
- **Bước 2:** Mở Terminal (Phím tắt: `Ctrl + J` hoặc `Ctrl + ` ` `).
- **Bước 3:** Cài đặt các thư viện phụ thuộc bằng lệnh:

    pip install -r requirements.txt

- Đợi màn hình báo "Successfully installed..." là thành công.

---

## 4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
Sau khi cài đặt xong, bạn chạy lệnh sau để bắt đầu huấn luyện mô hình:

    python main.py

- **Terminal sẽ hiện thông báo bắt đầu: Start Training MINE... True MI: ...
- **Chương trình chạy 5000 bước huấn luyện (mất khoảng 10-20 giây).
- **Các thông số ước lượng sẽ hiện liên tục: Step 500, Estimated MI: ...
- **Khi chạy xong, dòng chữ Training done. Result saved to result.png sẽ hiện ra.

---

## 5. PHÂN TÍCH KẾT QUẢ
Sau khi chạy xong, hãy mở file ảnh result.png vừa được tạo ra trong thư mục dự án.

- **Cách đọc biểu đồ:

    - ***Đường nét đứt màu Đỏ (True MI): Đây là đáp án đúng (tính bằng toán học).
    - ***Đường màu Xanh/Cam (MINE Estimated): Đây là kết quả do mạng Neuron học được.

- **Đánh giá:

    - ***Nếu đường màu Cam dao động xung quanh và bám sát đường màu Đỏ -> Mô hình thành công.
    - ***Nếu đường màu Cam nằm xa đường màu Đỏ -> Mô hình chưa tốt (cần huấn luyện thêm).



