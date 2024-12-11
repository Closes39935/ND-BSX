import cv2
import tkinter as tk
from tkinter import Label, Button, messagebox, Frame, Entry
from PIL import Image, ImageTk
from datetime import datetime
import DetectChars
import DetectPlates


class LicensePlateApp:
    def __init__(self, window, window_title, width=900, height=600):
        self.window = window
        self.window.title(window_title)

        # Đặt kích thước cố định cho cửa sổ
        self.window.geometry(f"{width}x{height}")
        self.window.resizable(False, False)

        # Đặt vị trí cửa sổ ở giữa màn hình
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.window.geometry(f"+{x}+{y}")

        # Khởi tạo camera
        self.video_source = 0  # 0 cho webcam mặc định
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera.")
            self.window.destroy()
            return

        # --- Giao diện chính ---
        self.main_frame = Frame(window)
        self.main_frame.pack(fill="both", expand=True)

        # Khung hiển thị video camera
        self.video_frame = Frame(self.main_frame, width=600, height=400)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = Label(self.video_frame, text="Camera feed")
        self.video_label.pack()

        # Khung hiển thị biển số và thông tin
        self.info_frame = Frame(self.main_frame)
        self.info_frame.grid(row=0, column=1, padx=10, pady=10)

        # Khung biển số
        self.plate_image_label = Label(self.info_frame, text="License Plate")
        self.plate_image_label.pack()

        self.plate_image_canvas = Label(self.info_frame, bg="white", width=30, height=10)
        self.plate_image_canvas.pack(pady=10)

        # Thông tin chi tiết
        Label(self.info_frame, text="Number Plate:").pack()
        self.number_plate_entry = Entry(self.info_frame, width=25, state="readonly")
        self.number_plate_entry.pack(pady=5)

        Label(self.info_frame, text="Date:").pack()
        self.date_entry = Entry(self.info_frame, width=25, state="readonly")
        self.date_entry.pack(pady=5)

        Label(self.info_frame, text="Time:").pack()
        self.time_entry = Entry(self.info_frame, width=25, state="readonly")
        self.time_entry.pack(pady=5)

        # Khung điều khiển
        self.control_frame = Frame(self.main_frame)
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_button = Button(self.control_frame, text="Start", command=self.detect_plate, width=15)
        self.start_button.pack(side="left", padx=10)

        self.exit_button = Button(self.control_frame, text="Exit", command=self.window.quit, width=15)
        self.exit_button.pack(side="right", padx=10)

        # Bắt đầu cập nhật video
        self.update_video()
        self.window.mainloop()

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            # Chuyển đổi hình ảnh thành ảnh PIL để hiển thị
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
        else:
            print("Không thể đọc khung hình từ camera.")

        # Gọi hàm update sau mỗi 10ms
        self.window.after(10, self.update_video)

    def detect_plate(self):
        # Lấy ngày và giờ hiện tại
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Lấy biển số xe từ camera
        ret, frame = self.vid.read()
        if ret:
            licPlateStr, plate_img = self.recognize_license_plate(frame)
            if licPlateStr:
                # Cập nhật thông tin vào giao diện
                self.number_plate_entry.config(state="normal")
                self.number_plate_entry.delete(0, "end")
                self.number_plate_entry.insert(0, licPlateStr)
                self.number_plate_entry.config(state="readonly")

                self.date_entry.config(state="normal")
                self.date_entry.delete(0, "end")
                self.date_entry.insert(0, current_date)
                self.date_entry.config(state="readonly")

                self.time_entry.config(state="normal")
                self.time_entry.delete(0, "end")
                self.time_entry.insert(0, current_time)
                self.time_entry.config(state="readonly")

                # Hiển thị ảnh biển số
                if plate_img is not None:
                    plate_photo = ImageTk.PhotoImage(image=Image.fromarray(plate_img))
                    self.plate_image_canvas.config(image=plate_photo)
                    self.plate_image_canvas.image = plate_photo
                else:
                    self.plate_image_canvas.config(image="", text="Không tìm thấy biển số")
            else:
                messagebox.showinfo("Kết quả", "Không phát hiện biển số xe.")
        else:
            messagebox.showerror("Lỗi", "Không thể đọc khung hình từ camera.")

    def recognize_license_plate(self, img):
        # Tải dữ liệu và đào tạo mô hình KNN
        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
        if not blnKNNTrainingSuccessful:
            print("\nerror: KNN training was not successful\n")
            return None, None

        # Phát hiện biển số
        listOfPossiblePlates = DetectPlates.detectPlatesInScene(img)
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        if len(listOfPossiblePlates) == 0:
            print("\nNo license plates detected\n")
            return None, None
        else:
            # Sắp xếp theo số ký tự, lấy biển số tốt nhất
            listOfPossiblePlates.sort(key=lambda plate: len(plate.strChars), reverse=True)
            best_plate = listOfPossiblePlates[0]

            # Kiểm tra nếu không có ký tự trong biển số
            if len(best_plate.strChars) == 0:
                print("\nNo characters detected\n")
                return None, None

            # Cắt vùng biển số từ ảnh
            plate_img = cv2.cvtColor(best_plate.imgPlate, cv2.COLOR_BGR2RGB)
            return best_plate.strChars, plate_img

    def __del__(self):
        # Giải phóng camera khi không sử dụng
        if self.vid.isOpened():
            self.vid.release()


# Khởi chạy ứng dụng
def main():
    root = tk.Tk()
    app = LicensePlateApp(root, "Nhận diện biển số xe"
                                )


if __name__ == "__main__":
    main()
