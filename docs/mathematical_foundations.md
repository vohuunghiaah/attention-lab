# Nền tảng toán học

## Vì sao chia cho căn bậc hai?

Nếu các thành phần q và k độc lập, kỳ vọng 0 và phương sai 1 thì tích vô hướng
của dk thành phần có phương sai dk. Chia cho √dk đưa phương sai về 1 theo giả
định này; đó không phải bảo đảm mọi input đều có phương sai 1. Điểm quá lớn làm
softmax bão hòa, khiến việc học khó hơn. Tab Ma trận cho phép so sánh trực tiếp.

## Softmax và mask

Với điểm s, trừ max(s) không đổi xác suất vì tử và mẫu cùng chia cho exp(max(s)).
Mask được áp dụng **trước** softmax để phần còn lại được chuẩn hóa đúng. Mask sau
softmax mà không chuẩn hóa lại làm tổng hàng nhỏ hơn 1.

Hàng toàn −∞ không có phân phối xác suất hợp lệ. Hai backend chọn quy ước trọng
số bằng 0, đầu ra bằng 0. NumPy dùng phép chia với `where`; PyTorch thay điểm của
hàng trống bằng 0 trước softmax rồi xóa mọi vị trí bị che. Như vậy tránh NaN cả
forward lẫn backward. Không gọi softmax trên một hàng toàn −∞ rồi mới sửa NaN.

## Ví dụ số

Q = [[1,1], [0,1], [1,2]], K = V = [[1,0], [0,1], [1,1]].
QKᵀ = [[1,1,2], [0,1,1], [1,2,3]], rồi chia từng phần tử cho √2.

Đặt a = exp(1/√2). Trọng số hàng đầu không mask là [1,1,a]/(2+a).
Vì V có hai cột [1,0,1] và [0,1,1], đầu ra hàng đó là
[(1+a)/(2+a), (1+a)/(2+a)]. Khi có causal mask, hàng đầu chỉ được nhìn key 0,
nên trọng số là [1,0,0] và đầu ra là [1,0]. Đây là kỳ vọng độc lập cho test.

## Multi-Head

D chia hết H. Mỗi phép chiếu XW có dạng (B,T,D), reshape thành (B,T,H,D/H),
transpose thành (B,H,T,D/H). Attention chạy trên hai trục cuối, không trộn head.
Sau AV, transpose về (B,T,H,D/H), ghép thành (B,T,D), rồi nhân W_O.
Trong PyTorch, `nn.Linear` lưu trọng số dạng (out,in), nên thực hiện X weightᵀ.
Các phép chiếu không dùng bias để khớp trực tiếp công thức XW.

Nhiều head tạo khả năng học các phép chiếu khác nhau; các head ngẫu nhiên không
đồng nghĩa đã học cú pháp, ngữ nghĩa hay quan hệ token. Không suy diễn ý nghĩa
ngôn ngữ từ màu sắc trong ứng dụng này.
