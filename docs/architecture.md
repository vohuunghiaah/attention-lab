# Thiết kế và hợp đồng API

Hai engine độc lập giúp đối chiếu NumPy với PyTorch; UI không triển khai lại attention.
`numpy_attention` trả dataclass chứa raw_scores, scaled_scores (trước mask),
masked_scores, weights, output. Cờ `scale=False` dành cho thí nghiệm so sánh.

`scaled_dot_product_attention(q,k,v,mask=None,causal=False)` trả `(output,weights)`.
Leading dimensions khớp chính xác. Dtype/device phải giống nhau; module có thể
chuyển bằng `.double()` hoặc `.to(device)`. Causal mask tự tạo trên device đầu vào.
Không tự di chuyển tensor hay đổi dtype âm thầm.

`MultiHeadAttention(D,H,seed=...)` tạo các projection riêng. `forward` trả output
và weights; `inspect` trả cả Q/K/V, head_outputs, concatenated và output, dùng
cùng pipeline. Seed tường minh dùng RNG scope để không thay đổi global RNG.
Không truyền seed thì dùng trạng thái RNG hiện tại như module PyTorch thông thường.

Mask `True` nghĩa là giữ key. Mask (Tq,Tk) dùng chung toàn batch/head. Cho MHA,
mask riêng batch là (B,1,Tq,Tk); padding mask là (B,1,1,Tk). Với functional
single-head shape (B,Tq,dk), dùng padding mask shape (B,1,Tk).
Mask hình chữ nhật dùng j ≤ i, không tự suy ra offset của KV cache.

AppTest chạy ứng dụng thực và đổi input/seed/mask/query/head/token/dimension,
đồng thời kiểm tra thông báo lỗi. Đây là smoke test logic UI, không thay thế
kiểm tra giao diện bằng trình duyệt. Lint và pytest chạy trong CI trên Python 3.11/3.12.

`test.ipynb` đã có thay đổi của người dùng trước phiên làm việc và được giữ nguyên.
Các module `attention.py` và `multi_head.py` giữ đường dẫn ban đầu của repository.
