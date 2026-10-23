"""Vietnamese Streamlit laboratory driven entirely by real attention results."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from attention_lab import MultiHeadAttention, mathematical_example, numpy_attention
from attention_lab.numpy_attention import Array
from attention_lab.playground import token_projections


def heatmap(
    values: Array,
    title: str,
    key: str,
    *,
    rows: list[str] | None = None,
    columns: list[str] | None = None,
    probability: bool = False,
) -> None:
    """Render a numerical matrix (rows, columns) with exact-value hover labels."""
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=columns,
            y=rows,
            colorscale="Teal",
            zmin=0 if probability else None,
            zmax=1 if probability else None,
            texttemplate="%{z:.2f}" if values.size <= 144 else None,
            hovertemplate="Hàng: %{y}<br>Cột: %{x}<br>Giá trị: %{z:.8f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=350,
        margin=dict(l=20, r=20, t=55, b=30),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", key=key)


def main() -> None:
    """Build controls and show the NumPy and PyTorch calculation pipelines."""
    st.set_page_config(page_title="Attention Lab", page_icon="🔬", layout="wide")
    st.title("Attention Lab")
    st.markdown("### Khám phá từng phép tính bên trong Transformer")
    st.caption("Chạy cục bộ · NumPy & PyTorch · Không cần API · Các phép chiếu chưa huấn luyện")
    with st.sidebar:
        st.header("Bàn thí nghiệm")
        mode = st.radio("Nguồn đầu vào", ["Ví dụ toán học", "Chuỗi token tùy chỉnh"])
        causal = st.toggle("Causal attention", value=False)
        seed = int(st.number_input("Random seed", 0, 100000, 42))
        if mode == "Chuỗi token tùy chỉnh":
            text = st.text_input("Chuỗi token (phân tách bằng khoảng trắng)", "Tôi học attention")
            d_model = st.select_slider("Chiều embedding", [4, 8, 16, 32], value=8)
            tokens = text.split()
            if not tokens or len(tokens) > 24:
                st.error("Vui lòng nhập từ 1 đến 24 token.")
                st.stop()
            projections = token_projections(tokens, d_model, seed)
            q, k, v = projections[-3:]
        else:
            q, k, v = mathematical_example()
            tokens = ["Token 1", "Token 2", "Token 3"]
        st.info("Seed điều khiển đầu vào tùy chỉnh và Multi-Head; ví dụ toán học luôn cố định.")
    labels = [f"{i + 1} · {token}" for i, token in enumerate(tokens)]
    result = numpy_attention(q, k, v, causal=causal)
    unscaled = numpy_attention(q, k, v, causal=causal, scale=False)
    cols = st.columns(3)
    cols[0].metric("Số token", len(tokens))
    cols[1].metric("Chiều Q / K", q.shape[-1])
    cols[2].metric("Chế độ", "Nhân quả" if causal else "Hai chiều")
    fundamentals, visualizer, masking, multihead = st.tabs(
        ["01 · Nền tảng", "02 · Ma trận", "03 · Causal mask", "04 · Multi-Head"]
    )
    with fundamentals:
        st.header("Từ Q, K, V đến đầu ra")
        st.write(
            "Query (Q) là truy vấn; Key (K) dùng để so khớp; Value (V) chứa thông tin "
            "được tổng hợp. Tích vô hướng Q với từng K tạo điểm tương thích."
        )
        st.latex(r"Q=XW_Q,\quad K=XW_K,\quad V=XW_V")
        st.latex(r"S=QK^\top,\quad S_{scaled}=S/\sqrt{d_k},\quad A=softmax(S_{scaled}),\quad O=AV")
        for column, name, matrix in zip(st.columns(3), ["Q", "K", "V"], [q, k, v], strict=True):
            with column:
                st.subheader(f"{name} · {matrix.shape}")
                st.dataframe(matrix, width="stretch")
        st.write(
            "Softmax được tính theo từng hàng (trục key). Chia cho √dₖ giúp hạn chế "
            "điểm số tăng theo chiều vector. Trừ giá trị lớn nhất trước khi lấy exp "
            "giúp softmax ổn định về số học."
        )
        st.latex(r"A_{ij}=\frac{\exp(s_{ij}-m_i)}{\sum_j\exp(s_{ij}-m_i)},\quad m_i=\max_j s_{ij}")
        terms = " + ".join(f"({a:.3f} × {b:.3f})" for a, b in zip(q[0], k[0], strict=True))
        st.code(
            f"S[0,0] = {terms} = {result.raw_scores[0, 0]:.6f}\n"
            f"S_scaled[0,0] = {result.raw_scores[0, 0]:.6f} / √{q.shape[-1]} "
            f"= {result.scaled_scores[0, 0]:.6f}"
        )
        st.write("Hàng đầu tiên sau mask và softmax:")
        st.dataframe({"Điểm sau mask": result.masked_scores[0], "Trọng số": result.weights[0]})
        output_terms = " + ".join(
            f"({a:.4f} × {b:.4f})" for a, b in zip(result.weights[0], v[:, 0], strict=True)
        )
        st.code(f"O[0,0] = {output_terms} = {result.output[0, 0]:.6f}")
        if mode == "Chuỗi token tùy chỉnh":
            st.warning(
                "Embedding và phép chiếu là ngẫu nhiên, chưa huấn luyện. Trọng số "
                "chỉ minh họa toán học, không phản ánh ngữ nghĩa đáng tin cậy. "
                "Token lặp lại có cùng embedding; chưa thêm positional encoding."
            )
            with st.expander("X và các ma trận phép chiếu (ánh xạ token bằng SHA-256)"):
                for name, matrix in zip(["X", "W_Q", "W_K", "W_V"], projections[:4], strict=True):
                    st.write(f"{name} · {matrix.shape}")
                    st.dataframe(matrix)
    with visualizer:
        st.header("Quan sát phân phối attention")
        query = st.selectbox(
            "Chọn query token", range(len(tokens)), format_func=lambda i: labels[i]
        )
        left, right = st.columns(2)
        with left:
            heatmap(result.raw_scores, "Điểm thô QKᵀ", "raw", rows=labels, columns=labels)
            heatmap(
                result.weights,
                "Trọng số attention",
                "weights",
                rows=labels,
                columns=labels,
                probability=True,
            )
        with right:
            heatmap(
                result.scaled_scores,
                "Điểm đã chia √dₖ (trước mask)",
                "scaled",
                rows=labels,
                columns=labels,
            )
            heatmap(result.output, "Đầu ra AV", "output", rows=labels)
        chart = go.Figure()
        chart.add_bar(x=labels, y=result.weights[query], name="Có scaling")
        chart.add_bar(x=labels, y=unscaled.weights[query], name="Không scaling")
        chart.update_layout(
            title=f"Query: {labels[query]}",
            barmode="group",
            yaxis_title="Xác suất",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(chart, width="stretch", key="comparison")
        st.dataframe({"Query": labels, "Tổng xác suất": result.weights.sum(axis=-1)})
        with st.expander("Giá trị số chi tiết"):
            st.dataframe(result.weights)
            st.dataframe(result.output)
    with masking:
        st.header("Chặn thông tin từ tương lai")
        st.write(
            "Ở vị trí i, causal attention chỉ cho phép key j ≤ i. Mask được áp dụng "
            "trước softmax: điểm bị chặn thành −∞ nên xác suất bằng 0."
        )
        allowed = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
        left, right = st.columns(2)
        with left:
            heatmap(
                allowed.astype(float),
                "Causal mask · 1 = được phép",
                "mask",
                rows=labels,
                columns=labels,
                probability=True,
            )
            heatmap(
                numpy_attention(q, k, v).weights,
                "Attention hai chiều",
                "bidirectional",
                rows=labels,
                columns=labels,
                probability=True,
            )
        masked = numpy_attention(q, k, v, causal=True)
        with right:
            heatmap(
                masked.weights,
                "Attention nhân quả",
                "causal",
                rows=labels,
                columns=labels,
                probability=True,
            )
            st.metric("Tổng xác suất tại vị trí tương lai", f"{masked.weights[~allowed].sum():.8f}")
        st.caption(
            "Nếu một hàng bị che hoàn toàn, thư viện trả trọng số và đầu ra bằng 0. "
            "Hàng đó không phải phân phối xác suất chuẩn hóa."
        )
    with multihead:
        st.header("Các head tính toán song song")
        left, middle, right = st.columns(3)
        count = left.slider("Số token cho Multi-Head", 1, 16, 4)
        width = middle.selectbox("d_model", [4, 8, 12, 16, 32], index=1)
        heads = right.selectbox("Số head", [1, 2, 3, 4, 8], index=1)
        st.caption(
            "Multi-Head dùng đầu vào ngẫu nhiên riêng; dùng chung seed và nút Causal ở thanh bên."
        )
        if width % heads:
            st.error("d_model phải chia hết cho số head. Hãy chọn tổ hợp khác.")
        else:
            model = MultiHeadAttention(width, heads, seed=seed)
            x = torch.randn(1, count, width, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                details = model.inspect(x, causal=causal)
            st.latex(r"head_i=Attention(Q_i,K_i,V_i),\quad MHA=Concat(head_1,\ldots,head_h)W_O")
            st.write(
                f"X: {tuple(x.shape)} · d_head = {width // heads} · "
                f"Q/K/V sau tách: {tuple(details.q.shape)}"
            )
            st.warning(
                "Các head khởi tạo độc lập chưa được huấn luyện; không đảm bảo "
                "học được mẫu ngôn ngữ có ý nghĩa."
            )
            for head in range(heads):
                with st.expander(f"Head {head + 1} · chiều {width // heads}", expanded=True):
                    a, b = st.columns(2)
                    with a:
                        heatmap(
                            details.weights[0, head].numpy(),
                            "Trọng số attention",
                            f"head{head}",
                            probability=True,
                        )
                    with b:
                        st.write("Đầu ra của head")
                        st.dataframe(details.head_outputs[0, head].numpy())
                    with st.expander("Xem Q, K, V của head"):
                        for name, tensor in [("Q", details.q), ("K", details.k), ("V", details.v)]:
                            st.write(name)
                            st.dataframe(tensor[0, head].numpy())
            a, b = st.columns(2)
            with a:
                st.subheader("Ghép các head")
                st.dataframe(details.concatenated[0].numpy())
            with b:
                st.subheader("Đầu ra sau phép chiếu W_O")
                st.dataframe(details.output[0].numpy())


if __name__ == "__main__":
    main()
