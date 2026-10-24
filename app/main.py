"""Vietnamese Streamlit laboratory driven entirely by real attention results."""

import numpy as np
import pandas as pd
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


def matrix_table(values: Array, key: str) -> None:
    """Show a (rows, columns) matrix with explicit zero-based row/column indices."""
    frame = pd.DataFrame(values, index=pd.Index(range(values.shape[0])))
    st.dataframe(
        frame,
        hide_index=False,
        width="stretch",
        key=key,
        column_config={
            str(j): st.column_config.NumberColumn(str(j), format="%.4f")
            for j in range(values.shape[1])
        },
    )


def equation(formula: str) -> None:
    """Place one mathematical statement in a distinct, readable panel."""
    with st.container(border=True):
        st.latex(formula)


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
        st.caption("Chỉ số bắt đầu từ 0: i là hàng token; cột là thành phần của vector.")
        for column, name, matrix in zip(st.columns(3), ["Q", "K", "V"], [q, k, v], strict=True):
            with column:
                st.subheader(name)
                st.caption(f"{matrix.shape[0]} hàng × {matrix.shape[1]} cột")
                equation(rf"\mathbf{{{name}}} = \mathbf{{X}}\mathbf{{W}}_{{{name}}}")
                matrix_table(matrix, f"matrix_{name}")
        st.caption("Ví dụ toán học cung cấp trực tiếp Q, K, V; đầu vào tùy chỉnh dùng X và W.")
        st.subheader("Bốn bước tính attention")
        st.write(
            "S và S_scaled là điểm số, A là xác suất, O là vector đầu ra. "
            "Trong S, S_scaled và A: hàng i là query, cột j là key. "
            "Trong O: hàng i là query, cột r là thành phần của Value."
        )
        stages = [
            (
                "1 · S — Điểm tương thích",
                r"\mathbf{S}=\mathbf{Q}\mathbf{K}^{\mathsf{T}}",
                r"S_{ij}=\sum_{r=0}^{d_k-1}Q_{ir}K_{jr}",
                result.raw_scores,
                "S",
                "Mỗi ô là tích vô hướng của một query với một key; chưa phải xác suất.",
            ),
            (
                "2 · S_scaled — Điểm sau scaling",
                r"\mathbf{S}_{\mathrm{scaled}}=\frac{\mathbf{S}}{\sqrt{d_k}}",
                rf"(S_{{\mathrm{{scaled}}}})_{{00}}=\frac{{{result.raw_scores[0, 0]:.4f}}}"
                rf"{{\sqrt{{{q.shape[-1]}}}}}={result.scaled_scores[0, 0]:.4f}",
                result.scaled_scores,
                "S_scaled",
                "Chia từng điểm cho căn bậc hai của chiều Q/K. Đây là điểm trước khi áp dụng mask.",
            ),
            (
                "3 · A — Trọng số attention",
                r"\mathbf{A}=\operatorname{softmax}_{\mathrm{key}}(\mathbf{Z})",
                r"A_{ij}=\frac{\exp(Z_{ij}-m_i)}{\sum_{\ell=0}^{T_k-1}\exp(Z_{i\ell}-m_i)}",
                result.weights,
                "A",
                "Z là điểm sau mask. Softmax theo từng hàng; mỗi hàng hợp lệ có tổng bằng 1.",
            ),
            (
                "4 · O — Đầu ra attention",
                r"\mathbf{O}=\mathbf{A}\mathbf{V}",
                r"O_{ir}=\sum_{j=0}^{T_k-1}A_{ij}V_{jr}",
                result.output,
                "O",
                "Mỗi vector đầu ra là tổng các vector Value, "
                "với hệ số lấy từ hàng tương ứng của A.",
            ),
        ]
        for title, formula, detail, values, name, explanation in stages:
            with st.container(border=True):
                st.subheader(title)
                st.write(explanation)
                st.latex(formula)
                st.latex(detail)
                if name == "A":
                    st.latex(r"m_i=\max_{\ell}Z_{i\ell}")
                    if causal:
                        st.latex(
                            r"Z_{ij}=\begin{cases}"
                            r"(S_{\mathrm{scaled}})_{ij} & j\leq i \\ "
                            r"-\infty & j>i\end{cases}"
                        )
                    else:
                        st.latex(r"\mathbf{Z}=\mathbf{S}_{\mathrm{scaled}}")
                    st.caption("Trừ cực đại của hàng trước khi lấy exp để ổn định số học.")
                st.caption(f"{name}: {values.shape[0]} hàng × {values.shape[1]} cột")
                matrix_table(values, f"matrix_{name}")
        with st.expander("Kiểm tra phép tính tại hàng i = 0"):
            terms = r" + ".join(
                rf"({a:.3f}\times {b:.3f})" for a, b in zip(q[0], k[0], strict=True)
            )
            st.latex(rf"S_{{00}}={terms}={result.raw_scores[0, 0]:.6f}")
            st.write("Điểm sau mask và xác suất ứng với từng key j:")
            st.dataframe(
                pd.DataFrame(
                    {"Z[0,j]": result.masked_scores[0], "A[0,j]": result.weights[0]},
                    index=pd.Index(range(len(tokens)), name="j"),
                ),
                hide_index=False,
            )
            st.latex(
                rf"O_{{00}}=\sum_{{j=0}}^{{{len(tokens) - 1}}}A_{{0j}}V_{{j0}}"
                rf"={result.output[0, 0]:.6f}"
            )
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
            heatmap(result.raw_scores, "S · Điểm thô QKᵀ", "raw", rows=labels, columns=labels)
            heatmap(
                result.weights,
                "A · Trọng số attention",
                "weights",
                rows=labels,
                columns=labels,
                probability=True,
            )
        with right:
            heatmap(
                result.scaled_scores,
                "S_scaled · Điểm sau scaling (trước mask)",
                "scaled",
                rows=labels,
                columns=labels,
            )
            heatmap(result.output, "O · Đầu ra AV", "output", rows=labels)
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
            equation(
                r"\mathrm{head}_i=\operatorname{Attention}(\mathbf{Q}_i,\mathbf{K}_i,\mathbf{V}_i)"
            )
            equation(
                r"\operatorname{MHA}(\mathbf{X})="
                r"\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\mathbf{W}_O"
            )
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
