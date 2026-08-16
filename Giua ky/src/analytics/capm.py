"""Mô hình định giá tài sản vốn (CAPM)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analytics.metrics import annual_return, daily_rf
from src.config import RISK_FREE_ANNUAL, TRADING_DAYS


@dataclass
class CAPMResult:
    alpha_daily: float
    alpha_annual: float
    beta: float
    t_alpha: float
    t_beta: float
    p_alpha: float
    p_beta: float
    r_squared: float
    n_obs: int
    resid_std_annual: float          # rủi ro phi hệ thống
    systematic_share: float          # tỷ trọng rủi ro hệ thống trong tổng phương sai
    expected_return: float           # lợi suất yêu cầu theo CAPM (trung bình số học, năm)
    realized_return: float           # lợi suất thực tế đã đạt (CAGR, năm)
    realized_arithmetic: float       # lợi suất thực tế theo trung bình số học (năm)
    treynor: float
    market_return_annual: float      # trung bình số học của lợi suất thị trường (năm)
    market_return_cagr: float        # lợi suất kép của thị trường (năm)
    rf_annual: float
    data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    model: object = field(repr=False, default=None)

    def to_frame(self) -> pd.DataFrame:
        rows = {
            "Beta (β)": self.beta,
            "Alpha ngày": self.alpha_daily,
            "Alpha năm (Jensen)": self.alpha_annual,
            "t-stat của β": self.t_beta,
            "t-stat của α": self.t_alpha,
            "p-value của α": self.p_alpha,
            "R²": self.r_squared,
            "Số quan sát": self.n_obs,
            "Rủi ro phi hệ thống (năm)": self.resid_std_annual,
            "Tỷ trọng rủi ro hệ thống": self.systematic_share,
            "Lợi suất yêu cầu theo CAPM (năm)": self.expected_return,
            "Lợi suất thực tế — trung bình số học (năm)": self.realized_arithmetic,
            "Lợi suất thực tế — lợi suất kép CAGR (năm)": self.realized_return,
            "Tỷ số Treynor": self.treynor,
        }
        return pd.DataFrame({"Giá trị": rows})


def align_returns(stock_ret: pd.Series, market_ret: pd.Series) -> pd.DataFrame:
    """Ghép hai chuỗi lợi suất theo ngày giao dịch chung."""
    df = pd.concat(
        [stock_ret.rename("stock"), market_ret.rename("market")], axis=1
    ).dropna()
    return df


def estimate_capm(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    rf_annual: float = RISK_FREE_ANNUAL,
) -> CAPMResult:
    df = align_returns(stock_ret, market_ret)
    rf_d = daily_rf(rf_annual)
    df["ex_stock"] = df["stock"] - rf_d
    df["ex_market"] = df["market"] - rf_d

    X = sm.add_constant(df["ex_market"])
    model = sm.OLS(df["ex_stock"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )

    alpha_d = float(model.params["const"])
    beta = float(model.params["ex_market"])
    resid_std_annual = float(model.resid.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # CAPM là mô hình một kỳ nên kỳ vọng dùng trung bình số học; CAGR để riêng cho
    # hiệu quả thực tế, hai đại lượng lệch nhau ~σ²/2 nên không trộn vào cùng phép so sánh.
    mkt_arith = float(df["market"].mean()) * TRADING_DAYS
    realized_arith = float(df["stock"].mean()) * TRADING_DAYS
    mkt_cagr = annual_return(df["market"])
    realized_cagr = annual_return(df["stock"])
    expected = rf_annual + beta * (mkt_arith - rf_annual)

    total_var = float(df["stock"].var(ddof=1))
    sys_var = float(beta ** 2 * df["market"].var(ddof=1))
    sys_share = sys_var / total_var if total_var else float("nan")

    treynor = (realized_arith - rf_annual) / beta if beta else float("nan")

    return CAPMResult(
        alpha_daily=alpha_d,
        alpha_annual=(1 + alpha_d) ** TRADING_DAYS - 1,
        beta=beta,
        t_alpha=float(model.tvalues["const"]),
        t_beta=float(model.tvalues["ex_market"]),
        p_alpha=float(model.pvalues["const"]),
        p_beta=float(model.pvalues["ex_market"]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        resid_std_annual=resid_std_annual,
        systematic_share=sys_share,
        expected_return=expected,
        realized_return=realized_cagr,
        realized_arithmetic=realized_arith,
        treynor=treynor,
        market_return_annual=mkt_arith,
        market_return_cagr=mkt_cagr,
        rf_annual=rf_annual,
        data=df,
        model=model,
    )


def rolling_beta(
    stock_ret: pd.Series, market_ret: pd.Series, window: int = 126
) -> pd.Series:
    """Beta trượt — cho thấy độ nhạy thị trường thay đổi theo thời gian."""
    df = align_returns(stock_ret, market_ret)
    cov = df["stock"].rolling(window).cov(df["market"])
    var = df["market"].rolling(window).var()
    return (cov / var).dropna().rename("beta")


def security_market_line(
    result: CAPMResult, betas: np.ndarray | None = None
) -> pd.DataFrame:
    """Toạ độ đường thị trường chứng khoán (SML) để vẽ đồ thị."""
    if betas is None:
        betas = np.linspace(0, max(2.0, result.beta * 1.5), 50)
    er = result.rf_annual + betas * (result.market_return_annual - result.rf_annual)
    return pd.DataFrame({"beta": betas, "expected_return": er})


def interpret(result: CAPMResult, ticker: str) -> list[str]:
    """Diễn giải kết quả bằng lời — dùng cho dashboard, chatbot và báo cáo."""
    b, a = result.beta, result.alpha_annual
    msgs = []
    if b > 1.05:
        msgs.append(
            f"β = {b:.2f} > 1: {ticker} biến động mạnh hơn thị trường. "
            f"VNINDEX tăng/giảm 1% thì {ticker} tăng/giảm trung bình {b:.2f}%."
        )
    elif b < 0.95:
        msgs.append(
            f"β = {b:.2f} < 1: {ticker} phòng thủ hơn thị trường, "
            f"biến động chỉ khoảng {b:.2f}% khi thị trường biến động 1%."
        )
    else:
        msgs.append(f"β = {b:.2f} ≈ 1: {ticker} biến động gần như cùng nhịp thị trường.")

    significant = result.p_alpha < 0.05
    sig = "có ý nghĩa thống kê" if significant else "chưa có ý nghĩa thống kê"
    msgs.append(
        f"Alpha Jensen = {a * 100:.2f}%/năm ({sig} ở mức 5%, p = {result.p_alpha:.3f}). "
        + (
            ("Cổ phiếu tạo thêm lợi suất vượt mức bù rủi ro thị trường."
             if a > 0 else
             "Cổ phiếu chưa bù đắp đủ rủi ro hệ thống mà nhà đầu tư gánh chịu.")
            if significant else
            "Về mặt kinh tế con số này lớn, nhưng chưa đủ cơ sở thống kê để khẳng định nó khác 0 "
            "— nghĩa là chưa thể kết luận cổ phiếu thực sự tạo ra lợi suất bất thường."
        )
    )
    msgs.append(
        f"R² = {result.r_squared:.2f}: khoảng {result.r_squared * 100:.0f}% biến động của "
        f"{ticker} được giải thích bởi thị trường; phần còn lại là rủi ro riêng của doanh "
        f"nghiệp, có thể giảm bằng đa dạng hoá danh mục."
    )
    msgs.append(
        f"CAPM đòi hỏi lợi suất {result.expected_return * 100:.2f}%/năm; lợi suất thực tế "
        f"tính theo cùng quy ước trung bình số học là {result.realized_arithmetic * 100:.2f}%/năm → "
        + ("vượt kỳ vọng."
           if result.realized_arithmetic > result.expected_return else "thấp hơn kỳ vọng.")
        + f" (Lợi suất kép thực nhận là {result.realized_return * 100:.2f}%/năm — thấp hơn trung "
          f"bình số học một khoảng xấp xỉ σ²/2 do tác động của biến động.)"
    )
    return msgs
