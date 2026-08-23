# Mô hình chi phí giao dịch trên sàn HOSE.

from dataclasses import dataclass

from src.config import FEE_RATE, SELL_TAX, SLIPPAGE


@dataclass(frozen=True)
class CostModel:
    # Chi phí tính theo tỷ lệ trên giá trị giao dịch.

    fee: float = FEE_RATE
    sell_tax: float = SELL_TAX
    slippage: float = SLIPPAGE

    @property
    def buy_rate(self) -> float:
        # Tổng chi phí cho một đồng giá trị mua vào.
        return self.fee + self.slippage

    @property
    def sell_rate(self) -> float:
        # Tổng chi phí cho một đồng giá trị bán ra.
        return self.fee + self.sell_tax + self.slippage

    @property
    def round_trip(self) -> float:
        # Chi phí một vòng mua rồi bán — ngưỡng lợi nhuận tối thiểu để hoà vốn.
        return self.buy_rate + self.sell_rate

    def cost_of(self, delta_weight: float) -> float:
        # Chi phí của một lần đổi vị thế, tính theo tỷ lệ trên tổng vốn.
        if delta_weight > 0:
            return delta_weight * self.buy_rate
        return -delta_weight * self.sell_rate

    def scaled(self, factor: float) -> "CostModel":
        # Bản sao với toàn bộ chi phí nhân lên `factor` — dùng cho kiểm định độ nhạy.
        return CostModel(
            fee=self.fee * factor,
            sell_tax=self.sell_tax * factor,
            slippage=self.slippage * factor,
        )


# Ba mức chi phí dùng trong thí nghiệm độ nhạy ở phần đánh giá.
NO_COST = CostModel(fee=0.0, sell_tax=0.0, slippage=0.0)
BASE_COST = CostModel()
