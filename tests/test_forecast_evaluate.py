import numpy as np
import pytest

from src.forecast.evaluate import mape, wmape, sign_f1


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),
        (np.array([1, 2, 3]), np.array([2, 2, 3]), (1/1 + 0 + 0)/3),
        (np.array([-1, -2]), np.array([-1, -1]), (0 + 1/2)/2),
        (np.array([0, 1]), np.array([1, 1]), (1/1e-8 + 0)/2),  # оставляем eps
        (np.array([1.5, 2.5]), np.array([1.0, 3.0]), (0.5/1.5 + 0.5/2.5)/2),
        (np.array([10]), np.array([15]), 0.5),
        (np.array([0, 0, 2]), np.array([1, 0, 3]), (1/1e-8 + 0 + 1/2)/3),
        (np.array([1e6, 2e6]), np.array([0.5e6, 2.5e6]), 0.375),
        (np.array([3, 0, -2, 4]), np.array([2, 0, -1, 5]), (1/3 + 0/1e-8 + 1/2 + 1/4)/4),
    ]
)
def test_mape(y_true, y_pred, expected):
    result = mape(y_true, y_pred)
    assert np.isclose(result, expected)

@pytest.mark.parametrize(
    "y_true, y_pred, weights, expected",
    [
        # идеальное совпадение
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 1, 1]), 0.0),

        # простая ошибка с равными весами
        (np.array([1, 2, 3]), np.array([2, 2, 3]), np.array([1, 1, 1]), (1*1 + 1*0 + 1*0)/(1*1 + 1*2 + 1*3)),  # 1/6 ≈ 0.1667

        # с разными весами
        (np.array([1, 2, 3]), np.array([2, 2, 3]), np.array([1, 2, 1]), (1*1 + 2*0 + 1*0)/(1*1 + 2*2 + 1*3)),  # 1/8 = 0.125

        # отрицательные значения
        (np.array([-1, -2]), np.array([-1, -1]), np.array([1, 1]), (0*1 + 1*1)/(1*1 + 1*2)),  # 1/3 ≈ 0.3333

        # нули в y_true
        (np.array([0, 1]), np.array([1, 1]), np.array([1, 1]), (1*1 + 1*0)/(1*1e-8 + 1*1)),  # ≈ 0.5

        # дробные числа
        (np.array([1.5, 2.5]), np.array([1.0, 3.0]), np.array([1, 1]), (1*0.5 + 1*0.5)/(1*1.5 + 1*2.5)),  # 1/4 = 0.25

        # массив из одного элемента
        (np.array([10]), np.array([15]), np.array([1]), 0.5),

        # все нули в y_true (тест eps)
        (np.array([0, 0]), np.array([1, 2]), np.array([1, 1]), (1*1 + 1*2)/(1*1e-8 + 1*1e-8)),  # огромная MAPE ~3e8
    ]
)
def test_wmape(y_true, y_pred, weights, expected):
    result = wmape(y_true, y_pred, weights)
    assert np.isclose(result, expected)

@pytest.mark.parametrize("y_true,y_pred,average,sample_weight,expected", [
    ([0.5, 0.9, -0.3], [0.1, -0.5, -0.8], 'macro', None, 2/3),
    # Мелкие дроби
    ([0.0001, 0.0002, -0.0003], [0.0004, -0.0005, -0.0006], 'macro', None, 2/3),
    # Большие дроби
    ([999.99, 888.88, -777.77], [111.11, -222.22, -333.33], 'macro', None, 2/3),
    # Очень маленькие
    ([1e-10, 2e-10, -3e-10], [4e-10, -5e-10, -6e-10], 'macro', None, 2/3),
    # Очень большие
    ([1e10, 2e10, -3e10], [1e10, -2e10, -3e10], 'macro', None, 2/3),
    # Разные значения, одинаковые знаки → 1.0
    ([0.001, 1000.0, -0.5], [0.9, 0.0001, -999.0], 'macro', None, 1.0),
    # Все разные, все wrong → 0.0
    ([0.1, 0.2, 0.3], [-1.1, -2.2, -3.3], 'macro', None, 0.0),
    # Perfect match
    ([0.1, -0.2, 0.3, -0.4], [9.1, -8.2, 7.3, -6.4], 'macro', None, 1.0),
    # 50/50
    ([-1.5, -2.5, 3.5, 4.5], [-1.5, 2.5, -3.5, 4.5], 'macro', None, 0.5),
    # Weights
    ([0.5, -0.5], [0.5, 0.5], 'macro', [1, 1], 1/3),
    ([0.5, -0.5], [0.5, 0.5], 'macro', [1, 0], 0.5),
    ([0.5, -0.5], [0.5, 0.5], 'macro', [0, 1], 0.0),
    # Binary
    ([-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], 'binary', None, 0.5),
    # Micro
    ([-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], 'micro', None, 0.5),
    # Numpy
    (np.array([0.5, -0.5, 0.9]), np.array([0.3, -0.7, 0.2]), 'macro', None, 1.0),
])
def test_sign_f1(y_true, y_pred, average, sample_weight, expected):
    result = sign_f1(y_true, y_pred, average=average, sample_weight=sample_weight)
    assert abs(result - expected) < 1e-9
