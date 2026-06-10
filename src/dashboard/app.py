from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.portfolio.optimizer import efficient_frontier
from src.portfolio.optimizer import optimize_latest_portfolio
from src.portfolio.optimizer import optimize_rebalancing_history
from src.utils import PROJECT_ROOT, load_config


st.set_page_config(page_title="AI 변동성 포트폴리오 대시보드", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(15, 23, 42, 0.28);
    }
    .dash-hero {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 24px 26px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
    }
    .dash-kicker {
        color: #38bdf8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .dash-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 8px;
    }
    .dash-subtitle {
        color: #cbd5e1;
        max-width: 980px;
        font-size: 1rem;
        line-height: 1.65;
    }
    .goal-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 6px 0 16px;
    }
    .goal-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 14px 14px 13px;
        background: rgba(15, 23, 42, 0.18);
        min-height: 118px;
    }
    .goal-index {
        color: #38bdf8;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 7px;
    }
    .goal-title { color: #f8fafc; font-weight: 760; margin-bottom: 6px; }
    .goal-text { color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; }
    .pipeline {
        border: 1px solid rgba(56, 189, 248, 0.26);
        border-radius: 8px;
        padding: 12px 14px;
        background: rgba(8, 47, 73, 0.22);
        color: #dbeafe;
        font-size: 0.94rem;
        margin-bottom: 8px;
    }
    .pipeline-note { color: #94a3b8; font-size: 0.86rem; margin-bottom: 18px; }
    @media (max-width: 900px) {
        .goal-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dash-title { font-size: 1.75rem; }
    }
    @media (max-width: 560px) { .goal-grid { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


config = load_config()
paths = config["paths"]


def read_table(path: str) -> pd.DataFrame:
    frame = pd.read_csv(PROJECT_ROOT / path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def source_label(source: str) -> str:
    return {
        "fred": "실제 FRED",
        "bls": "BLS",
        "federal_reserve": "Federal Reserve",
        "fred_mirror": "FRED mirror",
        "yahoo": "Yahoo Finance",
        "cache": "캐시",
        "sample": "샘플",
    }.get(source, source)


predictions_path = PROJECT_ROOT / paths["predictions"]
weights_path = PROJECT_ROOT / paths["portfolio_weights"]
metrics_path = PROJECT_ROOT / "data/predictions/model_metrics.csv"
macro_sources_path = PROJECT_ROOT / "data/raw/macro_sources.csv"
garch_midas_params_path = PROJECT_ROOT / "data/predictions/garch_midas_params.csv"

if not predictions_path.exists() or not weights_path.exists():
    st.warning("먼저 `python main.py`를 실행해 예측 및 포트폴리오 파일을 생성해 주세요.")
    st.stop()

predictions = read_table(paths["predictions"])
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
macro_sources = pd.read_csv(macro_sources_path) if macro_sources_path.exists() else pd.DataFrame()
garch_midas_params = pd.read_csv(garch_midas_params_path) if garch_midas_params_path.exists() else pd.DataFrame()

available_models = sorted(predictions["model"].unique()) if "model" in predictions.columns else ["RandomForest"]
default_model_index = available_models.index("GARCH-MIDAS") if "GARCH-MIDAS" in available_models else 0

st.sidebar.title("발표 네비게이션")
selected_page = st.sidebar.radio(
    "섹션",
    ["INTRO", "BACKGROUND", "DATAFLOW", "RESULT", "OUTRO"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("실험 설정")
selected_model = st.sidebar.selectbox("예측 모델", available_models, index=default_model_index)
scenario_options = {
    "기준 시나리오": {"return_shift": 0.0, "vol_multiplier": 1.0, "description": "모델 예측값을 그대로 사용합니다."},
    "금리·물가 상승": {"return_shift": -0.0010, "vol_multiplier": 1.12, "description": "긴축과 비용 부담으로 기대수익률을 낮추고 변동성을 높입니다."},
    "유가 급등": {"return_shift": -0.0006, "vol_multiplier": 1.10, "description": "에너지 가격 충격으로 전반적 위험을 확대합니다."},
    "달러 강세": {"return_shift": -0.0004, "vol_multiplier": 1.07, "description": "환율 부담과 글로벌 유동성 압력을 반영합니다."},
    "위험 완화": {"return_shift": 0.0005, "vol_multiplier": 0.92, "description": "거시 불확실성이 낮아지는 완화적 환경을 가정합니다."},
}
selected_scenario = st.sidebar.selectbox("거시경제 시나리오", list(scenario_options.keys()))
scenario = scenario_options[selected_scenario]

model_predictions = predictions[predictions["model"] == selected_model].copy() if "model" in predictions.columns else predictions.copy()
model_predictions["predicted_return"] = model_predictions["predicted_return"] + scenario["return_shift"]
model_predictions["predicted_volatility"] = (model_predictions["predicted_volatility"] * scenario["vol_multiplier"]).clip(lower=0.0001)
weights = optimize_latest_portfolio(model_predictions)

portfolio_expected_return = float((weights["expected_return"] * weights["weight"]).sum())
portfolio_expected_risk = float(((weights["predicted_volatility"] * weights["weight"]) ** 2).sum() ** 0.5)
portfolio_sharpe = portfolio_expected_return / portfolio_expected_risk if portfolio_expected_risk else 0.0


def render_intro() -> None:
    st.markdown(
        """
        <section class="dash-hero">
            <div class="dash-kicker">이스트캠프 AI퀀트 3기 6조</div>
            <div class="dash-title">AI 변동성 포트폴리오 대시보드</div>
            <div class="dash-subtitle">
                M7 일별 주가와 거시경제 지표를 결합해 장단기 변동성을 예측하고,
                Markowitz 평균-분산 최적화로 위험 대비 수익이 높은 포트폴리오 비중을 계산합니다.
            </div>
        </section>
        <div class="goal-grid">
            <div class="goal-card">
                <div class="goal-index">01</div>
                <div class="goal-title">혼합 주기 결합</div>
                <div class="goal-text">일별 M7 주가와 월별 거시지표를 영업일 기준 학습 데이터로 통합합니다.</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">02</div>
                <div class="goal-title">장단기 변동성 분해</div>
                <div class="goal-text">단기 주가 충격과 장기 거시 압력을 GARCH-MIDAS 구조로 분리합니다.</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">03</div>
                <div class="goal-title">포트폴리오 최적화</div>
                <div class="goal-text">예상 수익률과 예측 변동성으로 Markowitz 최적 비중을 계산합니다.</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">04</div>
                <div class="goal-title">자동화 대시보드</div>
                <div class="goal-text">수집, 전처리, 예측, 리밸런싱, 표출 과정을 Streamlit으로 통합합니다.</div>
            </div>
        </div>
        <div class="pipeline">
            데이터 수집 → 전처리/주기 정렬 → 예측 모델 → Markowitz 최적화 → 대시보드 표출
        </div>
        <div class="pipeline-note">
            왼쪽 탭을 따라가면 문제 배경, 데이터 흐름, 결과, 한계와 확장 방향 순서로 발표할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("발표 핵심 메시지")
    st.write(
        "이 프로젝트는 서로 주기가 다른 금융·거시 데이터를 하나의 일별 학습 테이블로 정렬하고, "
        "예측된 수익률과 변동성을 포트폴리오 의사결정으로 연결하는 자동화 파이프라인입니다."
    )


def render_background() -> None:
    st.title("BACKGROUND")
    st.subheader("왜 혼합 빈도 데이터인가?")
    st.write(
        "주가는 거래일마다 변하지만, 기준금리·CPI·M2·PPI·산업생산 같은 거시지표는 월별로 발표됩니다. "
        "이 프로젝트는 일별 주가와 저빈도 거시지표를 정보 손실 없이 결합해 장기 위험 요인을 모델에 반영합니다."
    )
    st.subheader("왜 GARCH-MIDAS인가?")
    st.write(
        "GARCH-MIDAS는 변동성을 단기 주가 충격과 장기 거시경제 압력으로 분해합니다. "
        "대시보드의 GARCH-MIDAS 선택 시 단기 변동성, 장기 거시 압력, 최종 예측 변동성을 따로 확인할 수 있습니다."
    )
    st.subheader("비교 모델")
    st.write(
        "Ridge, RandomForest, ExtraTrees, GradientBoosting, SVR, KNN, GARCH-MIDAS를 비교합니다. "
        "MSE, MAE, QLIKE로 수익률 및 변동성 예측 오차를 평가합니다."
    )
    st.info(
        f"현재 선택 모델은 `{selected_model}`, 선택 시나리오는 `{selected_scenario}`입니다. "
        f"{scenario['description']}"
    )


def render_dataflow() -> None:
    st.title("DATAFLOW")
    st.subheader("수집 및 정렬 구조")
    st.markdown(
        """
        - **주가 데이터:** yfinance 기반 M7 일별 OHLCV 및 로그수익률
        - **거시 데이터:** FRED, BLS, Federal Reserve, Yahoo Finance 기반 지표
        - **전처리:** 월별 지표를 영업일 기준으로 확장하고 forward-fill 및 보간
        - **최종 데이터:** 일별 주가와 거시지표가 결합된 학습 테이블
        """
    )
    if not macro_sources.empty:
        source_counts = macro_sources["source"].map(source_label).value_counts().to_dict()
        st.caption("데이터 출처 요약: " + ", ".join([f"{source} {count}개" for source, count in source_counts.items()]))
        source_table = macro_sources.rename(
            columns={"indicator": "지표", "fred_code": "FRED 코드", "source": "데이터 출처", "rows": "행 수"}
        )
        source_table["데이터 출처"] = source_table["데이터 출처"].map(source_label)
        st.dataframe(source_table, use_container_width=True, hide_index=True)

    if not garch_midas_params.empty:
        st.subheader("GARCH-MIDAS 선택 파라미터")
        params_table = garch_midas_params.rename(
            columns={
                "ticker": "종목",
                "omega": "omega",
                "alpha": "alpha",
                "beta": "beta",
                "macro_weight": "거시 가중치",
                "lookback": "MIDAS 기간",
                "return_window": "수익률 평균 기간",
                "volatility_scale": "변동성 스케일",
                "validation_volatility_mae": "검증 변동성 MAE",
            }
        )
        st.dataframe(params_table, use_container_width=True, hide_index=True)
        st.caption(
            "각 종목의 학습 구간 일부를 검증 구간으로 떼어 두고, 변동성 MAE가 가장 낮은 파라미터 조합을 선택했습니다."
        )


def render_result() -> None:
    st.title("RESULT")
    st.caption(
        f"`{selected_model}` 모델과 `{selected_scenario}`가 적용된 결과입니다. "
        f"수익률 조정 {scenario['return_shift'] * 100:.2f}%p, 변동성 배율 {scenario['vol_multiplier']:.2f}배가 반영됩니다."
    )

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("포트폴리오 예상 수익률", f"{portfolio_expected_return * 100:.2f}%")
    metric_middle.metric("포트폴리오 예상 위험", f"{portfolio_expected_risk * 100:.2f}%")
    metric_right.metric("위험 대비 수익률", f"{portfolio_sharpe:.2f}")

    top_weight = weights.loc[weights["weight"].idxmax()]
    bottom_weight = weights.loc[weights["weight"].idxmin()]
    top_return = weights.loc[weights["expected_return"].idxmax()]
    top_risk = weights.loc[weights["predicted_volatility"].idxmax()]

    with st.container(border=True):
        st.markdown("**포트폴리오 자동 해석**")
        st.markdown(
            f"`{selected_model}` 기준 최적 포트폴리오는 `{top_weight['ticker']}` 비중이 가장 높습니다 "
            f"({top_weight['weight'] * 100:.1f}%). 이는 예측 수익률과 예측 변동성을 함께 고려했을 때 "
            "위험 대비 기여도가 가장 높게 평가되었기 때문입니다."
        )
        st.markdown(
            f"예상 수익률이 가장 높은 종목은 `{top_return['ticker']}`({top_return['expected_return'] * 100:.2f}%)이고, "
            f"예측 변동성이 가장 높은 종목은 `{top_risk['ticker']}`({top_risk['predicted_volatility'] * 100:.2f}%)입니다. "
            f"`{bottom_weight['ticker']}`는 가장 낮은 비중({bottom_weight['weight'] * 100:.1f}%)을 받았습니다."
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("포트폴리오 투자 비중")
        fig = px.pie(weights, names="ticker", values="weight", hole=0.45, labels={"ticker": "종목", "weight": "투자 비중"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("비중이 클수록 선택한 모델 기준으로 위험 대비 기대수익이 높게 평가된 자산입니다.")

    with right:
        st.subheader("최근 예측 결과")
        forecast_table = weights.assign(weight_pct=lambda frame: frame["weight"] * 100)[
            ["ticker", "expected_return", "predicted_volatility", "weight_pct"]
        ].rename(
            columns={
                "ticker": "종목",
                "expected_return": "예상 수익률",
                "predicted_volatility": "예측 변동성",
                "weight_pct": "투자 비중(%)",
            }
        )
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.subheader("예측 추세")
    selected_ticker = st.selectbox("종목", sorted(model_predictions["ticker"].unique()))
    ticker_predictions = model_predictions[model_predictions["ticker"] == selected_ticker]
    trend = px.line(
        ticker_predictions,
        x="date",
        y=["target_next_return", "predicted_return"],
        labels={"date": "날짜", "value": "수익률", "variable": "구분"},
    )
    trend.for_each_trace(
        lambda trace: trace.update(
            name={"target_next_return": "실제 다음날 수익률", "predicted_return": "예측 다음날 수익률"}.get(trace.name, trace.name)
        )
    )
    st.plotly_chart(trend, use_container_width=True)

    if selected_model == "GARCH-MIDAS" and {"short_run_volatility", "long_run_macro_component"}.issubset(ticker_predictions.columns):
        st.subheader("장단기 변동성 분해")
        decomposition = ticker_predictions[
            ["date", "short_run_volatility", "long_run_macro_component", "predicted_volatility"]
        ].rename(
            columns={
                "short_run_volatility": "단기 변동성",
                "long_run_macro_component": "장기 거시 압력",
                "predicted_volatility": "최종 예측 변동성",
            }
        )
        decomposition_fig = px.line(
            decomposition,
            x="date",
            y=["단기 변동성", "장기 거시 압력", "최종 예측 변동성"],
            labels={"date": "날짜", "value": "값", "variable": "구분"},
        )
        st.plotly_chart(decomposition_fig, use_container_width=True)
        st.caption(
            "장기 거시 압력은 변동성에 곱해지는 위험 배율입니다. 1.05는 거시 환경이 단기 변동성을 약 5% 확대한다는 의미입니다."
        )

    frontier = efficient_frontier(model_predictions)
    if not frontier.empty:
        st.subheader("효율적 투자선")
        frontier_fig = px.line(frontier, x="risk", y="target_return", labels={"risk": "위험", "target_return": "목표 수익률"})
        st.plotly_chart(frontier_fig, use_container_width=True)
        st.caption("최소분산 포트폴리오 이후, 위험을 더 부담할 때 기대수익률이 높아지는 위쪽 효율 경계입니다.")

    rebalancing_history = optimize_rebalancing_history(model_predictions, days=60)
    if not rebalancing_history.empty:
        st.subheader("리밸런싱 비중 변화")
        rebalancing_mode = st.radio("리밸런싱 그래프 보기 방식", ["전체 보기", "선택 종목만 보기"], horizontal=True)
        if rebalancing_mode == "선택 종목만 보기":
            focused_ticker = st.selectbox("강조할 종목", sorted(rebalancing_history["ticker"].unique()))
            rebalancing_plot_data = rebalancing_history[rebalancing_history["ticker"] == focused_ticker]
        else:
            rebalancing_plot_data = rebalancing_history
        rebalancing_fig = px.line(
            rebalancing_plot_data,
            x="date",
            y="weight",
            color="ticker",
            labels={"date": "날짜", "weight": "투자 비중", "ticker": "종목"},
        )
        rebalancing_fig.update_layout(hovermode="closest")
        rebalancing_fig.update_traces(hovertemplate="종목=%{fullData.name}<br>날짜=%{x}<br>투자 비중=%{y:.2%}<extra></extra>")
        st.plotly_chart(rebalancing_fig, use_container_width=True)

    if not metrics.empty:
        st.subheader("모델 성능 요약")
        model_summary = (
            metrics.groupby("model", as_index=False)[
                ["return_mse", "return_mae", "volatility_mse", "volatility_mae", "volatility_qlike"]
            ]
            .mean()
            .sort_values("model")
        )
        model_summary["return_score"] = model_summary[["return_mse", "return_mae"]].mean(axis=1)
        model_summary["volatility_score"] = model_summary[["volatility_mse", "volatility_mae"]].mean(axis=1)
        model_summary["total_score"] = model_summary[["return_score", "volatility_score"]].mean(axis=1)
        best_return_model = model_summary.loc[model_summary["return_score"].idxmin()]
        best_volatility_model = model_summary.loc[model_summary["volatility_score"].idxmin()]
        best_total_model = model_summary.loc[model_summary["total_score"].idxmin()]
        summary_left, summary_middle, summary_right = st.columns(3)
        summary_left.metric("수익률 예측 최우수", best_return_model["model"])
        summary_middle.metric("변동성 예측 최우수", best_volatility_model["model"])
        summary_right.metric("종합 최우수", best_total_model["model"])
        st.dataframe(
            model_summary.rename(
                columns={
                    "model": "모델",
                    "return_mse": "평균 수익률 MSE",
                    "return_mae": "평균 수익률 MAE",
                    "volatility_mse": "평균 변동성 MSE",
                    "volatility_mae": "평균 변동성 MAE",
                    "volatility_qlike": "평균 변동성 QLIKE",
                    "return_score": "수익률 종합 오차",
                    "volatility_score": "변동성 종합 오차",
                    "total_score": "전체 종합 오차",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("모델 성능 지표")
        metrics_table = metrics.rename(
            columns={
                "model": "모델",
                "ticker": "종목",
                "return_mse": "수익률 MSE",
                "return_mae": "수익률 MAE",
                "volatility_mse": "변동성 MSE",
                "volatility_mae": "변동성 MAE",
                "volatility_qlike": "변동성 QLIKE",
            }
        )
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        st.caption("MSE, MAE, QLIKE는 모두 낮을수록 좋습니다.")


def render_outro() -> None:
    st.title("OUTRO")
    st.subheader("기획서 목표 대비 달성 내용")
    st.markdown(
        """
        - 고빈도 일별 주가와 저빈도 거시지표 결합 파이프라인 구축
        - GARCH-MIDAS 기반 장단기 변동성 분해 및 시각화
        - Markowitz 평균-분산 최적화와 효율적 투자선 구현
        - 모델 비교, 거시 시나리오, 리밸런싱 비중 변화까지 Streamlit으로 통합
        """
    )
    st.subheader("남은 확장 방향")
    st.markdown(
        """
        - 실시간 스케줄러 기반 자동 갱신
        - MySQL 또는 Parquet 기반 운영 저장소
        - LSTM/GRU/XGBoost 등 추가 모델 고도화
        - 시나리오별 종목 민감도와 백테스트 성과 지표 추가
        """
    )
    st.success("현재 버전은 발표용으로 전체 파이프라인 흐름과 핵심 결과를 설명할 수 있는 상태입니다.")


if selected_page == "INTRO":
    render_intro()
elif selected_page == "BACKGROUND":
    render_background()
elif selected_page == "DATAFLOW":
    render_dataflow()
elif selected_page == "RESULT":
    render_result()
else:
    render_outro()
