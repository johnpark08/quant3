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
    .block-container { padding-top: 3rem; padding-bottom: 2.4rem; }
    div[data-testid="stButton"] > button {
        border: 1px solid rgba(148, 163, 184, 0.34);
        border-radius: 999px;
        min-height: 42px;
        background: rgba(15, 23, 42, 0.46);
        color: #cbd5e1;
        font-weight: 760;
        letter-spacing: 0;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: rgba(56, 189, 248, 0.72);
        color: #f8fafc;
        background: rgba(8, 47, 73, 0.44);
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        border-color: #38bdf8;
        background: #38bdf8;
        color: #020617;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        border-color: #7dd3fc;
        background: #7dd3fc;
        color: #020617;
    }
    .section-nav-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0 0 8px;
    }
    .section-nav-bottom { margin-bottom: 18px; }
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
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px;
    }
    .flow-step {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 15px 15px 14px;
        background: rgba(15, 23, 42, 0.22);
        min-height: 148px;
        position: relative;
    }
    .flow-step::after {
        content: ">";
        position: absolute;
        right: -11px;
        top: 50%;
        transform: translateY(-50%);
        color: #38bdf8;
        font-weight: 900;
    }
    .flow-step:last-child::after { content: ""; }
    .flow-tag {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .flow-title {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 780;
        margin-bottom: 8px;
    }
    .flow-text {
        color: #cbd5e1;
        font-size: 0.86rem;
        line-height: 1.55;
    }
    .logic-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px;
    }
    .logic-card {
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 15px;
        background: rgba(8, 47, 73, 0.2);
        min-height: 128px;
    }
    .logic-title {
        color: #f8fafc;
        font-weight: 780;
        margin-bottom: 7px;
    }
    .logic-text {
        color: #cbd5e1;
        font-size: 0.86rem;
        line-height: 1.55;
    }
    @media (max-width: 900px) {
        .goal-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .flow-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .logic-grid { grid-template-columns: 1fr; }
        .flow-step::after { content: ""; }
        .dash-title { font-size: 1.75rem; }
    }
    @media (max-width: 560px) {
        .goal-grid { grid-template-columns: 1fr; }
        .flow-grid { grid-template-columns: 1fr; }
    }
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

section_pages = ["인트로", "배경", "데이터 흐름", "결과", "마무리"]
if "selected_page" not in st.session_state:
    st.session_state.selected_page = section_pages[0]
if st.session_state.selected_page not in section_pages:
    st.session_state.selected_page = section_pages[0]

st.markdown('<div class="section-nav-label">SECTION</div>', unsafe_allow_html=True)
nav_columns = st.columns(len(section_pages), gap="small")
for page, column in zip(section_pages, nav_columns):
    with column:
        if st.button(
            page,
            key=f"section_nav_{page}",
            type="primary" if st.session_state.selected_page == page else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_page = page
            st.rerun()
st.markdown('<div class="section-nav-bottom"></div>', unsafe_allow_html=True)
selected_page = st.session_state.selected_page

st.sidebar.subheader("실험 설정")
selected_model = st.sidebar.selectbox("예측 모델", available_models, index=default_model_index)
scenario_options = {
    "기준 시나리오": {"return_shift": 0.0, "vol_multiplier": 1.0, "description": "모델 예측값 원안 사용"},
    "금리·물가 상승": {"return_shift": -0.0010, "vol_multiplier": 1.12, "description": "긴축과 비용 부담에 따른 수익률 하향, 변동성 상향"},
    "유가 급등": {"return_shift": -0.0006, "vol_multiplier": 1.10, "description": "에너지 가격 충격에 따른 전반적 위험 확대"},
    "달러 강세": {"return_shift": -0.0004, "vol_multiplier": 1.07, "description": "환율 부담과 글로벌 유동성 압력 반영"},
    "위험 완화": {"return_shift": 0.0005, "vol_multiplier": 0.92, "description": "거시 불확실성 완화 환경 가정"},
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
            <div class="dash-kicker">6조 / 금정호(조장), 정총균, 박성준, 박진철</div>
            <div class="dash-title">혼합 빈도 거시경제 지표 기반 테크 대장주 변동성 예측 및 포트폴리오 최적화</div>
            <div class="dash-subtitle">
                GARCH-MIDAS로 일별 노이즈와 거시경제가 만드는 장기 위험 분리
            </div>
        </section>
        <div class="goal-grid">
            <div class="goal-card">
                <div class="goal-index">01</div>
                <div class="goal-title">혼합 주기 결합</div>
                <div class="goal-text">일별 M7 주가와 월별 거시지표를 영업일 기준 학습 데이터로 통합</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">02</div>
                <div class="goal-title">장단기 변동성 분해</div>
                <div class="goal-text">단기 주가 충격과 장기 거시 압력을 GARCH-MIDAS 구조로 분리</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">03</div>
                <div class="goal-title">포트폴리오 최적화</div>
                <div class="goal-text">예상 수익률과 예측 변동성으로 Markowitz 최적 비중 산출</div>
            </div>
            <div class="goal-card">
                <div class="goal-index">04</div>
                <div class="goal-title">자동화 대시보드</div>
                <div class="goal-text">수집, 전처리, 예측, 리밸런싱, 표출 과정을 Streamlit으로 통합</div>
            </div>
        </div>
        <div class="pipeline">
            데이터 수집 → 전처리/주기 정렬 → 예측 모델 → Markowitz 최적화 → 대시보드 표출
        </div>
        <div class="pipeline-note">
            상단 섹션 버튼 기준으로 문제 배경, 데이터 흐름, 결과, 한계와 확장 방향 순서 발표
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_background() -> None:
    st.title("배경")
    st.subheader("참고 논문")
    st.markdown(
        """
        이영임(2017), 「거시경제 변수를 이용한 미국 주식시장 변동성 예측」을 주요 배경 연구로 참고.

        - Engle et al.(2013)의 GARCH-MIDAS 접근으로 미국 주식시장 변동성 예측.
        - 변동성을 단기 시장 충격 성분과 거시경제 기반 장기 지속 성분으로 분해.
        - 산업생산, 물가, 국제유가, 환율 등 거시 변수가 장기 변동성 예측에 유의미한 정보 제공.
        - 거시 변수를 포함한 GARCH-MIDAS가 단순 GARCH(1,1) 대비 예측 성능 개선.
        - MSE, MAE, QLIKE로 평가. QLIKE는 변동성 과소예측 위험에 민감한 지표.
        """
    )
    st.subheader("프로젝트 반영점")
    st.markdown(
        """
        - 일별 M7 주가 데이터와 월별·일별 거시경제 지표를 하나의 학습 테이블로 결합.
        - GARCH-MIDAS 구조로 변동성을 단기 변동성과 장기 거시 압력으로 분해.
        - 기준금리, CPI, M2, PPI, 산업생산, WTI 유가, 환율, 달러지수, VIX를 거시 변수 후보로 확장.
        - GARCH-MIDAS와 Ridge, RandomForest, ExtraTrees, GradientBoosting, SVR, KNN 성능 비교.
        """
    )
    st.subheader("왜 혼합 빈도 데이터인가?")
    st.write(
        "주가는 거래일마다 변동. 기준금리·CPI·M2·PPI·산업생산 등 거시지표는 주로 월별 발표. "
        "일별 주가와 저빈도 거시지표를 결합해 장기 위험 요인을 모델에 반영."
    )
    st.subheader("왜 GARCH-MIDAS인가?")
    st.write(
        "GARCH-MIDAS는 변동성을 단기 주가 충격과 장기 거시경제 압력으로 분해. "
        "GARCH-MIDAS 선택 시 단기 변동성, 장기 거시 압력, 최종 예측 변동성 분리 확인."
    )
    st.subheader("모델 선택 근거")
    st.markdown(
        """
        - 분석 목표는 단순 주가 예측이 아니라 단기 노이즈와 장기 거시 위험의 분리.
        - GARCH(1,1)는 과거 변동성 패턴 중심. GARCH-MIDAS는 거시경제 변수를 장기 위험 추세에 연결.
        - 유가, 물가, 산업생산 등 거시 변수를 결합하면 변동성 예측 오차가 낮아진다는 설명 근거 확보.
        """
    )
    st.subheader("비교 모델")
    st.write(
        "Ridge, RandomForest, ExtraTrees, GradientBoosting, SVR, KNN, GARCH-MIDAS 비교. "
        "MSE, MAE, QLIKE 기준 수익률 및 변동성 예측 오차 평가."
    )
    st.info(
        f"현재 선택 모델: `{selected_model}` / 선택 시나리오: `{selected_scenario}`. "
        f"{scenario['description']}"
    )


def render_dataflow() -> None:
    st.title("데이터 흐름")
    st.markdown(
        """
        <div class="flow-grid">
            <div class="flow-step">
                <div class="flow-tag">STEP 01</div>
                <div class="flow-title">주가 수집</div>
                <div class="flow-text">yfinance 기반 M7 일별 OHLCV와 로그수익률 생성</div>
            </div>
            <div class="flow-step">
                <div class="flow-tag">STEP 02</div>
                <div class="flow-title">거시지표 수집</div>
                <div class="flow-text">금리, CPI, M2, PPI, 산업생산, 유가, 환율, VIX 수집</div>
            </div>
            <div class="flow-step">
                <div class="flow-tag">STEP 03</div>
                <div class="flow-title">혼합 빈도 정렬</div>
                <div class="flow-text">월별 지표를 영업일 기준으로 확장하고 결측 구간 보정</div>
            </div>
            <div class="flow-step">
                <div class="flow-tag">STEP 04</div>
                <div class="flow-title">학습 테이블 생성</div>
                <div class="flow-text">일별 주가와 거시 변수를 결합해 모델 입력 데이터 구성</div>
            </div>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )
    st.subheader("전처리 정당성")
    st.markdown(
        """
        <div class="logic-grid">
            <div class="logic-card">
                <div class="logic-title">원자료 한계</div>
                <div class="logic-text">거시지표를 그대로 투입하면 발표 주기 차이와 노이즈가 예측 성능을 약화할 수 있음.</div>
            </div>
            <div class="logic-card">
                <div class="logic-title">변동성 필드</div>
                <div class="logic-text">산업생산처럼 방향성보다 지표 자체의 불확실성이 장기 위험 설명에 유용한 경우 존재.</div>
            </div>
            <div class="logic-card">
                <div class="logic-title">모델 입력 정제</div>
                <div class="logic-text">Schwert(1989) 방식의 거시 변수 변동성 가공으로 장기 위험 신호 강화.</div>
            </div>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )
    if not macro_sources.empty:
        source_counts = macro_sources["source"].map(source_label).value_counts().to_dict()
        st.subheader("거시지표 수집 상태")
        source_count_frame = pd.DataFrame(
            [{"데이터 출처": source, "지표 수": count} for source, count in source_counts.items()]
        )
        source_fig = px.bar(
            source_count_frame,
            x="데이터 출처",
            y="지표 수",
            text="지표 수",
            color="데이터 출처",
        )
        source_fig.update_traces(textposition="outside")
        source_fig.update_layout(showlegend=False, yaxis_title="지표 수", xaxis_title=None, margin=dict(t=20, b=10))
        st.plotly_chart(source_fig, use_container_width=True)

        source_table = macro_sources.rename(
            columns={"indicator": "지표", "fred_code": "FRED 코드", "source": "데이터 출처", "rows": "행 수"}
        )
        source_table["데이터 출처"] = source_table["데이터 출처"].map(source_label)
        with st.expander("상세 수집 테이블 보기", expanded=False):
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
            "각 종목의 학습 구간 일부를 검증 구간으로 분리. 변동성 MAE가 가장 낮은 파라미터 조합 선택."
        )


def render_result() -> None:
    st.title("결과")
    st.caption(
        f"`{selected_model}` 모델 / `{selected_scenario}` 시나리오 적용 결과. "
        f"수익률 조정 {scenario['return_shift'] * 100:.2f}%p, 변동성 배율 {scenario['vol_multiplier']:.2f}배 반영."
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
        st.markdown("포트폴리오 자동 해석")
        st.markdown(
            f"`{selected_model}` 기준 최적 포트폴리오는 `{top_weight['ticker']}` 비중이 가장 높음 "
            f"({top_weight['weight'] * 100:.1f}%). 이는 예측 수익률과 예측 변동성을 함께 고려했을 때 "
            "위험 대비 기여도가 가장 높게 평가된 결과."
        )
        st.markdown(
            f"예상 수익률이 가장 높은 종목은 `{top_return['ticker']}`({top_return['expected_return'] * 100:.2f}%)이고, "
            f"예측 변동성이 가장 높은 종목은 `{top_risk['ticker']}`({top_risk['predicted_volatility'] * 100:.2f}%). "
            f"`{bottom_weight['ticker']}`는 최저 비중({bottom_weight['weight'] * 100:.1f}%)."
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("포트폴리오 투자 비중")
        fig = px.pie(weights, names="ticker", values="weight", hole=0.45, labels={"ticker": "종목", "weight": "투자 비중"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("비중이 클수록 선택 모델 기준 위험 대비 기대수익이 높게 평가된 자산.")

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
            "장기 거시 압력은 변동성에 곱해지는 위험 배율. 1.05는 거시 환경이 단기 변동성을 약 5% 확대한다는 의미."
        )
        st.info(
            "해석: 단기 변동성은 기업 뉴스와 투자심리의 빠른 흔들림. "
            "장기 거시 압력은 유가, 물가, 산업생산 등 거시 환경이 만드는 완만한 위험 추세. "
            "최종 예측 변동성은 두 요인이 결합된 포트폴리오 위험 입력값."
        )

    frontier = efficient_frontier(model_predictions)
    if not frontier.empty:
        st.subheader("효율적 투자선")
        frontier_fig = px.line(frontier, x="risk", y="target_return", labels={"risk": "위험", "target_return": "목표 수익률"})
        st.plotly_chart(frontier_fig, use_container_width=True)
        st.caption("최소분산 포트폴리오 이후, 위험을 더 부담할 때 기대수익률이 높아지는 위쪽 효율 경계.")

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
        st.info(
            "QLIKE는 변동성 예측 전용 손실 지표. 실제 변동성이 큰데 모델이 위험을 낮게 예측하는 과소예측 상황에 더 민감. "
            "포트폴리오 관점에서는 손실 가능성을 작게 보는 모델을 걸러내는 보조 기준."
        )


def render_outro() -> None:
    st.title("마무리")
    st.subheader("기획서 목표 대비 달성 내용")
    st.markdown(
        """
        - 고빈도 일별 주가와 저빈도 거시지표 결합 파이프라인 구축
        - GARCH-MIDAS 기반 장단기 변동성 분해 및 시각화
        - Markowitz 평균-분산 최적화와 효율적 투자선 구현
        - 모델 비교, 거시 시나리오, 리밸런싱 비중 변화까지 Streamlit으로 통합
        """
    )
    st.subheader("최종 결론")
    st.write(
        "거시경제 변수 결합과 장단기 변동성 분해를 통해 단순 예측이 아닌 위험 기반 포트폴리오 의사결정 구조 구현."
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


if selected_page == "인트로":
    render_intro()
elif selected_page == "배경":
    render_background()
elif selected_page == "데이터 흐름":
    render_dataflow()
elif selected_page == "결과":
    render_result()
else:
    render_outro()
