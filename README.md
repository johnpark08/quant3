# AI퀀트 3기 Volatility Portfolio 

 DAS, DSS, JPS, WSS 파트를 순서대로 진행할 수 있도록 만든 최소 실행형 프로젝트입니다.

## 실행 순서

1. 의존성 설치

```powershell
pip install -r requirements.txt
```

2. 데이터 수집부터 포트폴리오 계산까지 실행

```powershell
python main.py
```

3. 대시보드 실행

```powershell
streamlit run src/dashboard/app.py
```

## 산출물

- `data/raw/raw_prices.csv`: yfinance 기반 일별 주가와 로그 수익률
- `data/raw/raw_macro.csv`: FRED 기반 거시경제 원천 데이터
- `data/raw/macro_sources.csv`: 거시지표별 실제 FRED/캐시/샘플 데이터 출처
- `data/processed/training_dataset.csv`: 모델 학습용 공통 테이블
- `data/predictions/predictions.csv`: 예측 결과
- `data/predictions/model_metrics.csv`: 모델별 성능 지표
- `data/predictions/portfolio_weights.csv`: 최적 포트폴리오 비중

## 현재 MVP 범위

- 자산: AAPL, TSLA, NVDA
- 거시지표: 기준금리, CPI, M2, WTI 유가, USD/KRW
- 모델: Ridge Regression, RandomForest, ExtraTrees, GradientBoosting, SVR, KNN, 종목별 튜닝 GARCH-MIDAS
- 최적화: Markowitz 평균-분산 방식의 최대 Sharpe 비중
- 평가 지표: MSE, MAE, QLIKE
- 대시보드: Streamlit + Plotly, 모델 선택 및 거시경제 시나리오 선택

네트워크나 API 문제가 있으면 지표별로 `실제 FRED -> 캐시 -> 샘플` 순서로 자동 전환되어 전체 파이프라인 구조를 검증할 수 있습니다. 대시보드 하단의 `거시지표 수집 상태`에서 각 지표의 데이터 출처를 확인할 수 있습니다.
