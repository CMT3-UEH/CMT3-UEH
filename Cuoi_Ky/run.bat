@echo off
REM ===================================================================
REM  Giao dich thuat toan tren co phieu FPT — chay toan bo du an
REM
REM    run.bat            chay nhung buoc con thieu roi mo ung dung
REM    run.bat tat-ca     chay lai TAT CA tu dau (xoa cache mo hinh)
REM    run.bat app        chi mo ung dung, khong chay gi them
REM    run.bat kiem-tra   chay lai phep tu kiem bo may backtest roi mo ung dung
REM ===================================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

REM ---------- Buoc 0: moi truong ----------
if not exist ".venv" (
    echo [0/7] Tao moi truong ao va cai thu vien ^(mat vai phut^)...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements-dev.txt
) else (
    call .venv\Scripts\activate.bat
)

if "%1"=="app" goto MO_UNG_DUNG

if "%1"=="tat-ca" (
    echo Xoa cache mo hinh de huan luyen lai tu dau...
    if exist "reports\_cache_deep" rmdir /s /q "reports\_cache_deep"
    if exist "reports\_cache_atfn" rmdir /s /q "reports\_cache_atfn"
    if exist "data\processed\panel_FPT.csv" del /q "data\processed\panel_FPT.csv"
)

REM ---------- Buoc 1: du lieu ----------
if not exist "data\processed\panel_FPT.csv" (
    echo.
    echo [1/7] Tai du lieu FPT, VNINDEX va nhan to vi mo ^(~1 phut, can mang^)...
    python fetch_data.py || goto LOI
)

REM ---------- Buoc 2: tu kiem bo may backtest ----------
REM Chi chay lai khi chua tung dat, hoac khi goi "run.bat kiem-tra".
REM Phep thu mat ~3 phut nen khong chay lai moi lan chi de mo ung dung.
if "%1"=="kiem-tra" if exist "reports\.backtest_dat" del /q "reports\.backtest_dat"
if not exist "reports\.backtest_dat" (
    echo.
    echo [2/7] Tu kiem bo may backtest — 6 phep thu ^(~3 phut^)...
    python -m src.experiments.check_engine || goto LOI_BACKTEST
    echo dat > "reports\.backtest_dat"
)

REM ---------- Buoc 3: bao cao du lieu va moc tham chieu ----------
if not exist "reports\03_moc_mua_va_giu.csv" (
    echo.
    echo [3/7] Bao cao du lieu va moc mua-va-nam-giu ^(~2 phut^)...
    python -m src.experiments.report_data || goto LOI
)

REM ---------- Buoc 4: baseline co dien va may hoc ----------
if not exist "reports\05_baseline_valid.csv" (
    echo.
    echo [4/7] Baseline co dien + may hoc, chon bang kiem dinh cuon chieu ^(~5 phut^)...
    python -m src.experiments.run_baselines || goto LOI
)

REM ---------- Buoc 5: baseline hoc sau ----------
if not exist "reports\08_hoc_sau_valid.csv" (
    echo.
    echo [5/7] Baseline hoc sau: MLP, CNN, RNN, LSTM, GRU, Transformer
    echo       6 kien truc x 5 hat giong — LAU NHAT ^(~25 phut^)
    echo       Ket qua luu sau TUNG mo hinh, dung giua chung chay lai khong mat.
    python -m src.experiments.run_deep || goto LOI
)

REM ---------- Buoc 6: mo hinh de xuat va ablation ----------
if not exist "reports\09_ablation_valid.csv" (
    echo.
    echo [6/7] Mo hinh de xuat ATFN, 5 cau hinh ablation x 5 hat giong ^(~50 phut^)
    echo       Ket qua luu sau TUNG cau hinh.
    python -m src.experiments.run_atfn || goto LOI
)

REM ---------- Buoc 7: danh gia tren tap kiem tra ----------
if not exist "reports\10_ket_qua_test.csv" (
    echo.
    echo [7/7] Danh gia cuoi tren tap kiem tra + kiem dinh thong ke ^(~3 phut^)...
    python -m src.experiments.run_final || goto LOI
)

:MO_UNG_DUNG
echo.
echo ===================================================================
echo  Dang khoi dong ung dung tai http://localhost:8501
echo  Nhan Ctrl+C trong cua so nay de dung.
echo ===================================================================
streamlit run app.py
goto :EOF

:LOI_BACKTEST
echo.
echo ===================================================================
echo  DUNG LAI: bo may backtest khong dat phep tu kiem.
echo  Moi ket qua sau do se vo nghia neu bo may sai, nen khong chay tiep.
echo ===================================================================
exit /b 1

:LOI
echo.
echo DUNG LAI: buoc tren bao loi. Xem thong bao ngay phia tren.
exit /b 1
