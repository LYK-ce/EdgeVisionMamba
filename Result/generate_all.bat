@echo off
REM Presented by KeJi
REM Date: 2026-01-19
REM Generate all column charts for VisionMamba optimization results (PDF format)

echo Generating all column charts (PDF format)...

REM Create output directory
if not exist "Img" mkdir Img

REM Models by parameter size
for %%m in (vim_5m vim_10m vim_15m vim_20m) do (
    echo Processing %%m...
    python draw_column.py --model %%m --all -o Img\%%m_all.pdf
    python draw_column.py --model %%m --all --speedup --baseline "Python Original" -o Img\%%m_speedup.pdf
)

REM Models by FLOPs
for %%m in (vim_2gflops vim_3gflops vim_4gflops vim_5gflops) do (
    echo Processing %%m...
    python draw_column.py --model %%m --all -o Img\%%m_all.pdf
    python draw_column.py --model %%m --all --speedup --baseline "Python Original" -o Img\%%m_speedup.pdf
)

echo Done! All charts generated in Img folder.
pause
