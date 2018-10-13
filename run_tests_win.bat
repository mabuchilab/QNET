set doclist=
for /f "tokens=*" %%F in ('dir /b /s docs\*.rst') do call set doclist=%%doclist%% "%%~F"
pytest --doctest-modules --cov=qnet src tests %doclist%
