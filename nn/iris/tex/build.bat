@ECHO OFF
start /wait ./render.diagrams.py
echo diagram stage
IF NOT %ERRORLEVEL% == 0 GOTO ERROR

start /wait pdflatex -enable-installer -output-directory=..\ ./notes.tex
echo pdflatex stage
start /wait bibtex -enable-installer ..\notes.aux
echo bibtex stage
start /wait pdflatex -output-directory=..\ ./notes.tex
echo pdflatex stage 2
start /wait pdflatex -output-directory=..\ ./notes.tex
echo padflatex stage 3
IF NOT %ERRORLEVEL% == 0 GOTO ERROR

:CLEANUP
del .\iris.nn.png
del .\iris.nn.w1.png
del .\iris.nn.w2.png
del ..\notes.aux
del ..\notes.log
del ..\notes.out
del ..\notes.bbl
del ..\notes.blg

goto END

:ERROR
ECHO An error occured during the script execution. 
ECHO Errorlevel is %ERRORLEVEL%.
GOTO QUIT

:END
ECHO Successful!
ECHO Look for notes.pdf one folder above. 

:QUIT