Dim str
str = ".\popup.txt"

Dim fso
Set fso = CreateObject("Scripting.FileSystemObject")

Dim stream
Set stream = fso.OpenTextFile(str)

Dim txt
txt = ""
do Until stream.AtEndOfStream=True
   Dim Line
       Line = stream.ReadLine
   txt = txt+Line+vbCrLf
Loop

MsgBox(txt)