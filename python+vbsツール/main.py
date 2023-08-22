import subprocess
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

def btn_click():
    subprocess.Popen(r'wscript "Yomitori.vbs" ')
    
def btn_click2():
    import gurafu
    

root=tk.Tk()
root.title("ツール選択")
root.geometry("600x600")
root.configure(bg="blue")

frame=ttk.Frame(root)
frame.pack(fill = tk.BOTH, padx=20, pady=40)

frame2=ttk.Frame(root)
frame2.pack(fill = tk.BOTH, padx=20, pady=200)


button=tk.Button(frame, text="生成", command=btn_click)

button2=tk.Button(frame2, text="グラフ", command=btn_click2)

button.pack()

button2.pack()

lbl = tk.Label(text='検索対象')
lbl.place(x=70, y=50)

txt = tk.Entry(width=20)
txt.place(x=140, y=50)

def btn_click3():
    tt1=txt.get()
    messagebox.showinfo("結果", tt1)

button3=tk.Button(frame, text="ポップアップ", command=btn_click3)

button3.pack()

root.mainloop()