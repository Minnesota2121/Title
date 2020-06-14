# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 00:23:00 2019

@author: abcde
"""

import tkinter as tk  # 使用Tkinter前需要先导入

import tkinter.messagebox  # 要使用messagebox先要导入模块
 
#实例化object，建立窗口window
window = tk.Tk()
 
#可视化窗口名字
window.title('文章标题生成原型系统')
 
#设定窗口的大小(长 * 宽)
window.geometry('600x400')  # 这里的乘是小x
 
#在图形界面上设定标签
l1 = tk.Label(window, text='文章标题生成原型系统',fg='white', bg='gray', font=('Arial', 15), width=800, height=3)
# 说明： bg为背景，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
 
#放置标签
l1.pack()    # Label内容content区域放置位置，自动调节尺寸
# 放置lable的方法有：1）l.pack(); 2)l.place();
 
# 第6步，主窗口循环显示

l2 = tk.Label(window, text='请输入待生成标题的文章：', font=('StSong', 12), width=800, height=2)
# 说明： bg为背景，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高

# 第5步，放置标签
l2.pack()    # Label内容content区域放置位置，自动调节尺寸


#t1 = tk.Text(window, width=800,height=6)
#t1.pack()

#创建滚动条
#user_text=tk.Entry(window, width=800) #创建文本框
#user_text.pack()
#
#scrollbar = tk.Scrollbar(window, command=user_text.xview)
##scrollbar.pack()
#user_text.configure(xscrollcommand=scrollbar.set)


e2 = tk.Text(window,
            width=800,
            height=4,
            font=('StSong', 12),
            foreground='black')
e2.pack()

l4 = tk.Label(window, text='点击按钮生成标题：', font=('StSong', 12), width=800, height=1)
l4.pack()

from newtest import gen_title  # 先导入标题生成模型

def new_title():
    
    user=e2.get(1.0, 'end')
    print(user)
    
    if user == '\n':
        tkinter.messagebox.showerror(title='输入错误', message='输入为空！') 
    else:
        user = user.replace('\n','')
        title_result = gen_title(user)  #调用标题生成函数
        print(title_result)
        t2.insert('end', title_result, "tag_1")

#def new_title():
#    
#    user=e2.get(1.0, 'end')
#    print(user)
#    if user=='\n':
#        print('空的')
#        tkinter.messagebox.showerror(title='输入错误', message='输入为空！') 
#    else:
#        s1 = '夏天来临。'
#        title_result = gen_title(s1)  #调用标题生成函数
#        print(title_result)
#        t2.insert('end', title_result)

#def new_title():
#    if e2.get(1.0, 'end')=='\n':
#        print('空的')
#    else:
#        print('不是空的')

#def new_title():
#    user=e2.get(1.0, 'end')
#    print('0-----------------------------------------')
#    print(user)
#    print('1-----------------------------------------')
#    title_result=gen_title(user)
#    print('2-----------------------------------------')
#    print(title_result)
#    print('3-----------------------------------------')


start_button = tk.Button(window, text='生成标题', bg='grey', font=('StSong', 12), width=10, height=1, command=new_title)
start_button.pack()

l3 = tk.Label(window, text='标题生成结果：', font=('StSong', 12), width=800, height=2)
l3.pack()


t2 = tk.Text(window, width=800,height=3)
t2.pack()

t2.tag_config("tag_1", font=('StSong', 12), justify ="center")


#-----------------输入文本为空输入文本为空输入文本为空输入文本为空--------------------------------------------
#def hit_me():
#    tkinter.messagebox.showerror(title='输入错误', message='输入文本为空！')              # 提示信息对话窗
#
#tk.Button(window, text='hit me', bg='green', font=('Arial', 12), command=hit_me).pack()
#---------------------------------------------------------------------------------------------------------

def del_input():
    e2.delete(1.0, 'end')
    t2.delete(0.0, 'end')
 
l5 = tk.Label(window, text='点击按钮清空界面：', font=('StSong', 12), width=800, height=1)
l5.pack()    

clean_button  = tk.Button(window, text='清空界面', bg='yellow', font=('StSong', 12), width=10, height=1, command=del_input)
clean_button.pack()


window.mainloop()

