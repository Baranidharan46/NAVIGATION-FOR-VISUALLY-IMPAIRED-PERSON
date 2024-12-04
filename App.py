
from tkinter import *

import time,sys



def endprogram():
	print ("\nProgram terminated!")
	sys.exit()



def Camera1():
    import Predict





def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 500
    height = 500
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title("Obstacle Detection For Blind Person ")

    Label(text="Obstacle Detection For Blind Person", width="300", height="5", font=("Calibri", 16)).pack()
    Label(text="").pack()
    Label(text="").pack()
    Label(text="").pack()

    Button(text="Start", font=(
        'Verdana', 15), height="2", width="20", command=Camera1).pack(side=TOP)

    main_screen.mainloop()


main_account_screen()

