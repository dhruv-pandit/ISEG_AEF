# %%
from scipy.optimize import linprog
import pandas as pd
import tkinter as tk
from tkinter import ttk
import customtkinter as ck
from tkinter.messagebox import showerror, showwarning
import numpy as np

# %%
from tkinter import StringVar, IntVar, DoubleVar

class mainApp():
    global window, variables, constraints, rhs, obj_coef_lst, constraints_eq, rhs_eq
    variables = {}
    window = tk.Tk()
    window_width = 800
    window_height = 800
    window.title('Linear Programming Solver')
    global var_num
    var_num = 0
    # get the screen dimension
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    # find the center point
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)
    window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    window.resizable(False,False)
    constraints = np.array([[]])
    constraints_eq = np.array([[]])
    rhs=np.array([])
    rhs_eq=np.array([])

    obj_coef_lst = []
    variables = {}

    def __init__(self):
        pass
    def displayErrorMessage(self, str):
        showwarning(title='Error', message=str)
    def restart(self):
        frame7.forget()
        self.createMainWindow()
    def showresults(self):
        global obj_ceoef, obj_coef_lst, frame7
        temp = []
        for each in obj_ceoef:
            temp.append(each.get())
            print(each.get())
        temp = np.array(temp)
        if model_type.get() == 'Max':
            obj_coef_lst = np.append(obj_coef_lst,np.negative(temp))
        else:
            obj_coef_lst = np.append(obj_coef_lst,temp)

        #first we need the constraints array
        frame5.forget()
        frame7 = tk.Frame(window)

        if constraints_eq.size != 0:
            res  = linprog(A_ub = constraints, A_eq = constraints_eq,b_ub=rhs, b_eq = rhs_eq, c = obj_coef_lst, method = 'simplex')
            ttk.Label(frame7, text= ('Optimal value:', round(res.fun, ndigits=2), '\nx values:', res.x, '\nNumber of iterations performed:', res.nit, '\nStatus:', res.message)).pack()
        else:
            res  = res = linprog(obj_coef_lst, A_ub=constraints, b_ub=rhs)
            ttk.Label(frame7, text= ('Optimal value:', round(res.fun, ndigits=2), '\nx values:', res.x, '\nNumber of iterations performed:', res.nit, '\nStatus:', res.message)).pack()

        ck.CTkButton(frame7, text = 'Solve Another Model', command=self.restart).pack()
        ck.CTkButton(frame7, text = 'Quit', command=window.quit).pack()

        frame7.pack()
    
    def objectiveFunction(self):
        global frame5, obj_ceoef, obj_coef_lst, constraints, rhs
        temp = []
        for each in constraints_coeff:
            temp.append(each.get())
        temp = np.array([temp])
        if sign_type_var.get() == '>=':
            if constraints.size == 0:
                constraints = np.negative(temp)
                rhs = np.append(rhs, -rhs_var.get())
            else:
                constraints = np.vstack((constraints, np.negative(temp))) 
                rhs = np.append(rhs, -rhs_var.get())
        elif sign_type_var.get() == '=':
            if constraints_eq.size == 0:
                constraints_eq = np.negative(temp)
                rhs_eq = np.append(rhs_eq, -rhs_var.get())
            else:
                constraints_eq = np.vstack((constraints_eq, np.negative(temp))) 
                rhs_eq = np.append(rhs_eq, -rhs_var.get())
        else:
            if constraints.size == 0:
                constraints = temp
                rhs = np.append(rhs, rhs_var.get())
            else:
                constraints = np.vstack((constraints, temp)) 
                rhs = np.append(rhs, rhs_var.get())
        sign_type.append(sign_type_var.get())
        frame4.forget()

        frame5 = ttk.Frame(window)
        ck.CTkLabel(frame5, text = 'Objective').pack()
        obj_ceoef = [DoubleVar() for i in range(len(variables.keys()))]
        for each in range(len(variables.keys())):
            ck.CTkLabel(frame5, text=list(variables.keys())[each]).pack()
            tk.Entry(frame5, textvariable= obj_ceoef[each]).pack()
        ck.CTkButton(frame5, text = 'Solve', command = self.showresults).pack()
        frame5.pack()
        print(constraints)
        print(constraints.shape)
        print(rhs)
    def constraintFrameRe(self):
        global constraints, rhs, rhs_eq, constraints_eq
        temp = []
        for each in constraints_coeff:
            temp.append(each.get())
            print(each.get())
        print(temp)
        temp = np.array([temp])
        if sign_type_var.get() == '>=':
            if constraints.size == 0:
                constraints = np.negative(temp)
                rhs = np.append(rhs, -rhs_var.get())
            else:
                constraints = np.vstack((constraints, np.negative(temp))) 
                rhs = np.append(rhs, -rhs_var.get())
        elif sign_type_var.get() == '=':
            if constraints_eq.size == 0:
                constraints_eq = np.negative(temp)
                rhs_eq = np.append(rhs_eq, -rhs_var.get())
            else:
                constraints_eq = np.vstack((constraints_eq, np.negative(temp))) 
                rhs_eq = np.append(rhs_eq, -rhs_var.get())
        else:
            if constraints.size == 0:
                constraints = temp
                rhs = np.append(rhs, rhs_var.get())
            else:
                constraints = np.vstack((constraints, temp)) 
                rhs = np.append(rhs, rhs_var.get())
        sign_type.append(sign_type_var.get())
        print(constraints)
        print(sign_type)
        print(rhs)
        if sign_type_var.get() == '':
            self.displayErrorMessage('Please Select an Inequality/Equality')
        else:
            frame4.forget()
            print(sign_type_var.get())
            self.constraintFrame()
    
    def constraintFrame(self):
        global constraints, frame4, constraints_coeff, rhs, rhs_var, sign_type, sign_type_var
        try:
            variables[var_name.get()] = [None if var_low_bound.get().rstrip() == '' else float(var_low_bound.get()), None if var_high_bound.get().rstrip() == '' else float(var_high_bound.get())]
            print(variables)

            constraints_coeff = [DoubleVar() for i in range(len(variables.keys()))]
            rhs_var = DoubleVar()
            sign_type_var = StringVar()
            sign_type = []
            frame3.forget()
            frame4 = ttk.Frame(window)
            ck.CTkLabel(frame4, text = 'Add Constraint(s)').pack()
            for each in range(len(variables.keys())):
                ck.CTkLabel(frame4, text=list(variables.keys())[each]).pack()
                tk.Entry(frame4, textvariable= constraints_coeff[each]).pack()
            ck.CTkLabel(frame4, text='Enter the Right Hand Side Value').pack()
            tk.Entry(frame4, textvariable=rhs_var).pack()

            ck.CTkButton(frame4, text = 'Add Next Constraints', command = self.constraintFrameRe).pack()
            ck.CTkButton(frame4, text = 'Objective Function', command = self.objectiveFunction).pack()
            ck.CTkRadioButton(frame4, text = '>=', value='>=', variable=sign_type_var).pack()
            ck.CTkRadioButton(frame4, text = '<=', value='<=', variable=sign_type_var).pack()
            ck.CTkRadioButton(frame4, text = '=', value='=', variable=sign_type_var).pack()

            frame4.pack()
        except tk.TclError:
            self.displayErrorMessage('Please Enter a Decimal (Double/Float)')
    def frameInBetween(self):
        if var_name.get().rstrip() == '':
            self.displayErrorMessage('Please Give Your Variable A Name')
        elif var_name.get() in list(variables.keys()):
            self.displayErrorMessage("Please select another variable name that isn't already taken")
        else:
            
            self.constraintFrame()
    def varFrameRe(self):
        if var_name.get().rstrip() == '':
            self.displayErrorMessage('Please Give Your Variable A Name')
        elif var_name.get() in list(variables.keys()):
            self.displayErrorMessage("Please select another variable name that isn't already taken")
        else:
            variables[var_name.get()] = [None if var_low_bound.get().rstrip() == '' else float(var_low_bound.get()), None if var_high_bound.get().rstrip() == '' else float(var_high_bound.get())]
            frame3.forget()
            self.variableFrame()

    def variableFrame(self):
        frame1.forget()
        global frame3, var_name, var_low_bound, var_high_bound, variables
        var_name = ck.StringVar()
        var_nonneg = ck.IntVar()
        var_type = StringVar()
        var_low_bound = StringVar()
        var_high_bound = StringVar()
        frame3 = ttk.Frame(window)
        ttk.Label(frame3, text = 'Define Variable(s)').pack()
        ttk.Label(frame3, text = 'Preferably give your variable names in the form of X_i etc').pack()

        ttk.Entry(frame3, textvariable=var_name).pack()
        #Throw error if same name 
        ck.CTkCheckBox(frame3, text = 'Non Negative', onvalue=0, offvalue=1, variable= var_nonneg).pack()
        ttk.Label(frame3, text = 'Enter Low Bound (Type in None if no bound)').pack()
        ttk.Entry(frame3, textvariable=var_low_bound).pack()
        ttk.Label(frame3, text = 'Enter High Bound (Type in None if no bound)').pack()

        ttk.Entry(frame3,  textvariable=var_high_bound).pack()

        ck.CTkButton(frame3, text = 'Add Next Variable', command = self.varFrameRe).pack()
        ck.CTkButton(frame3, text = 'Add Constraints', command = self.frameInBetween).pack()
        
        frame3.pack()

    def createMainWindow(self):
        global frame1
        frame1 = ttk.Frame(window)
        #window.iconbitmap('./assets/pythontutorial.ico')
        #--------------------

        ttk.Label(frame1, text='Linear Programming Solver', font=('Futura', 20)).pack()
        ttk.Label(frame1, text='Sit consectetur anim commodo nisi culpa.', font=('Josefin Sans', 15)).pack()
        ttk.Label(frame1, text='Initialise Model', font=('Futura', 20)).pack()
        global model_type
        model_type = ck.StringVar()
        ck.CTkRadioButton(frame1, text = 'Max', value='Max', variable=model_type).pack()
        ck.CTkRadioButton(frame1, text = 'Min', value='Min', variable=model_type).pack()
        ck.CTkButton(frame1, text='Begin', command=self.variableFrame).pack()
        frame1.pack()
    def main(self):
        self.createMainWindow()
        window.mainloop()



# %%
m = mainApp()
m.main()


