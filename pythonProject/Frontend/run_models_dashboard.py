import os
import pickle
import tkinter

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo

import numpy as np
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import ImageTk
import pandas as pd

from pythonProject.Frontend.user_dashboard import UserDashboard

#todo Solve: File location
#todo saving name have a condition on the length
FOLDER_URL = 'saved_model'
file_path = []
LOCAL_URL = 'local_image'

class RunModelsDashboard(UserDashboard):
    def __init__(self, window):
        self.window = window
        windowWidth = self.window.winfo_reqwidth()
        windowHeight = self.window.winfo_reqheight()
        positionRight = int(self.window.winfo_screenwidth() / 6 - windowWidth / 2)
        positionDown = int(self.window.winfo_screenheight() / 5 - windowHeight / 2)
        self.window.geometry("+{}+{}".format(positionRight, positionDown))
        self.window.title("Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')
        self.set_frame()


    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=35, y=159)

        self.concatenate_frame = ImageTk.PhotoImage \
            (file='images\\run_models_frame.png')
        self.add_panel = Label(add_frame, image=self.concatenate_frame, bg="white")
        self.add_panel.pack(fill='both', expand='yes')

        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            comboLab = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            comboLab.configure(width=30)
            comboLab.place(x=114, y=250)

        self.entry_text_lab = StringVar()
        self.inFileTxt = Entry(self.window, textvariable=self.entry_text_lab)
        self.inFileTxt.configure(state="disabled")
        self.inFileTxt.place(x=113, y=360)

        self.add_file = ImageTk.PhotoImage \
            (file='images\\add_file_button_red.png')
        self.add_file_button_red = Button(self.window, image=self.add_file,
                                          font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                          , borderwidth=0, background="white", cursor="hand2",
                                          command=self.click_add_file)
        self.add_file_button_red.place(x=270, y=340)

        self.run_models_dashboard_cb = IntVar()
        checkbox = Checkbutton(self.window, text="Written Input?", variable=self.run_models_dashboard_cb, onvalue=1, offvalue=0, command=self.isChecked)
        checkbox.place(x=798, y=360)


        self.specific_value = StringVar()
        self.run_models_inFileTxt = Entry(self.window, textvariable=self.specific_value)
        self.run_models_inFileTxt.configure(state="disabled", width='27')
        self.run_models_inFileTxt.place(x=920, y=363)

        self.explanation_value = ['lime plot', 'Shap Bar Plot', 'Nothing']
        self.chosen_explanation_value = StringVar(self.window)
        self.combo_explanation_value = OptionMenu(self.window, self.chosen_explanation_value, *self.explanation_value)
        self.combo_explanation_value.configure(width=35)
        self.combo_explanation_value.place(x=115, y=495)

        self.explanation = ImageTk.PhotoImage \
            (file='images\\explanation_button_red.png')
        self.explanation_button = Button(self.window, image=self.explanation,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2",
                                         command=self.explain_local)
        self.explanation_button.place(x=1100, y=470)


    def click_add_file(self):
        data = [('All files', '*.*')]
        file = askopenfilename(filetypes=data, defaultextension=data,
                               title='Please select a file:')
        if len(file) != 0:
            file_path.append(file)
            file_name = os.path.basename(file)
            messagebox.showinfo("Selected File", "The added file is: \n {}".format(file))
            self.inFileTxt.configure(state="normal")
            self.entry_text_lab.set(file_name)
            self.inFileTxt.configure(state="disabled")

    def isChecked(self):
        if self.run_models_dashboard_cb.get() == 1:
            self.run_models_inFileTxt.configure(state="normal")
            self.add_file_button_red.configure(state='disabled')
        elif self.run_models_dashboard_cb.get() == 0:
            self.run_models_inFileTxt.configure(state="disabled")
            self.add_file_button_red.configure(state='normal')

        else:
            messagebox.showerror('PythonGuides', 'Something went wrong!')

    def explain_local(self):
        selectedFile = self.chosen_file.get()
        model = FOLDER_URL + '/'+selectedFile

        with open(model, 'rb') as f:
            training_type = pickle.load(f)

        X_train = np.load(FOLDER_URL +'/' + selectedFile[:-4] + '_train' + '.npy')
        label = np.load(FOLDER_URL +'/' + selectedFile[:-4] + '_label' + '.npy')
        # features = np.load(FOLDER_URL +'/' + selectedFile[:-4] + '_features' + '.npy')

        model_new = FOLDER_URL + '/'+ selectedFile[:-4] + '_features' + '.sav'
        with open(model_new, 'rb') as f:
            features = pickle.load(f)


        print(features)
        print(features.shape)
        # if self.run_models_dashboard_cb.get() == 1:
        #data = self.specific_value.get()
        sampleNumber = np.random.randint(low = 0,high = X_train.shape[1], size =1)[0]
        y_test = label[sampleNumber]
        X_test = X_train[sampleNumber].reshape(1, -1)

        # else:
        #     pass
        explanation_type = self.chosen_explanation_value.get()
        if explanation_type == 'Nothing':
            pass
        elif explanation_type == 'Shap Bar Plot':
            shapvalue = self.shap_dot_plot(training_type, X_test, y_test)
            shapArr = np.array(shapvalue)
            lime_values = [0] * (X_train.shape[1])
            ###################################################
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=features,
                class_names=['<6M', '6M-12M', '>12M'], verbose=True,
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_test[0],
                predict_fn=training_type.predict_proba
            )

            for v in exp.as_map()[1]:
                lime_values[v[0]] = v[1]

            ###################################################

            limeArr = np.array(lime_values)
            region = ''
            self.plotFig(region, shapArr, limeArr, features)

        elif explanation_type == 'lime plot':
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=features,
                class_names=['<6M', '6M-12M', '>12M'], verbose=True,
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_test[0],
                predict_fn=training_type.predict_proba
            )

            lime_values = [0] * (X_train.shape[1])
            for v in exp.as_map()[1]:
                lime_values[v[0]] = v[1]

            html_save = LOCAL_URL + '/' + selectedFile[:-4] + '.html'
            exp.save_to_file(html_save)

    def shap_dot_plot(self,training_type, X_train, i):
        expShap = shap.TreeExplainer(training_type)
        shap_values = expShap.shap_values(X_train)
        print(shap_values[i], type(shap_values))
        shapvalue = np.array(shap_values[i])
        shapvalue = shapvalue.reshape(-1)
        return shapvalue


    def plotFig(self, region, shapArr, limeArr, features):
        newF = []
        for f in features:
            # if f.endswith("(new)"):
            #     newF.append(f[0:-6])
            # else:
            newF.append(f)

        plt.rcParams['figure.figsize'] = (10, 8)  # set the figure size

        fig, ax = plt.subplots()

        width = 0.35

        ax.barh(np.arange(len(shapArr)) - width / 2, shapArr, height=width)
        ax.barh(np.arange(len(limeArr)) + width / 2, limeArr, height=width)
        ax.invert_yaxis()

        #    plt.ylabel('Prediction Factors', fontsize=12)
        plt.xlabel('Normalised Factor Contribution', fontsize=12)
        plt.title(region, fontsize=14)
        plt.legend(["SHAP", "LIME"], loc=4)
        plt.yticks(np.arange(len(shapArr)), newF, fontsize=12)

        plt.tight_layout()
        plt.show()

        if "<" in region:
            fileName = region[0:-5] + "_r_less_than_1.png"
        else:
            fileName = region[0:-6] + "_r_dec.png"
        fig.savefig(LOCAL_URL + fileName)
        print(LOCAL_URL + fileName)


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    RunModelsDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
