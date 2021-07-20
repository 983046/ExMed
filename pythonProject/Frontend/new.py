targetField = 'Survival Time'
Jam = mainAdd + "LungCancer_Base.csv"
Jam1 = mainAdd + "LungCancer_Encoded.csv"
ds = pd.read_csv(Jam1)
arrType = ['float', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int',
           'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'drop', 'drop', 'drop', 'bool',
           'int', 'cat']
ds = chnageDs(ds, arrType)
print(ds.shape)
ds = ds.fillna(0)
target = ds.pop(targetField)
print('---------------------')
ds1 = normalise_Dataset(ds)
ds1 = scale(ds1)
xOrg, yOrg = ds1, target
print('----', yOrg.value_counts(normalize=True) * 100)
x, y = smote(ds1, target)
print('---------------------')
n_components = 'mle'
# n_components = 3
(X_train, X_test, y_train, y_test) = split_data(x, y)

pcaCom = pca(X_train, n_components)
X_train = pcaCom[1]
X_test = pcaCom[2].transform(X_test)
print('---', X_test.shape, X_train.shape)

training_type = run_svm(
    X_train,
    X_test,
    y_train,
    y_test, saveResult=False, info='test')

y_pred = training_type.predict(X_test)
u, c = np.unique(y_test, return_counts=True)
print('=   =  = Balance : ', u, c)
cm = confusion_matrix(y_test, y_pred)
print('This is SVM confusion matrix: ')
print(cm)
print('Accuracy  Score : ', accuracy_score(y_test, y_pred))
print('precision Score : ', precision_score(y_test, y_pred, average=None))
print('recall_score Score : ', recall_score(y_test, y_pred, average=None))
print('---------------------')
print('---------------------')
print('---------------------')


def shap_dot_plot(training_type, X_train, i):
    expShap = shap.TreeExplainer(training_type)
    shap_values = expShap.shap_values(X_train)
    print(shap_values[i], type(shap_values))
    shap.summary_plot(shap_values[i], X_train, plot_type='dot', show=False)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(14)
    plt.tight_layout()
    plt.show()
    shapvalue = np.array(shap_values[i])
    shapvalue = shapvalue.reshape(-1)
    return shapvalue


def shap_bar_plot(training_type, X_train):
    expShap = shap.TreeExplainer(training_type)
    shap_values = expShap.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(14)
    plt.tight_layout()
    plt.show()
    print(shap_values[0], type(shap_values))


def shap_dependence_plot(training_type, X_train):
    expShap = shap.TreeExplainer(training_type)
    shap_values = expShap.shap_values(X_train)
    shap.dependence_plot(0, shap_values[0], X_train, show=False)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(14)
    plt.tight_layout()
    plt.show()


def lime_plot(training_type, X_train, X_test, features, file_name, good, bad):
    # columns=X_test.columns.values
    X_test = pd.DataFrame(X_test)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=features,
        class_names=[bad, good],
        mode='classification'
    )
    training_type = training_type(probability=True)

    exp = explainer.explain_instance(
        data_row=X_test.iloc[0],
        predict_fn=training_type.predict_proba
    )

    # window = tk.Toplevel()
    # render = ImageTk.PhotoImage(file = image_save)
    # img = tk.Label(window, image=render,bg="white")
    # img.pack(fill='both', expand='yes')


# In[113]:


Fan = mainAdd + "Cancer_Marcin.csv"
# Fan = mainAdd+"new_covid.csv"

targetField = 'Survival Time'
# targetField = 'Rt<1'

ds = pd.read_csv(Fan)

print(ds.columns)

ds = ds.fillna(0)
print(ds.shape)

# arrType =[ 'cat','int','int','float','float','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','cat','bool']
# ds = chnageDs(ds,arrType)
print(ds.shape)

target = ds.pop(targetField)
''' 
print('---------------------')
ds1 = normalise_Dataset(ds)
ds1 = scale(ds1)
xOrg , yOrg = ds1 , target
print('----',yOrg.value_counts(normalize=True) * 100)

print('---------------------')
n_components = 'mle'
#n_components = 3
(X_train, X_test, y_train, y_test) = split_data(x,y)


pcaCom = pca(X_train,n_components)
X_train  =  pcaCom[1]
X_test   =  pcaCom[2].transform(X_test)
print('---',X_test.shape, X_train.shape)


training_type = run_svm(
        X_train,
        X_test,
        y_train,
        y_test , saveResult = False , info = 'test'   )


y_pred = training_type.predict(X_test)
u, c = np.unique(y_test, return_counts=True)
print('=   =  = Balance : ',u,c)
cm = confusion_matrix(y_test, y_pred)
print('This is SVM confusion matrix: ')
print(cm)
print('Accuracy  Score : ',accuracy_score(y_test, y_pred))
print('precision Score : ',precision_score(y_test, y_pred, average=None))
print('recall_score Score : ',recall_score(y_test, y_pred, average=None))
print('---------------------')
print('---------------------')
print('---------------------')
'''

'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(512,1024,512), random_state=1,max_iter=1500)
(X_train, X_test, y_train, y_test) = split_data(x,y)
clf.fit(X_train, y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test)
clf.score(X_test, y_test)
print('Accuracy  Score : ',accuracy_score(y_test, y_pred))
print('precision Score : ',precision_score(y_test, y_pred, average=None))
print('recall_score Score : ',recall_score(y_test, y_pred, average=None))
confusion_matrix(y_test, y_pred)

m1 = 'Covid_MLPRegressor.sav'
m2 = 'Covid_SVM.sav'
m3 = 'Covid_TreeRegression.sav'
m4 = 'Covid_XGBoost.sav'
'''

m33 = 'Lung_cancer_regression.sav'
savedModel = mainAdd + m33
'''
print(savedModel)

with open(savedModel, 'rb') as fid:
    #loaded_model = cPickle.load(fid)
      training_type  = pickle.load(fid)
'''
with open(savedModel, 'rb') as f:
    training_type = pickle.load(f)

X_train = np.load(mainAdd + 'Lung_cancer_regression_train.npy')

print(X_train.shape)
y_test = np.array([[1]])
y_pred = training_type.predict(X_train)
''' 
training_type.score(X_test, y_test)
print('Accuracy  Score : ',accuracy_score(y_test, y_pred))
print('precision Score : ',precision_score(y_test, y_pred, average=None))
print('recall_score Score : ',recall_score(y_test, y_pred, average=None))
'''

# In[150]:


X_test = X_train[259].reshape(1, -1)


X_test = np.array([[39, 1, 0, 8046, 65, 1.62, 0, 4, 0, 1, 1, 0, 2, 3, 1, 1, 1201, 2, 9, 99]])
y_test = training_type.predict(X_test)
print(X_test, y_test)

# In[151]:


import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=ds.columns,
    class_names=['<6M', '6M-12M', '>12M'], verbose=True,
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=X_test[0],
    predict_fn=training_type.predict_proba
)
# exp.show_in_notebook(show_table=True, show_all=False)

# LIME stuff

lime_values = [0] * (X_train.shape[1])
for v in exp.as_map()[1]:
    lime_values[v[0]] = v[1]

# In[157]:


print(exp.as_map()[1])

# In[175]:


# SHAP
shapvalue = shap_dot_plot(training_type, X_test, 1)
print(ds.columns)


# In[168]:


# In[178]:


def plotFig(region, shapArr, limeArr, features):
    newF = []
    for f in features:
        if f.endswith("(new)"):
            newF.append(f[0:-6])
        else:
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
    fig.savefig(mainAdd + fileName)
    print(mainAdd + fileName)


shapArr = np.array(shapvalue)
limeArr = np.array(lime_values)
features = ds.columns
region = ''
plotFig(region, shapArr, limeArr, features)

# In[90]:


lime_values = [0] * (X_train.shape[1])
for v in exp.as_map()[1]:
    lime_values[v[0]] = v[1]
print(lime_values)

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ds.columns
y_pos = np.arange(len(people))
performance = np.array(lime_values)

mask1 = performance < 0.0
mask2 = performance >= 0.0

ax.barh(y_pos[mask1], performance[mask1], align='center', color='#FA7F0A')
ax.barh(y_pos[mask2], performance[mask2], align='center', color='blue')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
plt.show()

# In[125]:


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ds.columns
y_pos = np.arange(len(people))
performance = np.array(
    [0.06373924, 0.00279475, 0.01429565, 0.05894891, 0.06036747, 0.0957469, 0.00486897, 0.01507216, -0.00082485,
     0.02699536, 0.05291194, 0., 0.04472349, 0.06945249, 0.09925872, 0.01424324, 0.03017553, 0.0069287, 0.01052711,
     -0.00223212])

mask1 = performance < 0.0
mask2 = performance >= 0.0

ax.barh(y_pos[mask1], performance[mask1], align='center', color='#FA7F0A')
ax.barh(y_pos[mask2], performance[mask2], align='center', color='blue')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
plt.show()

# In[ ]:


print(ds.columns)

# In[ ]:


shap_bar_plot(training_type, X_test)

# shap_bar_plot( training_type, X_test)

#


